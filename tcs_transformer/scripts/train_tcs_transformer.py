"""
Train Transformer on TCS Data

This module provides functions to train a transformer model on TCS (Type-driven
Compositional Semantics) data. The model treats structured semantic data as
sequences suitable for transformer processing.

The key difference from VarAutoencoder:
- VarAutoencoder: Encodes predicate-argument structure → latent space → decodes to semantic functions
- This Transformer: Flattens predicate-argument structure into sequences → predicts semantic function indices

Data Conversion Strategy:
- Input: pred_func_nodes_ctxt_predargs (list of lists of predicate-argument indices)
- Convert to: Flattened sequence with special separator tokens
- Output: Predict semantic function indices for each node
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm
from pathlib import Path

# Import model components
from tcs_transformer.models.tcs_transformer_model import (
    TCSTransformer,
    TCSTransformerConfig,
)
from tcs_transformer.data.dataset import TCSTransformerDataset
from tcs_transformer.data.collators import TCSTransformerCollator


def load_vocab_info(transformed_dir: str) -> Tuple[int, int]:
    """
    Load vocabulary information from transformed data directory

    Args:
        transformed_dir: Path to transformed data directory

    Returns:
        vocab_size: Size of predicate-argument vocabulary
        num_sem_funcs: Number of semantic functions
    """
    transformed_info_dir = os.path.join(transformed_dir, "info")

    # Load predicate-argument vocabulary
    with open(os.path.join(transformed_info_dir, "pred2ix.txt"), "r") as f:
        pred2ix = (
            json.load(f) if transformed_info_dir.endswith(".json") else eval(f.read())
        )

    vocab_size = len(pred2ix) + 10  # Add buffer for special tokens

    # Load semantic functions vocabulary
    with open(os.path.join(transformed_info_dir, "pred_func2ix.txt"), "r") as f:
        pred_func2ix = (
            json.load(f) if transformed_info_dir.endswith(".json") else eval(f.read())
        )
    num_sem_funcs = len(pred_func2ix)

    return vocab_size, num_sem_funcs


def setup_model(
    vocab_size: int,
    num_sem_funcs: int,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 1024,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
) -> TCSTransformer:
    """
    Create and configure TCS transformer model

    Args:
        vocab_size: Vocabulary size
        num_sem_funcs: Number of semantic functions to predict
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        device: Device to use
        verbose: Print model info

    Returns:
        Configured TCS transformer model
    """
    if verbose:
        print("\nConfiguring TCS Transformer model...")

    # Create model configuration
    model_cfg = TCSTransformerConfig(
        vocab_size=vocab_size,
        num_sem_funcs=num_sem_funcs,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        device=device,
    )

    # Create model
    model = TCSTransformer(model_cfg).to(device)

    # Print parameter counts
    if verbose:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params:,}")

    return model


def setup_optimizer(
    model: nn.Module,
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Setup Adam optimizer for TCS transformer

    Args:
        model: TCS transformer model
        learning_rate: Learning rate
        betas: Adam beta parameters
        weight_decay: Weight decay coefficient

    Returns:
        Configured optimizer
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
    )
    return optimizer


def train_step(
    model: TCSTransformer,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    device: str,
    grad_clip: Optional[float] = None,
) -> torch.Tensor:
    """
    Perform a single training step

    Args:
        model: TCS transformer model
        optimizer: Optimizer
        batch: Batch of data from dataloader
        device: Device to use
        grad_clip: Gradient clipping value

    Returns:
        loss: Training loss
    """
    input_ids = batch["input_ids"].to(device)
    node_boundaries = batch["node_boundaries"]
    targets = batch["targets"].to(device)

    # Forward pass
    logits = model(input_ids, node_boundaries)  # [B, num_nodes, num_sem_funcs]

    # Compute loss (cross-entropy over semantic functions)
    B, num_nodes, num_classes = logits.shape
    loss = F.cross_entropy(
        logits.view(-1, num_classes),
        targets.view(-1),
        ignore_index=0,  # Ignore padding
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss


def train_epoch(
    model: TCSTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: Optional[float] = 1.0,
) -> float:
    """
    Train for one epoch

    Args:
        model: TCS transformer model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        grad_clip: Gradient clipping value

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        loss = train_step(model, optimizer, batch, device, grad_clip)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def save_checkpoint(
    save_path: str,
    model: TCSTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    avg_loss: float,
):
    """
    Save model checkpoint

    Args:
        save_path: Path to save checkpoint
        model: TCS transformer model
        optimizer: Optimizer
        epoch: Current epoch number
        avg_loss: Average loss for the epoch
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "config": model.cfg.__dict__,
    }

    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def train_tcs_transformer(
    # Data parameters
    config_path: str = "configs/config.json",
    # Model architecture
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 1024,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    # Training parameters
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    grad_clip: float = 1.0,
    # Save/load
    save_dir: str = "saved/tcs_transformer",
    seed: int = 42,
    device: str = None,
) -> Tuple[TCSTransformer, List[Tuple[int, float]]]:
    """
    Train a TCS transformer model

    This function trains a transformer on TCS data, treating structured semantic
    predicate-argument data as flattened sequences with separator tokens.

    Args:
        config_path: Path to TCS config file (to extract data paths and vocab info)
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size (should be 1 for TCS data)
        grad_clip: Gradient clipping value
        save_dir: Directory to save checkpoints
        seed: Random seed
        device: Device to train on (None = auto-detect)

    Returns:
        model: Trained TCS transformer model
        train_history: List of (epoch, avg_loss) tuples
    """
    print("=" * 80)
    print("TCS Transformer Training")
    print("=" * 80)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # -------------------------
    # 1) Load config and prepare data
    # -------------------------
    with open(config_path, "r") as f:
        config = json.load(f)

    transformed_dir = config["data_loader"]["args"]["transformed_dir"]
    print(f"Loading data from: {transformed_dir}")

    # Load vocabulary information
    vocab_size, num_sem_funcs = load_vocab_info(transformed_dir)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of semantic functions: {num_sem_funcs}")

    # -------------------------
    # 2) Setup model
    # -------------------------
    model = setup_model(
        vocab_size=vocab_size,
        num_sem_funcs=num_sem_funcs,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        device=device,
        verbose=True,
    )

    # -------------------------
    # 3) Setup data
    # -------------------------
    print("\nLoading dataset...")
    dataset = TCSTransformerDataset(transformed_dir)
    collator = TCSTransformerCollator(sep_token_id=1, pad_token_id=0)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0
    )

    # -------------------------
    # 4) Setup optimizer
    # -------------------------
    print(f"\nSetup optimizer (lr={learning_rate})")
    optimizer = setup_optimizer(model, learning_rate)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    # 5) Training loop
    # -------------------------
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    train_history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch, grad_clip)
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        train_history.append((epoch, avg_loss))

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
        save_checkpoint(checkpoint_path, model, optimizer, epoch, avg_loss)

    print("=" * 80)
    print("Training completed!")
    print("=" * 80)

    return model, train_history


def load_checkpoint(
    checkpoint_path: str, device: str = None
) -> Tuple[TCSTransformer, Dict[str, Any]]:
    """
    Load a TCS transformer model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (None = auto-detect)

    Returns:
        model: Loaded TCS transformer model
        checkpoint: Full checkpoint dictionary
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate config
    cfg_dict = checkpoint["config"]
    cfg_dict["device"] = device
    cfg = TCSTransformerConfig(**cfg_dict)

    # Recreate model
    model = TCSTransformer(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Loss: {checkpoint['loss']:.4f}")

    return model, checkpoint
