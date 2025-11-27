"""
Train Hybrid Transformer-TCS Model

This script trains a hybrid model in JOINT mode, training both:
1. GPT-style Transformer for sequence modeling and attention-based learning
2. TCS Variational Autoencoder for semantic plausibility constraints

The model learns to generate fluent text while respecting semantic structure.
Both models are trained jointly end-to-end.

For separate training:
- Use tcs_transformer.scripts.transformer_scripts.train_transformer for transformer-only
- Use tcs_transformer.scripts.tcs_scripts.train for TCS VAE-only
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

from tcs_transformer.models.hybrid_model import (
    HybridTransformerTCS,
    create_hybrid_model,
)
from tcs_transformer.scripts.transformer_scripts.train_transformer import (
    prepare_data_and_tokenizer,
    make_batch,
)
from tcs_transformer.utils.token_to_tcs import create_converter, TokenToTCSConverter


def load_tcs_vae(pretrained_path: str, device: str):
    """
    Load a pretrained TCS VAE from checkpoint

    Args:
        pretrained_path: Path to TCS VAE checkpoint
        device: Device to load on

    Returns:
        Loaded TCS VAE model
    """
    # TODO: Implement actual TCS VAE loading
    # This is a placeholder for when TCS VAE loading is implemented
    print(f"Loading TCS VAE from: {pretrained_path}")
    print("WARNING: TCS VAE loading not implemented yet")
    return None


def setup_hybrid_model(
    vocab_size: int,
    tcs_vae: Optional[Any],
    n_layers: int,
    n_heads: int,
    d_model: int,
    d_ff: int,
    block_size: int,
    semantic_weight: float,
    device: str,
    verbose: bool = True,
) -> HybridTransformerTCS:
    """
    Create and configure hybrid model for joint training

    Args:
        vocab_size: Vocabulary size
        tcs_vae: Pretrained TCS VAE (optional)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward dimension
        block_size: Context window size
        semantic_weight: Weight for semantic loss
        device: Device to use
        verbose: Print model info

    Returns:
        Configured hybrid model (both components trainable)
    """
    if verbose:
        print("\nConfiguring hybrid model for JOINT training...")

    # Create model
    model = create_hybrid_model(
        vocab_size=vocab_size,
        tcs_vae=tcs_vae,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=block_size,
        semantic_weight=semantic_weight,
        device=device,
    )

    # Print parameter counts
    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        transformer_params = sum(
            p.numel() for p in model.transformer.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Transformer parameters: {transformer_params:,}")
        if tcs_vae:
            tcs_params = sum(
                p.numel() for p in model.tcs_vae.parameters() if p.requires_grad
            )
            print(f"TCS VAE parameters: {tcs_params:,}")

    # Ensure both components are trainable (joint mode only)
    model.unfreeze_all()

    return model


def setup_optimizer(
    model: nn.Module,
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Setup AdamW optimizer for hybrid model

    Args:
        model: Hybrid model
        learning_rate: Learning rate
        betas: Adam beta parameters
        weight_decay: Weight decay coefficient

    Returns:
        Configured optimizer
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )
    return optimizer


def train_step(
    model: HybridTransformerTCS,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    tcs_data: Optional[Dict[str, Any]],
    grad_clip: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform a single training step

    Args:
        model: Hybrid model
        optimizer: Optimizer
        x: Input token IDs
        y: Target token IDs
        tcs_data: TCS encoder/decoder data (optional)
        grad_clip: Gradient clipping value

    Returns:
        loss: Total loss
        loss_dict: Dictionary with loss components
    """
    # Forward pass
    logits, loss, loss_dict = model(x, targets=y, tcs_data=tcs_data)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    return loss, loss_dict


def evaluate_step(
    model: HybridTransformerTCS,
    x: torch.Tensor,
    y: torch.Tensor,
    tcs_data: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Perform a single evaluation step

    Args:
        model: Hybrid model
        x: Input token IDs
        y: Target token IDs
        tcs_data: TCS encoder/decoder data (optional)

    Returns:
        loss: Total loss
        loss_dict: Dictionary with loss components
    """
    model.eval()
    with torch.no_grad():
        _, loss, loss_dict = model(x, targets=y, tcs_data=tcs_data)
    model.train()

    return loss, loss_dict


def format_log_message(
    step: int,
    train_loss: float,
    val_loss: float,
    loss_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    """
    Format training log message for joint training

    Args:
        step: Current training step
        train_loss: Training loss
        val_loss: Validation loss
        loss_dict: Dictionary with loss components

    Returns:
        Formatted log string
    """
    msg = f"step {step:4d} | train {train_loss:.3f} | val {val_loss:.3f}"

    if loss_dict:
        lm_loss = float(loss_dict.get("lm_loss", 0.0))
        sem_loss = float(loss_dict.get("semantic_loss", 0.0))
        if sem_loss > 0:
            msg += f" | lm {lm_loss:.3f} | sem {sem_loss:.3f}"

    return msg


def save_checkpoint(
    save_path: str,
    model: HybridTransformerTCS,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    tokenizer_model: str,
    train_history: List[Tuple],
    semantic_weight: float,
):
    """
    Save model checkpoint for joint training

    Args:
        save_path: Path to save checkpoint
        model: Hybrid model
        optimizer: Optimizer
        tokenizer: Tokenizer
        tokenizer_model: Tokenizer model name
        train_history: Training history
        semantic_weight: Semantic loss weight
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.cfg,
        "train_history": train_history,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_model": tokenizer_model,
        "use_tcs": True,  # Always True for joint training
        "semantic_weight": semantic_weight,
    }

    torch.save(checkpoint, save_path)
    print(f"\nModel saved to: {save_path}")


def run_training_loop(
    model: HybridTransformerTCS,
    optimizer: torch.optim.Optimizer,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    tokenizer,
    block_size: int,
    batch_size: int,
    num_steps: int,
    eval_every: int,
    grad_clip: Optional[float],
    device: str,
    semantic_weight: float,
    warmup_steps: int,
    tcs_converter: Optional[TokenToTCSConverter],
    tcs_convert_every: int = 10,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Main training loop for hybrid model (joint training only)

    Args:
        model: Hybrid model
        optimizer: Optimizer
        train_ids: Training token IDs
        val_ids: Validation token IDs
        tokenizer: Tokenizer for decoding tokens
        block_size: Context window size
        batch_size: Batch size
        num_steps: Number of training steps
        eval_every: Evaluate every N steps
        grad_clip: Gradient clipping value
        device: Device to use
        semantic_weight: Weight for semantic loss
        warmup_steps: Steps to train transformer-only before adding TCS
        tcs_converter: TokenToTCSConverter instance (required for joint training)
        tcs_convert_every: Convert tokens to TCS every N steps (parsing is slow)

    Returns:
        Training history list
    """
    train_history = []
    model.train()

    # Validate TCS converter is provided
    if tcs_converter is None:
        raise ValueError(
            "TCS converter is required for joint training. Cannot proceed."
        )

    for step in range(1, num_steps + 1):
        # Warmup: disable TCS for initial steps
        if step <= warmup_steps:
            current_semantic_weight = 0.0
        else:
            current_semantic_weight = semantic_weight

        if step == warmup_steps + 1 and warmup_steps > 0:
            print(
                f"\nWarmup complete! Enabling TCS semantic loss (λ={semantic_weight})"
            )
            print("=" * 80)

        # Get batch
        x, y = make_batch(train_ids, block_size, batch_size)
        x = x.to(device)
        y = y.to(device)

        # Convert tokens to TCS data (always, after warmup)
        tcs_data = None
        if step > warmup_steps:
            # Only convert periodically to avoid slowing down training too much
            if step % tcs_convert_every == 0:
                try:
                    # Convert first sequence in batch to TCS format
                    tcs_sample = tcs_converter.convert_tokens(
                        y[0], tokenizer, sample_id=f"step_{step}"
                    )
                    if tcs_sample:
                        tcs_data = tcs_sample

                    # Print conversion statistics occasionally
                    if step % (eval_every * 5) == 0:
                        stats = tcs_converter.get_statistics()
                        success_rate = (
                            stats["successes"] / max(stats["total_attempts"], 1) * 100
                        )
                        print(
                            f"  TCS conversion: {success_rate:.1f}% success rate "
                            f"({stats['successes']}/{stats['total_attempts']} attempts)"
                        )
                except Exception as e:
                    print(f"  TCS conversion error at step {step}: {e}")

        # Training step
        loss, loss_dict = train_step(model, optimizer, x, y, tcs_data, grad_clip)

        # Evaluation
        if step % eval_every == 0 or step == 1:
            vx, vy = make_batch(val_ids, block_size, batch_size)
            vx = vx.to(device)
            vy = vy.to(device)

            # Convert validation tokens to TCS
            val_tcs_data = None
            if step > warmup_steps:
                try:
                    val_tcs_sample = tcs_converter.convert_tokens(
                        vy[0], tokenizer, sample_id=f"val_step_{step}"
                    )
                    if val_tcs_sample:
                        val_tcs_data = val_tcs_sample
                except:
                    pass

            vloss, vloss_dict = evaluate_step(model, vx, vy, val_tcs_data)

            train_loss_val = float(loss)
            val_loss_val = float(vloss)
            lm_loss_val = float(loss_dict["lm_loss"]) if loss_dict else 0.0
            sem_loss_val = float(loss_dict["semantic_loss"]) if loss_dict else 0.0

            train_history.append(
                (step, train_loss_val, val_loss_val, lm_loss_val, sem_loss_val)
            )

            # Print log
            log_msg = format_log_message(step, train_loss_val, val_loss_val, loss_dict)
            print(log_msg)

    return train_history


def train_hybrid_model(
    # Data parameters
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer_model="google/gemma-2-2b-it",
    # Transformer architecture
    n_layers=4,
    n_heads=4,
    d_model=256,
    d_ff=1024,
    block_size=128,
    # Training parameters
    batch_size=12,
    learning_rate=3e-4,
    num_steps=2000,
    eval_every=200,
    grad_clip=1.0,
    # Hybrid-specific parameters (JOINT TRAINING ONLY)
    semantic_weight=0.1,  # λ - weight for semantic loss
    pretrained_tcs_path=None,  # Path to pretrained TCS VAE checkpoint (required)
    warmup_steps=0,  # Number of steps to train transformer-only before adding TCS
    # Save/load
    save_path=None,
    device=None,
):
    """
    Train a hybrid Transformer-TCS model in JOINT mode

    This function trains both the transformer and TCS VAE together end-to-end.
    Tokens are decoded, parsed to DMRS, and transformed to TCS format on-the-fly
    to provide semantic supervision to both models.

    Note: ERG parsing is slow (~0.5-2s per sentence). Training will be slower.
    Consider using warmup_steps to train transformer-only first, then enable TCS.

    For separate model training:
        - Transformer-only: Use tcs_transformer.scripts.transformer_scripts.train_transformer
        - TCS VAE-only: Use tcs_transformer.scripts.tcs_scripts.train

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        tokenizer_model: Pretrained tokenizer to use
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward dimension
        block_size: Context window size
        batch_size: Batch size
        learning_rate: Learning rate
        num_steps: Number of training steps
        eval_every: Evaluate every N steps
        grad_clip: Gradient clipping value
        semantic_weight: Weight for semantic loss (λ)
        pretrained_tcs_path: Path to pretrained TCS VAE checkpoint (required for joint training)
        warmup_steps: Steps to train transformer-only before enabling TCS (0 = enable TCS from start)
        save_path: Path to save model checkpoint
        device: Device to train on (None = auto-detect)

    Returns:
        model: Trained HybridTransformerTCS (both models trained jointly)
        tokenizer: The tokenizer used
        train_history: List of (step, train_loss, val_loss, lm_loss, sem_loss) tuples
    """
    print("=" * 80)
    print("Hybrid Transformer-TCS Training (JOINT MODE)")
    print("=" * 80)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Semantic weight (λ): {semantic_weight}")
    if warmup_steps > 0:
        print(f"Warmup: {warmup_steps} steps (transformer-only), then TCS enabled")
    else:
        print("TCS enabled from start")
    print("WARNING: TCS parsing is slow. Training speed will be affected.")
    print(
        f"Recommended: Use warmup_steps={num_steps // 2} for faster initial training."
    )

    # -------------------------
    # 1) Prepare data and tokenizer
    # -------------------------
    train_ids, val_ids, tokenizer = prepare_data_and_tokenizer(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        tokenizer_model=tokenizer_model,
        verbose=True,
    )

    # -------------------------
    # 2) Load or initialize TCS VAE (REQUIRED for joint training)
    # -------------------------
    tcs_vae = None
    if pretrained_tcs_path:
        tcs_vae = load_tcs_vae(pretrained_tcs_path, device)
        if tcs_vae is None:
            print("ERROR: TCS VAE loading failed from:", pretrained_tcs_path)
            print("Cannot proceed with joint training without TCS VAE.")
            print(
                "Please provide a valid pretrained_tcs_path or train TCS separately first."
            )
            raise ValueError("TCS VAE loading failed - required for joint training")
    else:
        print("\nWARNING: No pretrained TCS model provided (pretrained_tcs_path=None)")
        print(
            "Joint training requires a TCS VAE. Initializing new TCS VAE from scratch."
        )
        print(
            "Note: Training from scratch may not converge well. Recommended to pretrain TCS first."
        )
        # Initialize new TCS VAE (will be trained from scratch)
        # TODO: Add TCS VAE initialization here if needed

    # -------------------------
    # 3) Setup hybrid model
    # -------------------------
    model = setup_hybrid_model(
        vocab_size=tokenizer.vocab_size,
        tcs_vae=tcs_vae,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        block_size=block_size,
        semantic_weight=semantic_weight,
        device=device,
        verbose=True,
    )

    # -------------------------
    # 4) Setup optimizer
    # -------------------------
    print(f"\nSetup optimizer (lr={learning_rate})")
    optimizer = setup_optimizer(model, learning_rate)

    # -------------------------
    # 5) Training loop
    # -------------------------
    print(f"\nTraining for {num_steps} steps...")
    print(f"Block size: {block_size}, Batch size: {batch_size}")
    if warmup_steps > 0:
        print(f"Warmup: {warmup_steps} steps (transformer-only)")
    print("=" * 80)

    # -------------------------
    # 5b) Setup TCS converter (ALWAYS, for joint training)
    # -------------------------
    tcs_converter = None
    if tcs_vae:
        print("\nInitializing TCS converter...")
        try:
            tcs_converter = create_converter(
                erg_path="examples/erg/erg-1214-x86-64-0.9.34.dat",
                unk2pos_path="examples/erg/unk2pos.json",
                transform_config_path="configs/transform_config.json",
                vocab_dir=None,  # TODO: Load pre-computed vocabularies if available
            )
            print("TCS converter initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize TCS converter: {e}")
            print("Cannot proceed with joint training without TCS converter.")
            raise RuntimeError(f"TCS converter initialization failed: {e}")
    else:
        print("WARNING: No TCS VAE provided - cannot perform joint training")
        print("Model will only have transformer component.")

    train_history = run_training_loop(
        model=model,
        optimizer=optimizer,
        train_ids=train_ids,
        val_ids=val_ids,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        num_steps=num_steps,
        eval_every=eval_every,
        grad_clip=grad_clip,
        device=device,
        semantic_weight=semantic_weight,
        warmup_steps=warmup_steps,
        tcs_converter=tcs_converter,
    )

    print("=" * 80)
    print("Training complete!")

    # -------------------------
    # 6) Save model if requested
    # -------------------------
    if save_path:
        save_checkpoint(
            save_path=save_path,
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            tokenizer_model=tokenizer_model,
            train_history=train_history,
            semantic_weight=semantic_weight,
        )

    return model, tokenizer, train_history


def load_hybrid_checkpoint(checkpoint_path, device=None):
    """
    Load a hybrid model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (None = auto-detect)

    Returns:
        model: Loaded HybridTransformerTCS
        tokenizer: The tokenizer
        train_history: Training history
    """
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    cfg.device = device

    # Recreate model (without TCS for now)
    model = HybridTransformerTCS(cfg, tcs_vae=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer_model"])
    train_history = checkpoint.get("train_history", [])

    print(f"Loaded hybrid model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Vocab size: {cfg.vocab_size}")
    print(f"Used TCS: {checkpoint.get('use_tcs', False)}")

    return model, tokenizer, train_history


def generate_text_hybrid(
    model,
    tokenizer,
    prompt="Hello, how are you",
    max_new_tokens=100,
    temperature=0.8,
    device=None,
):
    """
    Generate text using a trained hybrid model

    Args:
        model: Trained HybridTransformerTCS
        tokenizer: HuggingFace tokenizer
        prompt: Starting text prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        device: Device to use

    Returns:
        Generated text string
    """
    if device is None:
        device = model.cfg.device

    model.eval()

    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(
        0
    )

    # Generate
    with torch.no_grad():
        out_ids = model.generate(
            prompt_tensor, max_new_tokens=max_new_tokens, temperature=temperature
        )

    # Decode
    decoded = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)

    return decoded
