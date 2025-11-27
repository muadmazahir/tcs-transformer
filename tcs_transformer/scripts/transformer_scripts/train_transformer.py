"""Train GPT-style decoder-only Transformer on text data"""

import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path

from tcs_transformer.models.transformer_model import GPTConfig, DecoderOnlyTransformer


def make_batch(
    ids: torch.Tensor, block_size: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch of training data from token IDs

    Args:
        ids: Tensor of token IDs
        block_size: Context window size
        batch_size: Number of samples per batch

    Returns:
        x: Input sequences (batch_size, block_size)
        y: Target sequences (batch_size, block_size)
    """
    assert ids.numel() > block_size + 1, (
        f"Token IDs ({ids.numel()}) must be > block_size + 1 ({block_size + 1})"
    )
    ix = torch.randint(0, ids.numel() - block_size - 1, (batch_size,))
    x = torch.stack([ids[i : i + block_size] for i in ix])
    y = torch.stack([ids[i + 1 : i + block_size + 1] for i in ix])
    return x, y


def encode_corpus(texts, tokenizer):
    """
    Encode a corpus of texts into a single stream of token IDs

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer

    Returns:
        Tensor of token IDs
    """
    ids = []
    for t in texts:
        ids.extend(tokenizer.encode(t))
    return torch.tensor(ids, dtype=torch.long)


def prepare_data_and_tokenizer(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer_model="google/gemma-2-2b-it",
    verbose=True,
):
    """
    Load dataset, tokenizer, and encode corpus into token IDs

    Args:
        dataset_name: HuggingFace dataset name (default: "wikitext")
        dataset_config: Dataset configuration (default: "wikitext-2-raw-v1")
        tokenizer_model: Pretrained tokenizer to use (default: "google/gemma-2-2b-it")
        verbose: Print progress information (default: True)

    Returns:
        train_ids: Encoded training token IDs (torch.Tensor)
        val_ids: Encoded validation token IDs (torch.Tensor)
        tokenizer: HuggingFace tokenizer
    """
    # Import dataset loaders
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "Required packages not found. Please install:\n"
            "  pip install datasets transformers"
        )

    # Load dataset
    if verbose:
        print(f"Loading dataset: {dataset_name} ({dataset_config})")
    ds = load_dataset(dataset_name, dataset_config)
    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]

    # Filter out empty lines
    train_texts = [t for t in train_texts if t and not t.isspace()]
    val_texts = [t for t in val_texts if t and not t.isspace()]

    if verbose:
        print(f"Train texts: {len(train_texts)}")
        print(f"Val texts: {len(val_texts)}")

    # Load tokenizer
    if verbose:
        print(f"\nLoading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    if verbose:
        print(f"Vocab size: {tokenizer.vocab_size}")

    # Encode corpus
    if verbose:
        print("\nEncoding corpus...")
    train_ids = encode_corpus(train_texts, tokenizer)
    val_ids = encode_corpus(val_texts, tokenizer)

    if verbose:
        print(f"Train tokens: {train_ids.numel():,}")
        print(f"Val tokens: {val_ids.numel():,}")

    return train_ids, val_ids, tokenizer


def train_transformer(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer_model="google/gemma-2-2b-it",
    n_layers=4,
    n_heads=4,
    d_model=256,
    d_ff=1024,
    block_size=128,
    batch_size=12,
    learning_rate=3e-4,
    num_steps=2000,
    eval_every=200,
    grad_clip=1.0,
    save_path=None,
    device=None,
):
    """
    Train a GPT-style decoder-only Transformer on text data

    Args:
        dataset_name: HuggingFace dataset name (default: "wikitext")
        dataset_config: Dataset configuration (default: "wikitext-2-raw-v1")
        tokenizer_model: Pretrained tokenizer to use (default: "google/gemma-2-2b-it")
        n_layers: Number of transformer layers (default: 4)
        n_heads: Number of attention heads (default: 4)
        d_model: Model dimension (default: 256)
        d_ff: Feed-forward dimension (default: 1024)
        block_size: Context window size (default: 128)
        batch_size: Batch size (default: 12)
        learning_rate: Learning rate (default: 3e-4)
        num_steps: Number of training steps (default: 2000)
        eval_every: Evaluate every N steps (default: 200)
        grad_clip: Gradient clipping value (default: 1.0)
        save_path: Path to save model checkpoint (default: None = don't save)
        device: Device to train on (default: None = auto-detect)

    Returns:
        model: Trained DecoderOnlyTransformer
        tokenizer: The tokenizer used
        train_history: List of (step, train_loss, val_loss) tuples
    """
    print("=" * 80)
    print("GPT-Style Transformer Training")
    print("=" * 80)

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    # 2) Configure model
    # -------------------------
    print("\nConfiguring model...")
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=block_size,
        attn_dropout=0.1,
        resid_dropout=0.1,
        emb_dropout=0.1,
        device=device,
    )

    model = DecoderOnlyTransformer(cfg)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # -------------------------
    # 3) Setup optimizer
    # -------------------------
    print(f"\nSetup optimizer (lr={learning_rate})")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1
    )

    # -------------------------
    # 4) Training loop
    # -------------------------
    print(f"\nTraining for {num_steps} steps...")
    print(f"Block size: {block_size}, Batch size: {batch_size}")
    print("=" * 80)

    train_history = []
    model.train()

    for step in range(1, num_steps + 1):
        # Get batch
        x, y = make_batch(train_ids, block_size, batch_size)
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        _, loss = model(x, targets=y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Evaluation
        if step % eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batch(val_ids, block_size, batch_size)
                vx = vx.to(device)
                vy = vy.to(device)
                _, vloss = model(vx, targets=vy)

            train_loss_val = float(loss)
            val_loss_val = float(vloss)
            train_history.append((step, train_loss_val, val_loss_val))

            print(
                f"step {step:4d} | train loss {train_loss_val:.3f} | val loss {val_loss_val:.3f}"
            )
            model.train()

    print("=" * 80)
    print("Training complete!")

    # -------------------------
    # 5) Save model if requested
    # -------------------------
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "train_history": train_history,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_model": tokenizer_model,
        }

        torch.save(checkpoint, save_path)
        print(f"\nModel saved to: {save_path}")

    return model, tokenizer, train_history


def generate_text(
    model,
    tokenizer,
    prompt="Hello, how are you",
    max_new_tokens=100,
    temperature=0.8,
    device=None,
):
    """
    Generate text using a trained model

    Args:
        model: Trained DecoderOnlyTransformer
        tokenizer: HuggingFace tokenizer
        prompt: Starting text prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to use (default: None = use model's device)

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


def load_checkpoint(checkpoint_path, device=None):
    """
    Load a model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on (default: None = auto-detect)

    Returns:
        model: Loaded DecoderOnlyTransformer
        tokenizer: The tokenizer
        train_history: Training history if available
    """
    from transformers import AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    cfg.device = device

    model = DecoderOnlyTransformer(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint["tokenizer_model"])
    train_history = checkpoint.get("train_history", [])

    print(f"Loaded model from: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Vocab size: {cfg.vocab_size}")

    return model, tokenizer, train_history
