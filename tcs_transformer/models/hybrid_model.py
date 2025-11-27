"""
Hybrid Transformer-TCS Model

Combines GPT-style Transformer with TCS Variational Autoencoder for
semantically-aware language modeling.

The hybrid model learns:
1. Sequential patterns from transformer attention
2. Semantic plausibility from TCS VAE

Architecture:
- Base: GPT-style decoder-only transformer for language modeling
- Semantic Module: TCS VAE (encoder + decoder) for semantic constraints
- Combined Loss: Language modeling loss + weighted semantic plausibility loss
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from tcs_transformer.models.transformer_model import GPTConfig, DecoderOnlyTransformer
from tcs_transformer.models.tcs_model import VarAutoencoder


@dataclass
class HybridConfig:
    """Configuration for Hybrid Transformer-TCS Model (Joint Training Only)"""

    # GPT Transformer config
    vocab_size: int
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 128
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    emb_dropout: float = 0.1

    # TCS VAE config (always enabled for joint training)
    semantic_weight: float = 0.1  # λ - weight for semantic loss

    # Device
    device: str = "cpu"


class HybridTransformerTCS(nn.Module):
    """
    Hybrid model combining GPT Transformer with TCS VAE

    This model learns both:
    1. Language modeling via transformer (next-token prediction)
    2. Semantic plausibility via TCS VAE (predicate-argument structure)

    Loss: total_loss = lm_loss + λ * semantic_loss
    where λ controls the trade-off between fluency and semantic plausibility
    """

    def __init__(self, cfg: HybridConfig, tcs_vae: VarAutoencoder):
        """
        Initialize Hybrid Model for Joint Training

        Args:
            cfg: Hybrid model configuration
            tcs_vae: Pre-trained or initialized TCS VAE (required for joint training)
        """
        super().__init__()
        self.cfg = cfg

        # GPT Transformer for language modeling
        gpt_cfg = GPTConfig(
            vocab_size=cfg.vocab_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
            attn_dropout=cfg.attn_dropout,
            resid_dropout=cfg.resid_dropout,
            emb_dropout=cfg.emb_dropout,
            device=cfg.device,
        )
        self.transformer = DecoderOnlyTransformer(gpt_cfg)

        self.tcs_vae = tcs_vae
        self.semantic_weight = cfg.semantic_weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        tcs_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass

        Args:
            idx: Input token indices (B, L)
            targets: Target token indices (B, L) for language modeling loss
            tcs_data: Dictionary with TCS encoder/decoder data (optional)
                Format: {
                    "encoder": tuple of encoder inputs,
                    "decoder": tuple of decoder inputs
                }

        Returns:
            logits: Output logits (B, L, vocab_size)
            loss: Combined loss if targets provided, else None
            loss_dict: Dictionary with individual loss components
        """
        # 1. Transformer forward pass (language modeling)
        logits, lm_loss = self.transformer(idx, targets=targets)

        # 2. TCS semantic loss (if data provided)
        semantic_loss = None
        kl_div = None
        semantic_log_prob = None

        if tcs_data is not None and targets is not None:
            # Run TCS VAE to get semantic plausibility
            (
                log_truth_batch,
                kl_div,
                l2_norm_reg,
                pos_sum,
                neg_sum,
                mu_batch,
                sigma2_batch,
            ) = self.tcs_vae.run(**tcs_data)

            # Semantic loss: negative log-likelihood + KL divergence
            # We want to maximize log_truth_batch, so minimize -log_truth_batch
            semantic_nll = -log_truth_batch.mean()

            # Add KL divergence for variational inference
            if self.tcs_vae.variational:
                beta = self.tcs_vae.start_beta  # Can be annealed during training
                semantic_loss = semantic_nll + beta * kl_div.mean()
            else:
                semantic_loss = semantic_nll

            semantic_log_prob = log_truth_batch.mean()

        # 3. Combined loss
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            if semantic_loss is not None:
                total_loss = total_loss + self.semantic_weight * semantic_loss

        # 4. Return loss components for logging
        loss_dict = None
        if total_loss is not None:
            loss_dict = {
                "total_loss": total_loss,
                "lm_loss": lm_loss,
                "semantic_loss": semantic_loss
                if semantic_loss is not None
                else torch.tensor(0.0),
                "kl_div": kl_div.mean() if kl_div is not None else torch.tensor(0.0),
                "semantic_log_prob": semantic_log_prob
                if semantic_log_prob is not None
                else torch.tensor(0.0),
            }

        return logits, total_loss, loss_dict

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        tcs_filter: bool = False,
        tcs_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional TCS filtering

        Args:
            idx: Starting sequence (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            tcs_filter: Whether to filter generations by TCS semantic plausibility
            tcs_threshold: Minimum semantic plausibility score (if tcs_filter=True)

        Returns:
            Generated sequence (B, L + max_new_tokens)
        """
        if not tcs_filter or not self.use_tcs:
            # Standard transformer generation
            return self.transformer.generate(idx, max_new_tokens, temperature)
        else:
            # TCS-filtered generation (future work)
            # Would require extracting pred-arg structures from generated tokens
            # and scoring them with TCS VAE
            raise NotImplementedError(
                "TCS-filtered generation requires mapping tokens to "
                "predicate-argument structures - coming soon!"
            )

    def get_transformer_embeddings(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get transformer embeddings for input tokens (useful for analysis)

        Args:
            idx: Input token indices (B, L)

        Returns:
            Embeddings (B, L, d_model)
        """
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.transformer.token_emb(idx) + self.transformer.pos_emb(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x)
        return x

    def set_semantic_weight(self, weight: float):
        """
        Adjust semantic loss weight (useful for curriculum learning)

        Args:
            weight: New semantic weight (λ)
        """
        self.semantic_weight = weight
        print(f"Semantic weight updated to: {weight}")

    def freeze_transformer(self):
        """Freeze transformer parameters (only train TCS)"""
        for param in self.transformer.parameters():
            param.requires_grad = False
        print("Transformer parameters frozen")

    def freeze_tcs(self):
        """Freeze TCS parameters (only train transformer)"""
        if self.use_tcs:
            for param in self.tcs_vae.parameters():
                param.requires_grad = False
            print("TCS VAE parameters frozen")

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen")


def create_hybrid_model(
    vocab_size: int,
    tcs_vae: Optional[VarAutoencoder] = None,
    n_layers: int = 4,
    n_heads: int = 4,
    d_model: int = 256,
    d_ff: int = 1024,
    max_seq_len: int = 128,
    semantic_weight: float = 0.1,
    device: Optional[str] = None,
) -> HybridTransformerTCS:
    """
    Convenience function to create a hybrid model for joint training

    Args:
        vocab_size: Size of vocabulary
        tcs_vae: Pre-trained or initialized TCS VAE (required for joint training)
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model dimension
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        semantic_weight: Weight for semantic loss (λ)
        device: Device to use (None = auto-detect)

    Returns:
        Initialized HybridTransformerTCS model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tcs_vae is None:
        raise ValueError(
            "TCS VAE is required for joint training. Provide a pretrained TCS VAE."
        )

    cfg = HybridConfig(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        semantic_weight=semantic_weight,
        device=device,
    )

    model = HybridTransformerTCS(cfg, tcs_vae)
    model.to(device)

    return model
