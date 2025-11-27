"""
Decoder-only Transformer (GPT-style) implementation

A minimal, clean implementation of a GPT-style decoder-only Transformer.
Features:
- Token + learned positional embeddings
- Multi-Head Causal Self-Attention with mask
- Pre-Norm residual blocks
- GELU feed-forward (MLP)
- Dropout knobs
- Tied output projection (weights tied with token embedding)
- An autoregressive generate utility
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for GPT model"""

    vocab_size: int
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 64
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    emb_dropout: float = 0.1
    device: str = "cpu"


class LayerNorm(nn.Module):
    """Layer normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln(x)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal_mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        y = attn_weights @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer decoder block with pre-norm"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.attn_dropout)
        self.ln2 = LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.resid_dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer (GPT-style)"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        """
        Forward pass

        Args:
            idx: Input token indices (B, L)
            targets: Target token indices (B, L) for computing loss

        Returns:
            logits: Output logits (B, L, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        B, L = idx.shape
        pos = torch.arange(0, L, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens=50, temperature=0.8):
        """
        Autoregressive generation

        Args:
            idx: Starting sequence (B, L)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated sequence (B, L + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
