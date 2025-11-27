"""
TCS Transformer Model

A transformer model designed to process TCS (Type-driven Compositional Semantics) data.
This model treats structured semantic predicate-argument data as sequences suitable
for transformer processing.

Architecture:
- Flattens predicate-argument structure into sequences with separator tokens
- Processes sequences through transformer blocks
- Pools representations for each node
- Predicts semantic functions for each node
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TCSTransformerConfig:
    """Configuration for Transformer trained on TCS data"""

    vocab_size: int  # Number of unique predicate-argument tokens
    num_sem_funcs: int  # Number of semantic functions to predict
    d_model: int = 256  # Model dimension
    n_layers: int = 4  # Number of transformer layers
    n_heads: int = 4  # Number of attention heads
    d_ff: int = 1024  # Feed-forward dimension
    max_seq_len: int = 512  # Maximum sequence length
    dropout: float = 0.1  # Dropout rate
    pad_token_id: int = 0  # Padding token ID
    sep_token_id: int = 1  # Separator token ID (between nodes)
    device: str = "cpu"


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, padding_mask=None):
        # Self-attention with pre-norm
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=padding_mask)
        x = x + attn_out
        # Feed-forward with pre-norm
        x = x + self.mlp(self.ln2(x))
        return x


class TCSTransformer(nn.Module):
    """
    Transformer model for TCS data

    Architecture:
    1. Embed predicate-argument tokens
    2. Add positional embeddings
    3. Process through transformer blocks
    4. Pool representations for each node
    5. Predict semantic functions
    """

    def __init__(self, cfg: TCSTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.token_emb = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id
        )
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                for _ in range(cfg.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Output heads for semantic function prediction
        # Each node can participate in multiple semantic functions
        self.sem_func_head = nn.Linear(cfg.d_model, cfg.num_sem_funcs)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, node_boundaries, padding_mask=None):
        """
        Forward pass

        Args:
            input_ids: Flattened sequence of predicate-argument tokens [B, L]
            node_boundaries: List of node boundary indices for each batch item
            padding_mask: Boolean mask for padding [B, L]

        Returns:
            node_logits: Logits for semantic function prediction [B, num_nodes, num_sem_funcs]
        """
        B, L = input_ids.shape

        # Embed tokens
        positions = (
            torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        )
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask)

        x = self.ln_f(x)

        # Pool representations for each node (use mean pooling over tokens in each node)
        node_representations = []
        for batch_idx in range(B):
            batch_nodes = []
            boundaries = node_boundaries[batch_idx]
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if start < end:
                    # Mean pool over tokens in this node
                    node_repr = x[batch_idx, start:end].mean(dim=0)
                    batch_nodes.append(node_repr)
            if batch_nodes:
                node_representations.append(torch.stack(batch_nodes))
            else:
                # Handle empty case
                node_representations.append(
                    torch.zeros(1, self.cfg.d_model, device=x.device)
                )

        # Pad to same length
        max_nodes = max(len(nodes) for nodes in node_representations)
        padded_nodes = []
        for nodes in node_representations:
            if len(nodes) < max_nodes:
                padding = torch.zeros(
                    max_nodes - len(nodes), self.cfg.d_model, device=x.device
                )
                nodes = torch.cat([nodes, padding], dim=0)
            padded_nodes.append(nodes)

        node_embeddings = torch.stack(padded_nodes)  # [B, max_nodes, d_model]

        # Predict semantic functions for each node
        logits = self.sem_func_head(node_embeddings)  # [B, max_nodes, num_sem_funcs]

        return logits
