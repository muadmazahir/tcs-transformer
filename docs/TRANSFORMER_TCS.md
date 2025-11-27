# Training Transformer on TCS Data

This document explains how to use `train_tcs_transformer.py` to train a transformer model on the same semantic data used by VarAutoencoder.

## Overview

### Architecture Comparison

| Model | VarAutoencoder (Original) | TCS Transformer (New) |
|-------|---------------------------|----------------------|
| **Encoder** | PASEncoder: Embeds predicate-arguments → Mean pool → Linear layers → μ, σ² | Transformer: Embeds predicate-arguments → Self-attention → Mean pool per node → Node embeddings |
| **Latent Space** | Gaussian latent vectors (z ~ N(μ, σ²)) | Deterministic node embeddings |
| **Decoder** | OneLayerSemFuncsDecoder: Semantic function matching | Linear classifier: Predict semantic functions |
| **Loss** | ELBO (reconstruction + KL divergence) | Cross-entropy classification |
| **Training** | Variational inference | Standard supervised learning |

### Key Differences

**VarAutoencoder**:
- Structured semantic processing (predicate-argument pairs)
- Probabilistic latent representations
- Explicitly models semantic functions via dot products
- Reconstructs semantic structure

**TCS Transformer**:
- Treats structured data as sequences
- Deterministic representations
- Learns to classify nodes into semantic function categories
- Standard transformer architecture (self-attention)

## Data Format

Both models use the same transformed TCS data:

```python
# Sample structure:
[
    pred_func_nodes_ctxt_predargs,  # List[List[int]] - predicate-argument indices per node
    decoder_info,                    # Nested structure
    pred_funcs_ix_list,             # List - semantic function indices
    vars_list,                       # List - variable assignments
    args_num_sum_list               # List - argument counts
]
```

**Conversion to Transformer Format**:
```python
# Original: [[9, 1, 2, 7], [10, 11, 7], [12, 13, 7]]
# Flattened: [9, 1, 2, 7, <SEP>, 10, 11, 7, <SEP>, 12, 13, 7, <SEP>]
# Node boundaries: [0, 5, 9, 14]  # Used for pooling
```

## Usage

### Basic Training

```bash
python tcs_transformer/scripts/train_tcs_transformer.py \
    --config configs/config.json \
    --epochs 10 \
    --lr 1e-4 \
    --device cuda \
    --save_dir saved/tcs_transformer
```

### Arguments

- `--config`: Path to TCS config file (default: `configs/config.json`)
  - Uses same config as VarAutoencoder to access data paths and vocabulary
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--device`: Device to train on (`cuda`/`cpu`, auto-detects if not specified)
- `--save_dir`: Directory to save checkpoints (default: `saved/tcs_transformer`)
- `--seed`: Random seed (default: 42)

### Example

```python
# Train transformer on dummy TCS data
python tcs_transformer/scripts/train_tcs_transformer.py \
    --config configs/config.json \
    --epochs 5 \
    --lr 2e-4 \
    --device cuda:0
```

## Model Architecture

```
Input: Predicate-argument indices [B, L]
  ↓
Token Embeddings [B, L, d_model]
  ↓
+ Positional Embeddings
  ↓
Dropout
  ↓
Transformer Blocks × N
  - Multi-Head Self-Attention
  - LayerNorm (pre-norm)
  - Feed-Forward (MLP)
  - Residual connections
  ↓
Final LayerNorm
  ↓
Pool per Node (mean pooling)
  ↓
Node Embeddings [B, num_nodes, d_model]
  ↓
Linear Classifier
  ↓
Logits [B, num_nodes, num_sem_funcs]
```

### Hyperparameters

Default configuration (adjustable in `TCSTransformerConfig`):

```python
d_model = 256          # Model dimension
n_layers = 4           # Number of transformer layers
n_heads = 4            # Number of attention heads
d_ff = 1024           # Feed-forward dimension
max_seq_len = 512     # Maximum sequence length
dropout = 0.1          # Dropout rate
```

## Output

### Training Progress

```
Using device: cuda
Loading data from: examples/dummy_data/transformed/TCS_f0-lgF-PAS-Gen_dummy
Loaded 1 samples from 1 files
Vocabulary size: 110
Number of semantic functions: 123
Number of trainable parameters: 1,234,567
================================================================================
Starting training...
================================================================================
Epoch 1: 100%|███████████████| 1/1 [00:01<00:00, loss=4.8234]
Epoch 1/10 - Average Loss: 4.8234
Saved checkpoint to saved/tcs_transformer/checkpoint_epoch_1.pth
...
```

### Checkpoint Structure

```python
{
    'epoch': 5,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'loss': 2.3456,
    'config': {
        'vocab_size': 110,
        'num_sem_funcs': 123,
        'd_model': 256,
        ...
    }
}
```

## Loading and Using Trained Model

```python
import torch
from tcs_transformer.models.tcs_transformer_model import (
    TCSTransformer, TCSTransformerConfig
)
# Or use the convenience function
from tcs_transformer.scripts.train_tcs_transformer import load_checkpoint

# Load checkpoint using convenience function
model, checkpoint = load_checkpoint('saved/tcs_transformer/checkpoint_epoch_5.pth')

# Or manually:
# checkpoint = torch.load('saved/tcs_transformer/checkpoint_epoch_5.pth')
# cfg = TCSTransformerConfig(**checkpoint['config'])
# model = TCSTransformer(cfg)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Inference
with torch.no_grad():
    logits = model(input_ids, node_boundaries)
    predictions = logits.argmax(dim=-1)
```

## Comparison with VarAutoencoder

### Training Command Comparison

**VarAutoencoder**:
```bash
python tcs_transformer/scripts/tcs_scripts/train.py \
    --config configs/config.json
```

**TCS Transformer**:
```bash
# As a module (recommended)
python -c "from tcs_transformer.scripts.train_tcs_transformer import train_tcs_transformer; train_tcs_transformer(config_path='configs/config.json', num_epochs=10)"
```

### Expected Behavior

- **VarAutoencoder**: Should converge to ELBO loss around -10 to -50 (depends on β)
- **TCS Transformer**: Should converge to cross-entropy loss around 1.0-3.0

The losses are not directly comparable because:
- VarAutoencoder: ELBO = reconstruction + KL divergence
- TCS Transformer: Cross-entropy classification

## Technical Details

### Data Flow

1. **Load transformed TCS data**: Same files as VarAutoencoder
2. **Collate function**: Flattens structured data into sequences
3. **Forward pass**: Transformer processes sequences
4. **Pooling**: Extract per-node representations
5. **Classification**: Predict semantic functions
6. **Loss**: Cross-entropy over predicted vs. true semantic functions

### Memory Requirements

- **VarAutoencoder**: ~500MB (depends on vocab size)
- **TCS Transformer**: ~1-2GB (depends on d_model, n_layers)

Transformer uses more memory due to self-attention.

### Speed

- **VarAutoencoder**: ~1-2 batches/sec (includes semantic function matching)
- **TCS Transformer**: ~2-5 batches/sec (standard backprop)

## Limitations

1. **Simplified target**: Current implementation uses first predicate-argument as target (reconstruction-style)
   - Could be extended to multi-label classification over semantic functions
2. **Batch size = 1**: Follows TCS data convention
   - Could implement dynamic batching for speed
3. **No variational inference**: Deterministic representations
   - Could add VAE-style reparameterization trick

## Future Improvements

1. **Multi-label classification**: Predict multiple semantic functions per node
2. **Graph attention**: Incorporate DMRS graph structure directly
3. **Hierarchical pooling**: Better aggregation of node representations
4. **Contrastive learning**: Like VarAutoencoder's negative sampling
5. **Hybrid architecture**: Combine transformer encoder with semantic decoder

## Questions?

This script demonstrates how to train a standard transformer on the same structured semantic data used by VarAutoencoder. The key insight is that **structured predicate-argument data can be linearized** into sequences suitable for transformer processing, while still preserving semantic information through careful pooling and classification.
