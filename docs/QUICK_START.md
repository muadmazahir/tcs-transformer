# Quick Start Guide

This guide provides simple examples to get started with training each of the four models in the TCS-Transformer project.

## Prerequisites

Make sure you have completed the installation steps:

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## 1. Training the TCS VAE

The TCS VAE learns semantic representations of predicate-argument structures using a variational autoencoder.

```python
from tcs_transformer.scripts.tcs_scripts.train import train_model

# Train TCS VAE
train_model(config_path='configs/config.json')
```

**Note**: You need to have transformed TCS data prepared. See the full documentation for data preparation steps.

## 2. Training the GPT-style Transformer

A standard decoder-only transformer for language modeling.

```python
from tcs_transformer.scripts.transformer_scripts.train_transformer import train_transformer

# Train GPT-style transformer on WikiText-2
model, tokenizer, history = train_transformer(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    num_steps=2000,
    learning_rate=3e-4,
    batch_size=12,
    device='cuda'  # or 'cpu'
)
```

### Generate Text

```python
from tcs_transformer.scripts.transformer_scripts.train_transformer import generate_text

# Generate text from trained model
text = generate_text(
    model, 
    tokenizer, 
    prompt="Hello, how are you",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

## 3. Training the Hybrid Model

Combines a GPT-style transformer with the TCS VAE for joint training.

```python
from tcs_transformer.scripts.train_hybrid import train_hybrid_model

# Joint training of transformer + TCS VAE
# Note: This requires ERG parser and is slower due to on-the-fly parsing
model, tokenizer, history = train_hybrid_model(
    # Data parameters
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    
    # Model parameters
    n_layers=4,
    n_heads=4,
    d_model=256,
    
    # Training parameters
    num_steps=3000,
    learning_rate=3e-4,
    batch_size=12,
    
    # Hybrid-specific parameters
    semantic_weight=0.1,      # Weight for semantic loss
    warmup_steps=1000,        # Train transformer first, then add TCS
    pretrained_tcs_path=None, # Path to pretrained TCS VAE (optional)
    
    device='cuda'
)
```

### Generate Text from Hybrid Model

```python
from tcs_transformer.scripts.train_hybrid import generate_text_hybrid

# Generate text
text = generate_text_hybrid(
    model, 
    tokenizer, 
    prompt="Hello, how are you",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

## 4. Training the TCS Transformer

Directly learns to embed semantic functions using a transformer architecture.

```python
from tcs_transformer.scripts.train_tcs_transformer import train_tcs_transformer

# Train TCS transformer
model, history = train_tcs_transformer(
    # Data parameters
    config_path='configs/config.json',
    
    # Model architecture
    d_model=256,
    n_layers=4,
    n_heads=4,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1,
    
    # Training parameters
    num_epochs=10,
    learning_rate=1e-4,
    batch_size=1,
    grad_clip=1.0,
    
    # Save/load
    save_dir='saved/tcs_transformer',
    seed=42,
    device='cuda'
)
```

### Load Trained Model

```python
from tcs_transformer.scripts.train_tcs_transformer import load_checkpoint

# Load a trained model
model, checkpoint = load_checkpoint(
    'saved/tcs_transformer/checkpoint_epoch_10.pth',
    device='cuda'
)

# Use for inference
model.eval()
# ... inference code ...
```

## Complete Example: TCS VAE Training Pipeline

Here's a complete workflow from data generation to training:

```python
from tcs_transformer.scripts.tcs_scripts.generate_dummy_data import generate_dummy_data
from tcs_transformer.scripts.tcs_scripts.prepare_train import prepare_training_data
from tcs_transformer.scripts.tcs_scripts.train import train_model

# 1. Generate dummy data (optional - examples already provided)
generate_dummy_data(
    erg_path='examples/erg/erg-1214-x86-64-0.9.34.dat',
    unk2pos_path='examples/erg/unk2pos.json',
    output_dir='examples/dummy_data'
)

# 2. Transform data for training
prepare_training_data(
    corpus_dir='examples/dummy_data',
    output_dir='examples/dummy_data/transformed/TCS_f0-lgF-PAS-Gen_dummy',
    transform_config_path='configs/transform_config.json'
)

# 3. Train the TCS VAE
train_model(
    config_path='configs/config.json',
    seed=42
)
```

## Next Steps

- See `TRANSFORMER_TCS.md` for detailed information about the TCS Transformer model
- See `HYBRID_MODEL.md` for detailed information about the Hybrid model
- See `TCS_VAE_ARCHITECTURE.md` for detailed information about the TCS VAE
- Check `configs/` directory for configuration options
- Refer to individual model documentation for advanced usage

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or model dimensions
2. **ERG Parser Errors**: Ensure ERG grammar files are in `examples/erg/`
3. **Slow Training (Hybrid)**: Use `warmup_steps` to train transformer-only first
4. **Import Errors**: Make sure you've activated the Poetry virtual environment

For more detailed troubleshooting, see `HYBRID_MODEL_ISSUES.md`.
