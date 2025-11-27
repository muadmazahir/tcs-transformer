# Hybrid Transformer-TCS Model

## Overview

The Hybrid Transformer-TCS model combines two powerful architectures:

1. **GPT-Style Transformer**: Learns sequential patterns and contextual representations through multi-head self-attention
2. **TCS Variational Autoencoder**: Learns semantic plausibility of predicate-argument structures

This integration allows the model to generate fluent text while respecting semantic constraints.

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                  Hybrid Model                            │
│                                                          │
│  ┌──────────────────┐       ┌────────────────────────┐ │
│  │  GPT Transformer │       │     TCS VAE            │ │
│  │                  │       │                        │ │
│  │  - Token Emb     │       │  ┌──────────────────┐ │ │
│  │  - Position Emb  │       │  │  PAS Encoder     │ │ │
│  │  - N Layers:     │       │  │  (Pred-Arg)      │ │ │
│  │    * Self-Attn   │       │  └──────────────────┘ │ │
│  │    * Feed-Fwd    │       │           │           │ │
│  │  - LM Head       │       │     ┌─────▼────┐      │ │
│  │                  │       │     │  μ, σ²   │      │ │
│  │                  │       │     └─────┬────┘      │ │
│  └────────┬─────────┘       │           │           │ │
│           │                  │  ┌────────▼────────┐ │ │
│           │                  │  │  SemFuncs      │ │ │
│           │                  │  │  Decoder       │ │ │
│           │                  │  └────────────────┘ │ │
│           │                  └────────┬─────────────┘ │
│           │                           │                │
│           ▼                           ▼                │
│     LM Loss (Cross-Entropy)    Semantic Loss          │
│           │                           │                │
│           └───────────┬───────────────┘                │
│                       ▼                                 │
│             Total Loss = LM_Loss + λ * Sem_Loss        │
└─────────────────────────────────────────────────────────┘
```

### Loss Function

```
Total Loss = Language Modeling Loss + λ × Semantic Loss

where:
- Language Modeling Loss: Cross-entropy for next-token prediction
- Semantic Loss: -log P(semantic_structure) + β × KL_divergence
- λ (lambda): Trade-off parameter between fluency and semantic plausibility
- β (beta): KL divergence weight for variational inference
```

---

## Integration Strategy

### Why This Approach?

**Semantic-Aware Language Modeling**: The model learns to generate text that is both fluent (from transformer) and semantically plausible (from TCS VAE).

**Key Benefits:**
1. ✅ **Modular**: Clean separation - transformer handles sequences, TCS handles semantics
2. ✅ **Flexible**: Can control trade-off with λ parameter
3. ✅ **Trainable**: End-to-end differentiable, joint or staged training
4. ✅ **Interpretable**: Individual loss components for debugging and analysis

---

## Usage

**Note**: The hybrid training script now operates in **JOINT MODE ONLY**. Both transformer and TCS VAE are trained together. For separate training of individual components, use:
- Transformer-only: `tcs_transformer.scripts.transformer_scripts.train_transformer`
- TCS VAE-only: `tcs_transformer.scripts.tcs_scripts.train`

### 1. Basic Training (Transformer Only)

Start with transformer-only training to establish baseline:

```python
from tcs_transformer.scripts import train_hybrid_model

# Train transformer only (TCS disabled)
model, tokenizer, history = train_hybrid_model(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    use_tcs=False,  # No semantic constraints
    n_layers=4,
    n_heads=4,
    d_model=256,
    num_steps=2000,
    save_path="models/transformer_baseline.pth"
)
```

### 2. Joint Hybrid Training (With TCS)

Once you have a pretrained TCS VAE, enable semantic constraints for joint training:

```python
from tcs_transformer.scripts import train_hybrid_model

# Joint training with TCS semantic constraints
# NOTE: This will be slower due to on-the-fly ERG parsing
model, tokenizer, history = train_hybrid_model(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    
    # Enable TCS (trains both models jointly)
    use_tcs=True,
    pretrained_tcs_path="models/pretrained_tcs_vae.pth",  # Optional: Your TCS VAE
    semantic_weight=0.1,  # λ - balance fluency vs semantics
    
    # Architecture
    n_layers=4,
    n_heads=4,
    d_model=256,
    
    # Training
    num_steps=5000,
    save_path="models/hybrid_model.pth"
)
```

**Important Notes:**
- ⚠️ **TCS parsing is slow (~0.5-2s per sentence)**. Training will take significantly longer.
- The script converts tokens → text → DMRS → TCS format on-the-fly during training
- Both transformer and TCS VAE are trained together (joint mode only)
- ERG grammar file required: `examples/erg/erg-1214-x86-64-0.9.34.dat`

### 3. Warmup Training (Recommended)

For better efficiency, use warmup to train transformer first, then add TCS:

```python
# Warmup: Train transformer first, then enable TCS
model, tokenizer, history = train_hybrid_model(
    use_tcs=True,
    pretrained_tcs_path="models/pretrained_tcs_vae.pth",
    warmup_steps=2000,  # Train transformer-only for 2000 steps
    semantic_weight=0.1,
    num_steps=5000,  # TCS enabled for remaining 3000 steps
)
```

This avoids slow TCS parsing during initial training when the transformer is still learning basics.

### 4. Generation

```python
from tcs_transformer.scripts import generate_text_hybrid

# Generate text
text = generate_text_hybrid(
    model,
    tokenizer,
    prompt="The concept of semantic plausibility",
    max_new_tokens=100,
    temperature=0.8
)
print(text)
```

### 5. Loading Saved Models

```python
from tcs_transformer.scripts import load_hybrid_checkpoint, generate_text_hybrid

# Load checkpoint
model, tokenizer, history = load_hybrid_checkpoint("models/hybrid_model.pth")

# Generate
text = generate_text_hybrid(model, tokenizer, prompt="Once upon a time")
print(text)
```

---

## Training Parameters

### Data Parameters
- `dataset_name`: HuggingFace dataset (default: "wikitext")
- `dataset_config`: Dataset config (default: "wikitext-2-raw-v1")
- `tokenizer_model`: Pretrained tokenizer (default: "google/gemma-2-2b-it")

### Architecture
- `n_layers`: Transformer layers (default: 4)
- `n_heads`: Attention heads (default: 4)
- `d_model`: Model dimension (default: 256)
- `d_ff`: Feed-forward dimension (default: 1024)
- `block_size`: Context window (default: 128)

### Training
- `batch_size`: Batch size (default: 12)
- `learning_rate`: Learning rate (default: 3e-4)
- `num_steps`: Training steps (default: 2000)
- `eval_every`: Evaluation frequency (default: 200)
- `grad_clip`: Gradient clipping (default: 1.0)

### Hybrid-Specific
- `use_tcs`: Enable TCS semantic constraints with on-the-fly parsing (default: False)
- `semantic_weight`: λ - weight for semantic loss (default: 0.1)
- `pretrained_tcs_path`: Path to TCS VAE checkpoint (optional)
- `warmup_steps`: Steps to train transformer-only before enabling TCS (default: 0)

**Note**: The hybrid script trains both models jointly (no separate modes). For training individual models, use:
- `tcs_transformer.scripts.transformer_scripts.train_transformer`
- `tcs_transformer.scripts.tcs_scripts.train`

---

## Understanding the Semantic Weight (λ)

The `semantic_weight` parameter controls the trade-off between fluency and semantic plausibility:

- **λ = 0.0**: Pure transformer (no semantic constraints)
  - Maximum fluency
  - May generate semantically implausible outputs

- **λ = 0.1 - 0.3** (Recommended): Balanced approach
  - Good fluency with semantic constraints
  - Best for most applications

- **λ = 0.5 - 1.0**: Strong semantic bias
  - Highly semantically plausible
  - May sacrifice some fluency

- **λ > 1.0**: Semantic-dominated
  - Maximum semantic plausibility
  - Significant fluency loss

### Tuning λ

Start with λ = 0.1 and adjust based on your task:

```python
# For tasks requiring high fluency (e.g., creative writing)
semantic_weight=0.05

# For balanced performance (recommended)
semantic_weight=0.1

# For tasks requiring semantic accuracy (e.g., knowledge-grounded generation)
semantic_weight=0.3
```

---

## Advanced Features

### Dynamic Semantic Weight

Adjust semantic weight during training:

```python
# Start training
model, tokenizer, history = train_hybrid_model(
    use_tcs=True,
    semantic_weight=0.1,
    ...
)

# Later, adjust weight for fine-tuning
model.set_semantic_weight(0.2)  # Increase semantic influence
```

### Selective Freezing (Advanced)

For custom training loops, you can control which parts train:

```python
# Freeze transformer, train TCS only (manual training loop required)
model.freeze_transformer()

# Freeze TCS, train transformer only (manual training loop required)
model.freeze_tcs()

# Unfreeze everything
model.unfreeze_all()
```

### Extract Embeddings

Get transformer embeddings for analysis:

```python
import torch

# Get embeddings for input tokens
token_ids = torch.tensor([[1, 2, 3, 4, 5]])
embeddings = model.get_transformer_embeddings(token_ids)
print(embeddings.shape)  # (1, 5, d_model)
```

---

## Current Limitations & Future Work

### Current Limitations

1. **TCS Data Extraction**: Currently, the model cannot automatically extract predicate-argument structures from text tokens. You need to provide TCS data separately.

2. **TCS-Filtered Generation**: The `tcs_filter` option in generation is not yet implemented.

3. **Token-to-Semantic Mapping**: Requires custom preprocessing to map tokens to TCS semantic representations.

### Future Enhancements

1. **Automatic Pred-Arg Extraction**: Build a module to extract predicate-argument structures from generated tokens

2. **TCS-Guided Decoding**: Implement beam search with TCS semantic constraints

3. **Cross-Attention Integration**: Add cross-attention layers to inject TCS embeddings directly into transformer

4. **Multi-Task Learning**: Train on multiple objectives (LM + semantic similarity + entailment)

---

## Example: Full Training Pipeline

```python
from tcs_transformer.scripts import train_hybrid_model, generate_text_hybrid

# Step 1: Train transformer baseline
print("Training transformer baseline...")
model, tokenizer, history = train_hybrid_model(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    use_tcs=False,
    num_steps=2000,
    save_path="models/transformer_baseline.pth"
)

# Step 2: Add TCS semantic constraints (when ready)
# print("Fine-tuning with TCS constraints...")
# model, tokenizer, history = train_hybrid_model(
#     use_tcs=True,
#     pretrained_tcs_path="models/pretrained_tcs_vae.pth",
#     semantic_weight=0.1,
#     warmup_steps=500,
#     num_steps=3000,
#     training_mode="joint",
#     save_path="models/hybrid_final.pth"
# )

# Step 3: Generate text
print("\nGenerating text...")
prompts = [
    "The concept of semantic",
    "In natural language processing",
    "The transformer architecture"
]

for prompt in prompts:
    text = generate_text_hybrid(
        model, 
        tokenizer, 
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8
    )
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {text}")
```

---

## Files Created

1. **`tcs_transformer/models/hybrid_model.py`**
   - `HybridTransformerTCS`: Main hybrid model class
   - `HybridConfig`: Configuration dataclass
   - `create_hybrid_model()`: Convenience function

2. **`tcs_transformer/transformer_scripts/train_hybrid.py`**
   - `train_hybrid_model()`: Main training function
   - `load_hybrid_checkpoint()`: Load saved models
   - `generate_text_hybrid()`: Text generation

3. **`HYBRID_MODEL.md`** (this file)
   - Comprehensive documentation

---

## Questions?

This is a novel architecture combining attention-based sequence modeling with semantic constraint learning. The implementation provides a solid foundation for:

- Semantic-aware text generation
- Structured knowledge integration
- Multi-objective language modeling
- Interpretable NLP systems

For questions or issues, refer to the code documentation or experiment with the parameters!
