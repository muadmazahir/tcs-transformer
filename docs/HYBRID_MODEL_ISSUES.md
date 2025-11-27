# Hybrid Model: Implementation Status

## Problem Details

### Current Forward Pass

```python
# Step 1: Transformer generates logits
logits, lm_loss = self.transformer(idx, targets=targets)

# Step 2: TCS computes semantic loss (SEPARATE DATA!)
log_truth_batch, kl_div, ... = self.tcs_vae.run(**tcs_data)

# Step 3: Combine losses
total_loss = lm_loss + λ * semantic_loss
```

**Issue**: `tcs_data` is passed separately and has no connection to the transformer's output `logits`!

### Gradient Flow Diagram

```
User provides:
- idx (input tokens) → Transformer → logits
- tcs_data (separate semantic structures) → TCS VAE → semantic_loss

Combined loss = lm_loss + λ * semantic_loss
                    ↓                ↓
                Transformer    TCS VAE (frozen)
                   ✅              ❌

Problem: semantic_loss has no gradient path to transformer outputs!
```

## Why This Matters

### Joint Training (`training_mode="joint"`)
- Both models update ✅
- **BUT**: Still no connection between outputs ❌
- Semantic loss is computed on separate data, not transformer's predictions

## The Missing Link

What we need:
```python
# Extract semantic structures FROM transformer outputs
semantic_structures = extract_pred_arg_structures(logits, tokenizer)

# THEN compute TCS loss on those structures
semantic_loss = self.tcs_vae.run(semantic_structures)

# NOW gradients can flow: total_loss → semantic_loss → structures → logits → transformer
```

## Current Workaround

For now, the model works as:
- **Joint mode**: Both models update, but semantic signal is weak without connection

The semantic loss currently serves more as a **monitoring metric** than a training signal.
