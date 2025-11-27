# TCS-Transformer

Integration of Functional Distributional Semantics and Transformer architectures for semantic representation learning.

## Overview

This project is an attempt to integrate [Functional Distributional Semantics](https://www.cl.cam.ac.uk/~gete2/thesis.pdf) with the Transformer architecture. The work is based on the implementation from "[Functional Distributional Semantics at Scale](https://aclanthology.org/2023.starsem-1.37/)" (Lo et al., 2023). Much of the code is adapted from the [corresponding GitHub repository](https://github.com/aaronlolo326/TCSfromDMRS).

Functional Distributional Semantics provides a framework for learning truth-conditional semantic representations from distributional information extracted from DMRS (Dependency Minimal Recursion Semantics) graphs.

## Models

The project includes training scripts for four different models:

### 1. TCS VAE (Variational Autoencoder)
**Script**: `tcs_transformer/scripts/tcs_scripts/train.py`

Similar to the approach in the [Lo et al., 2023](https://aclanthology.org/2023.starsem-1.37/), this model uses a VAE to learn semantic representations of predicate-argument structures from DMRS graphs. The encoder processes predicate-argument pairs into a latent space, and the decoder reconstructs semantic functions.

### 2. GPT-style Transformer
**Script**: `tcs_transformer/scripts/transformer_scripts/train_transformer.py`

A standard decoder-only transformer (GPT-style) for language modeling. This serves as a baseline and can be used independently or as part of the hybrid model.

### 3. Hybrid Transformer-TCS Model
**Script**: `tcs_transformer/scripts/train_hybrid.py`

Combines a GPT-style transformer with the TCS VAE for joint training. The model learns to generate fluent text while respecting semantic structure constraints. Both components are trained end-to-end with a weighted combination of language modeling loss and semantic loss.

### 4. TCS Transformer
**Script**: `tcs_transformer/scripts/train_tcs_transformer.py`

This model takes a different approach from the VAE: instead of learning a latent space, it directly learns to embed semantic functions using a transformer architecture.

**How it works**:
- Takes the same predicate-argument structure data as the TCS VAE
- Flattens the structured data into sequences with special separator tokens between nodes
- Processes these sequences through transformer blocks (self-attention + feed-forward)
- Pools representations for each semantic node (mean pooling over tokens)
- Predicts semantic functions for each node using a linear classifier
- Uses standard cross-entropy loss instead of variational inference

Unlike the VAE's probabilistic approach, this model learns deterministic embeddings directly optimized for semantic function classification. This makes it simpler and faster to train, while still capturing the semantic structure of the input.

## Documentation

For detailed information about each model, training procedures, configuration options, and quick start guides see the `docs/` directory.


## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- ACE Parser with ERG grammar (for data generation)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tcs-transformer
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```
