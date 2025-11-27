# TCS Variational Autoencoder Architecture

## Overview

The TCS (Truth Conditional Semantics) VAE is a variational autoencoder designed to learn semantic representations of predicate-argument structures. It consists of three main components:

1. **Encoder** (PASEncoder or MyEncoder): Encodes predicate-argument structures into latent embeddings
2. **Latent Space**: Gaussian variational representations with mean (μ) and variance (σ²)
3. **Decoder** (OneLayerSemFuncsDecoder): Decodes embeddings into semantic plausibility scores

---

## 1. Encoder Architecture

### Input to Encoder

The encoder receives **predicate-argument structures** extracted from DMRS (Dependency Minimal Recursion Semantics) graphs.

#### PASEncoder Input Format:
```python
# Input: pred_func_nodes_ctxt_predargs
# Shape: (num_nodes, num_pred_args)
# Example: Tensor of predicate-argument pair indices

# Each row represents a predicate node
# Each column represents an argument slot (ARG0, ARG1, ARG2, etc.)
# Values are indices into a vocabulary of predicate@argument combinations

pred_func_nodes_ctxt_predargs = [
    [pred1@ARG0, pred1@ARG1, pred1@ARG2, ...],  # Node 1
    [pred2@ARG0, pred2@ARG1, 0, ...],            # Node 2 (padded)
    ...
]

# Also receives:
pred_func_nodes_ctxt_predargs_len = [num_args_node1, num_args_node2, ...]  # Actual lengths before padding
```

### Encoder Processing

**PASEncoder** (`tcs_model.py:127-176`):

```python
def forward(self, pred_func_nodes_ctxt_predargs, pred_func_nodes_ctxt_predargs_len, device, train_mode=1):
    # 1. Dropout on ARG0 (for regularization during training)
    keep_arg0s = 1 - Bernoulli(dropout_prob).sample([num_nodes]) * train_mode
    pred_func_nodes_ctxt_predargs[:, 0] *= keep_arg0s  # Zero out some ARG0s
    
    # 2. Embed predicate-argument pairs
    pred_arg_embs = self.embeds(pred_func_nodes_ctxt_predargs + 1)
    # Shape: (num_nodes, num_pred_args, input_dim)
    
    # 3. Mean pooling across arguments (with padding correction)
    mean_a_padded = torch.mean(pred_arg_embs, dim=1)
    mean_a = mean_a_padded * (num_pred_args / actual_lengths).unsqueeze(-1)
    # Shape: (num_nodes, input_dim)
    
    # 4. Apply activation
    a = ReLU(mean_a)  # or other activation
    h = a
    # Shape: (num_nodes, input_dim) -> hidden representation
    
    # 5. Project to latent space
    mu = self.fc_mu(h)  # Shape: (num_nodes, mu_dim)
    log_sigma2 = self.fc_sigma(h)  # Shape: (num_nodes, sigma_dim)
    
    # 6. Add batch dimension
    mu_batch = mu.unsqueeze(0)  # Shape: (1, num_nodes, mu_dim)
    log_sigma2_batch = log_sigma2.unsqueeze(0)  # Shape: (1, num_nodes, sigma_dim)
    
    return mu_batch, log_sigma2_batch
```

**Key Points**:
- Each predicate node gets its own latent representation
- Embedding dimension: `input_dim` (typically 50-200)
- Latent dimension: `mu_dim` (typically 10-50)
- ARG0 dropout: Prevents overfitting by randomly zeroing primary arguments

---

## 2. Latent Space Representation

### Gaussian Latent Variables

The encoder outputs parameters of a Gaussian distribution for each predicate node:

```python
# For each node i:
z_i ~ N(μ_i, σ²_i)

# Where:
μ_i = [μ_i1, μ_i2, ..., μ_i_d]        # Mean vector (mu_dim dimensions)
σ²_i = [σ²_i1, σ²_i2, ..., σ²_i_d]    # Variance vector (diagonal covariance)
```

### Latent Space Shapes

```python
mu_batch.shape = (batch_size, num_nodes, mu_dim)
sigma2_batch.shape = (batch_size, num_nodes, mu_dim)

# Example with batch_size=1, num_nodes=5, mu_dim=20:
mu_batch = [
    [
        [0.12, -0.34, 0.56, ...],  # Node 0: 20-dim mean vector
        [0.78, 0.21, -0.45, ...],  # Node 1: 20-dim mean vector
        [-0.33, 0.89, 0.12, ...],  # Node 2: 20-dim mean vector
        [0.45, -0.67, 0.23, ...],  # Node 3: 20-dim mean vector
        [0.91, 0.34, -0.78, ...],  # Node 4: 20-dim mean vector
    ]
]

sigma2_batch = [
    [
        [0.01, 0.02, 0.01, ...],  # Node 0: 20-dim variance vector
        [0.03, 0.01, 0.02, ...],  # Node 1: 20-dim variance vector
        [0.02, 0.01, 0.03, ...],  # Node 2: 20-dim variance vector
        [0.01, 0.02, 0.01, ...],  # Node 3: 20-dim variance vector
        [0.02, 0.03, 0.01, ...],  # Node 4: 20-dim variance vector
    ]
]
```

### Sampling (Reparameterization Trick)

```python
def sample_from_gauss(mu_batch, sigma2_batch, num_samples=1):
    # Sample ε ~ N(0, 1)
    sample_eps = N(0, 1).sample([batch_size, num_nodes])
    
    # Reparameterization: z = μ + σ * ε
    sample_zs = mu_batch + torch.sqrt(sigma2_batch) * sample_eps
    
    return sample_zs  # Shape: (batch_size, num_nodes, mu_dim)
```

**Interpretation**:
- Each node's embedding captures semantic properties of that predicate
- Variance represents uncertainty in the semantic representation
- Similar predicates should have nearby embeddings in latent space

---

## 3. Decoder Architecture

The decoder (`OneLayerSemFuncsDecoder`) computes **semantic plausibility scores** for predicate-argument structures.

### Decoder Input Format

```python
# Decoder receives:
decoder_input = {
    "mu_batch": mu_batch,           # Latent means (1, num_nodes, mu_dim)
    "sigma2_batch": sigma2_batch,   # Latent variances (1, num_nodes, mu_dim)
    "logic_expr": logic_expr,       # Semantic functions to evaluate
    "args_vars": args_vars          # Variable assignments
}
```

#### Logic Expression Format (Truth Mode):

```python
# logic_expr: Tensor of semantic function indices
# Shape: (batch_size, num_semantic_functions)
# Each index refers to a learned semantic function (predicate@argument combination)

logic_expr = [
    [sem_func_idx_1, sem_func_idx_2, sem_func_idx_3, ...]
]

# Example semantic functions:
# 0: "_cat_n@ARG0"           - "x is a cat"
# 1: "_chase_v@ARG0_ARG1"    - "x chases y"
# 2: "_mouse_n@ARG0"         - "x is a mouse"
```

#### Variable Assignments Format:

```python
# args_vars: Maps semantic functions to node indices
# Shape: (batch_size, 2, num_semantic_functions)
# args_vars[0] = x indices (first argument)
# args_vars[1] = y indices (second argument)

args_vars = [
    [
        [node_idx_x1, node_idx_x2, node_idx_x3, ...],  # X variables
        [node_idx_y1, node_idx_y2, node_idx_y3, ...],  # Y variables
    ]
]

# Example:
# Semantic function: "_chase_v@ARG0_ARG1" (x chases y)
# args_vars[0][i] = 2  # x = node 2 (e.g., "cat")
# args_vars[1][i] = 3  # y = node 3 (e.g., "mouse")
```

### Decoder Processing (Truth Mode)

**Forward Pass** (`tcs_model.py:388-472`):

```python
def _forward_truth(mu_batch, sigma2_batch, logic_expr, args_vars, samp_neg, variational, device):
    # Unpack batch (currently batch_size=1)
    mu = mu_batch[0]  # Shape: (num_nodes, mu_dim)
    sigma2 = sigma2_batch[0]  # Shape: (num_nodes, mu_dim)
    pos_sem_funcs = logic_expr[0]  # Semantic function indices
    args_vars = args_vars[0]  # Variable assignments
    
    # 1. Pad embeddings with zero (for padding index)
    mu_pad = F.pad(mu, (0, 0, 1, 0), "constant", 0)  # Add zero row at start
    sigma2_pad = F.pad(sigma2, (0, 0, 1, 0), "constant", 0)
    
    # 2. Select embeddings for x and y arguments
    mu_x = mu_pad[args_vars[0]]  # Shape: (num_funcs, mu_dim)
    mu_y = mu_pad[args_vars[1]]  # Shape: (num_funcs, mu_dim)
    mu_xy = torch.cat([mu_x, mu_y], dim=1)  # Shape: (num_funcs, 2*mu_dim)
    mu_xy_bias = F.pad(mu_xy, (0, 1), "constant", 1)  # Add bias term
    # Shape: (num_funcs, 2*mu_dim + 1)
    
    # 3. Same for variances
    sigma2_x = sigma2_pad[args_vars[0]].expand([-1, mu_dim])
    sigma2_y = sigma2_pad[args_vars[1]].expand([-1, mu_dim])
    sigma2_xy = torch.cat([sigma2_x, sigma2_y], dim=1)
    sigma2_xy_bias = F.pad(sigma2_xy, (0, 1), "constant", 0)
    
    # 4. Get learned semantic function weights
    sem_funcs_w = self.sem_funcs[pos_sem_funcs]
    # Shape: (num_funcs, 2*mu_dim + 1)
    # These are learned weight vectors for each semantic function
    
    # 5. Compute linear combination: μ_a = w^T · μ
    mu_a = torch.sum(mu_xy_bias * sem_funcs_w, dim=1)
    # Shape: (num_funcs,)
    
    # 6. Compute variance: σ²_a = w^T · Σ · w (diagonal covariance)
    sem_funcs_w2 = sem_funcs_w * sem_funcs_w
    sigma2_a = torch.sum(sigma2_xy_bias * sem_funcs_w2, dim=1)
    # Shape: (num_funcs,)
    
    # 7. Apply sigmoid with probit approximation (variational)
    if variational:
        kappa = 1 / torch.sqrt(1 + π * sigma2_a / 8)  # Probit approximation
        truths = torch.sigmoid(kappa * mu_a)
    else:
        truths = torch.sigmoid(mu_a)
    
    # Shape: (num_funcs,) - truth values in [0, 1]
    
    # 8. Positive log-likelihood
    pos_log_truths = torch.log(truths)  # log P(semantic_function is true)
    
    # 9. Sample negative examples (if training with contrastive loss)
    if samp_neg:
        # Sample random semantic functions not in the positive set
        neg_sem_funcs = sample_negative_functions(num_negative_samples)
        
        # Compute for negative samples (similar to above, but negate before sigmoid)
        neg_truths = torch.sigmoid(-kappa * mu_a_neg)  # P(negative function is false)
        neg_log_truths = torch.log(neg_truths)
        
        # Combine positive and negative
        pos_neg_log_truths = torch.cat([pos_log_truths, neg_log_truths])
        pos_neg_log_truth = torch.sum(pos_neg_log_truths)
    
    return pos_neg_log_truth_batch, pos_sum, neg_sum
```

### Semantic Function Representation

Each semantic function is a learned weight vector:

```python
# Semantic functions are stored as:
self.sem_funcs = nn.Parameter(
    torch.empty(num_sem_funcs, 2*mu_dim + 1)
)

# Example:
# sem_func["_cat_n@ARG0"] = [w1, w2, ..., w_{2*mu_dim}, bias]
# Shape: (2*mu_dim + 1,)

# This weight vector defines a linear classifier in latent space:
# score = w^T · [μ_x; μ_y; 1]
# truth_value = sigmoid(score)
```

**Interpretation**:
- Each semantic function learns a hyperplane in latent space
- High score = predicate-argument structure is semantically plausible
- Low score = structure is implausible
- Weights are learned during training

### Decoder Output

```python
# Output shape: (batch_size,)
log_truth_batch = sum(log P(positive_functions)) + sum(log P(¬negative_functions))

# Also returns:
pos_sum = sum(log P(positive_functions))    # For monitoring
neg_sum = sum(log P(¬negative_functions))   # For monitoring
```

---

## 4. Loss Calculation

The VAE is trained with a combination of **reconstruction loss** (semantic plausibility) and **KL divergence** (regularization).

### ELBO (Evidence Lower Bound)

```python
def run(encoder_data, decoder_data):
    # 1. Encode to latent
    mu_batch, log_sigma2_batch = encoder(*encoder_data)
    sigma2_batch = torch.exp(log_sigma2_batch)
    
    # 2. Decode to truth values
    log_truth_batch, pos_sum, neg_sum = decoder(
        mu_batch, sigma2_batch, *decoder_data
    )
    
    # 3. KL divergence (regularization toward N(0,1) prior)
    if variational:
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl_div = 0.5 * (
            torch.sum(sigma2_batch, dim=(1,2)) * mu_dim +      # Σ σ²
            torch.sum(torch.square(mu_batch), dim=(1,2)) -     # Σ μ²
            num_nodes * mu_dim -                                # -k
            torch.sum(log_sigma2_batch, dim=(1,2)) * mu_dim    # -Σ log(σ²)
        )
    else:
        kl_div = 0.0
    
    return log_truth_batch, kl_div, pos_sum, neg_sum, mu_batch, sigma2_batch
```

### Total Loss

```python
# In trainer:
log_truth_batch, kl_div, _, pos_sum, neg_sum, _, _ = model.run(**batch)

# Reconstruction loss (maximize log-likelihood)
reconstruction_loss = -log_truth_batch.mean()

# KL divergence loss (with annealing)
kl_loss = beta * kl_div.mean()

# Total loss (negative ELBO)
total_loss = reconstruction_loss + kl_loss

# Optimize:
total_loss.backward()
optimizer.step()
```

### Beta Annealing

```python
# Beta starts small and increases during training
beta = model.start_beta + (model.end_beta - model.start_beta) * min(epoch / model.end_beta_epoch, 1.0)

# Typical values:
# start_beta = 0.0    (no KL penalty initially)
# end_beta = 0.01     (small regularization)
# end_beta_epoch = 50 (ramp up over 50 epochs)
```

**Why?**: Starting with low beta allows the model to learn good reconstructions first, then gradually adds regularization.

---

## 5. End-to-End Data Flow

### Complete Training Example

```python
# 1. Input: Predicate-argument structure from DMRS
input_data = {
    "encoder": [
        pred_func_nodes_ctxt_predargs,     # (num_nodes, num_args) - indices
        pred_func_nodes_ctxt_predargs_len  # (num_nodes,) - actual lengths
    ],
    "decoder": [
        logic_expr,  # (1, num_sem_funcs) - semantic function indices
        args_vars    # (1, 2, num_sem_funcs) - variable assignments
    ]
}

# 2. Encoder: predicate-argument indices → latent embeddings
mu_batch, log_sigma2_batch = encoder(
    pred_func_nodes_ctxt_predargs,
    pred_func_nodes_ctxt_predargs_len
)
# mu_batch.shape: (1, num_nodes, mu_dim)
# log_sigma2_batch.shape: (1, num_nodes, mu_dim)

# 3. Latent space: Gaussian distribution per node
sigma2_batch = torch.exp(log_sigma2_batch)
# Each node has: z_i ~ N(μ_i, σ²_i)

# 4. Decoder: latent embeddings → semantic plausibility
log_truth_batch, pos_sum, neg_sum = decoder(
    mu_batch,
    sigma2_batch,
    logic_expr,
    args_vars,
    samp_neg=True,
    variational=True
)
# log_truth_batch.shape: (1,) - scalar log-probability

# 5. Loss calculation
kl_div = compute_kl_divergence(mu_batch, sigma2_batch)
reconstruction_loss = -log_truth_batch.mean()
kl_loss = beta * kl_div.mean()
total_loss = reconstruction_loss + kl_loss

# 6. Backpropagation
total_loss.backward()
optimizer.step()
```

### Dimensions Summary

```
Input:
  - Predicate-argument indices: (num_nodes, num_args)
  
Encoder Output:
  - μ: (1, num_nodes, mu_dim)          e.g., (1, 10, 20)
  - log(σ²): (1, num_nodes, mu_dim)    e.g., (1, 10, 20)
  
Latent Space:
  - Each node: z ~ N(μ, σ²) where μ, σ² ∈ ℝ^{mu_dim}
  
Decoder Input:
  - μ, σ²: From encoder
  - Semantic functions: (1, num_sem_funcs)
  - Variable assignments: (1, 2, num_sem_funcs)
  
Decoder Output:
  - log_truth: (1,) - scalar
  - pos_sum: (1,)
  - neg_sum: (1,)
  
Loss:
  - Scalars: reconstruction_loss, kl_loss, total_loss
```

---

## 6. Key Insights

### What the Model Learns

1. **Encoder learns**:
   - How to embed predicate-argument pairs into semantic space
   - Similar semantic concepts should have nearby embeddings
   - Example: "_cat_n@ARG0" and "_kitten_n@ARG0" should be close

2. **Decoder learns**:
   - Which combinations of predicates are semantically plausible
   - Linear separability in latent space for semantic functions
   - Example: "cat chases mouse" (plausible) vs "mouse chases cat" (less plausible)

3. **Latent space captures**:
   - Semantic properties of predicates
   - Uncertainty via variance
   - Compositionality: combining predicates via learned weights

### Variational Component

The variational aspect (σ²) provides:
- **Uncertainty quantification**: How confident is the model?
- **Regularization**: Prevents overfitting via KL penalty
- **Probit approximation**: `σ(κμ)` where `κ = 1/√(1 + πσ²/8)`
  - Accounts for variance when computing truth values
  - More variance → less confident predictions

### Negative Sampling

During training with `samp_neg=True`:
- Samples random semantic functions as negative examples
- Learns to distinguish plausible from implausible structures
- Typical ratio: 1 positive to 10-100 negatives
- Uses frequency-based sampling (more common predicates sampled more often)

---

## 7. Training Configuration

### Typical Hyperparameters

```json
{
  "encoder_arch": {
    "type": "PASEncoder",
    "args": {
      "input_dim": 100,      // Embedding dimension
      "hidden_dim": 50,      // Hidden layer size
      "mu_dim": 20,          // Latent dimension
      "sigma_dim": 20,       // Variance dimension
      "dropout_prob": 0.5,   // ARG0 dropout rate
      "hidden_act_func": "ReLU"
    }
  },
  "decoder_arch": {
    "type": "OneLayerSemFuncsDecoder",
    "args": {
      "freq_sampling": true,          // Frequency-based negative sampling
      "num_negative_samples": 10,     // Negatives per positive
      "contrastive_loss": true,       // Use contrastive formulation
      "alpha": 1.0                     // Loss weight
    }
  },
  "autoencoder_arch": {
    "type": "VarAutoencoder",
    "args": {
      "start_beta": 0.0,        // Initial KL weight
      "end_beta": 0.01,         // Final KL weight
      "end_beta_epoch": 50,     // Annealing duration
      "variational": true,      // Use variational inference
      "with_logic": false       // Truth mode (not logic mode)
    }
  }
}
```

---

## 8. Summary

**Input** → Predicate-argument structure indices

**Encoder** → Latent Gaussian embeddings (μ, σ²) for each predicate node

**Latent Space** → Continuous semantic representation with uncertainty

**Decoder** → Semantic plausibility scores via learned linear classifiers

**Output** → Log-likelihood of semantic structures

**Loss** → -log_likelihood + β * KL_divergence

**Result** → Model learns to score semantic plausibility of predicate-argument structures

The TCS VAE effectively learns a **probabilistic semantic space** where semantically plausible predicate-argument combinations score highly, and implausible ones score low.
