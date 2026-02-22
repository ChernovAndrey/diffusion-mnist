# Latent Diffusion for MNIST

A from-scratch implementation of a latent diffusion model (DDPM) for MNIST digit generation. The denoiser is a **DiT-style transformer** (Adaptive LayerNorm-Zero conditioning) operating in the latent space of a convolutional autoencoder.

## Pipeline Overview

```
Training:  image → AE encoder → latent tokens → normalize → corrupt with noise → DiT predicts noise
Sampling:  pure noise → DiT denoises (100 steps) → denormalize → AE decoder → generated image
```

## 1. Convolutional Autoencoder (`AE.py`)

Compresses 28x28 images into a sequence of latent tokens for the diffusion model to operate on.

**Encoder**: `[B, 1, 28, 28]` → Conv(1→32, stride 2) → ReLU → Conv(32→6, stride 2) → `[B, 6, 7, 7]`

The 7x7 spatial grid is reshaped into **49 tokens of dimension 6**: `[B, 6, 7, 7]` → permute → `[B, 49, 6]`

**Decoder**: Inverse — reshape back to `[B, 6, 7, 7]`, then two transposed convolutions back to `[B, 1, 28, 28]`.

The decoder outputs **logits** (no final activation). Training uses `BCEWithLogitsLoss`; sampling applies `sigmoid` to get pixel values in [0, 1].

This is a **plain autoencoder** (not a VAE) — no KL regularization. The latent space is unconstrained to give the diffusion model maximum freedom.

### Latent Normalization

Before diffusion training, we compute per-channel statistics over the full training set:

```
mean = E[z]      over all samples and token positions, shape [6]
std  = Std[z]    over all samples and token positions, shape [6]
```

Latents are normalized to approximately zero mean / unit variance:

```
z_norm = (z - mean) / std
```

This is critical because the diffusion forward process assumes the data lives in a space compatible with standard Gaussian noise.

## 2. Diffusion Process (`diffusion.py`)

### Forward Process (Corruption)

Given a clean latent `z_0` and a timestep `t ∈ {0, 1, ..., T-1}` (T=100), the forward process adds noise:

```
q(z_t | z_0) = N(z_t; sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)
```

Equivalently:  `z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon`,  where `epsilon ~ N(0, I)`

- At t=0: `alpha_bar ≈ 1`, so `z_t ≈ z_0` (almost clean)
- At t=99: `alpha_bar ≈ 0`, so `z_t ≈ epsilon` (almost pure noise)

### Cosine Noise Schedule

`alpha_bar_t` follows the cosine schedule (Nichol & Dhariwal 2021):

```
f(t) = cos((t/T + s) / (1 + s) * pi/2)^2,    s = 0.008
alpha_bar_t = f(t) / f(0)
```

This gives a smoother noise progression than the original linear schedule, with less noise destruction at early timesteps.

Betas are derived: `beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}`, clamped to [0, 0.999].

### Training Objective (Epsilon-Prediction)

The model learns to predict the noise that was added:

```
L = E_{z_0, t, epsilon} [ || epsilon_theta(z_t, t) - epsilon ||^2 ]
```

For each training step:
1. Encode a batch of images → normalize latents
2. Sample random `t ~ Uniform(0, T-1)` per sample
3. Corrupt: `z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon`
4. Predict: `epsilon_pred = DiffusionNet(z_t, t)`
5. Loss: `MSE(epsilon_pred, epsilon)`

### Reverse Process (DDPM Sampling)

Starting from pure noise `z_T ~ N(0, I)`, iteratively denoise:

```
For t = T-1 down to 0:
    epsilon_pred = model(z_t, t)
    mu = (1 / sqrt(alpha_t)) * (z_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred)

    if t > 0:  z_{t-1} = mu + sqrt(beta_t) * N(0, I)
    if t == 0: z_0 = mu
```

Then denormalize and decode: `image = sigmoid(AE.decode(z_0 * std + mean))`

## 3. DiT — Diffusion Transformer (`diffusion.py`, `transformer.py`)

### Why Not a Regular Transformer?

In a naive approach, the timestep embedding is added to the input tokens once, then passes through multiple transformer layers. Each layer's **LayerNorm normalizes away the additive time signal** (it subtracts mean, scales to unit variance). By deeper layers, the model loses awareness of the noise level and learns a useless "average" denoising.

### AdaLN-Zero (Adaptive Layer Normalization)

The solution from the **DiT paper** (Peebles & Xie 2023): instead of fixed LayerNorm parameters, each layer's normalization is **modulated by the timestep**.

Standard LayerNorm: `output = gamma * normalize(x) + beta`  (gamma, beta are fixed learned params)

AdaLN: `output = gamma(t) * normalize(x) + beta(t)`  (gamma, beta are functions of timestep)

Each `DiTBlock` produces 6 modulation vectors from the time embedding `c`:

```
gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = MLP(c)
```

- `gamma, beta` — modulate the LayerNorm output (replace learned affine params)
- `alpha` — **gate** on the residual connection: `output = x + alpha * sublayer(x)`

The gate is the "Zero" part of AdaLN-Zero: initialized to 0, so each block starts as an **identity function** (`x + 0 * sublayer(x) = x`). Blocks gradually "turn on" during training.

### DiTBlock Forward Pass

```
# Attention path
h = LayerNorm(x) * (1 + gamma_1) + beta_1       ← AdaLN modulation
h = MultiHeadAttention(h)
h = Dropout(h)
x = x + alpha_1 * h                              ← gated residual

# FFN path
h = LayerNorm(x) * (1 + gamma_2) + beta_2       ← AdaLN modulation
h = Linear → GELU → Linear (h)
h = Dropout(h)
x = x + alpha_2 * h                              ← gated residual
```

### Full DiffusionNet Architecture

```
Input:  z_t [B, 49, 6]     (noisy latent tokens)
        t   [B]             (integer timestep)

1. in_proj:    z_t → Linear(6, 128)               → [B, 49, 128]
2. t_embed:    t → sinusoidal(128) → MLP(128→512→128)  → c [B, 128]
3. 4x DiTBlock(x, c):  AdaLN-Zero conditioned transformer layers
4. final_adaLN: LayerNorm(x) * (1 + scale(c)) + shift(c)
5. out_proj:   Linear(128, 6), zero-initialized    → [B, 49, 6]

Output: predicted noise epsilon_pred [B, 49, 6]
```

### Sinusoidal Time Embedding

Maps integer timestep to a continuous vector using fixed frequencies (same idea as positional encoding):

```
freq_i = 10000^(-i / (dim/2 - 1))        for i = 0, 1, ..., dim/2 - 1
embedding = [sin(t * freq_0), ..., sin(t * freq_{d/2-1}), cos(t * freq_0), ..., cos(t * freq_{d/2-1})]
```

This gives a 128-dim vector where nearby timesteps have similar representations. A learned MLP (128→512→128) then transforms it into the conditioning vector `c`.

### Multi-Head Attention with RoPE

Standard scaled dot-product attention with Rotary Position Embeddings:

```
Q, K, V = W_q(x), W_k(x), W_v(x)               ← linear projections
Q, K = RoPE(Q, K)                                ← rotate Q, K by position-dependent angles
Attention = softmax(Q @ K^T / sqrt(d_head)) @ V  ← scaled dot-product
Output = W_o(concat(heads))                       ← output projection
```

RoPE encodes relative position by rotating each pair of adjacent dimensions by an angle proportional to the position:

```
theta_i = position / 10000^(2i/dim)
[x_{2i}, x_{2i+1}] → [x_{2i} cos(theta) - x_{2i+1} sin(theta),
                        x_{2i} sin(theta) + x_{2i+1} cos(theta)]
```

No causal mask — all 49 tokens attend to all others (bidirectional, appropriate for diffusion).

Config: 4 heads, 32 dims per head, hidden_size=128.

## 4. Training Details

### Training Loop

- **Optimizer**: AdamW, lr=5e-4
- **LR schedule**: Cosine annealing over 200 epochs (lr decays from 5e-4 to ~0)
- **Gradient clipping**: `clip_grad_norm_(1.0)` — prevents attention gradient spikes
- **Batch size**: 256 for diffusion (gives ~47K gradient steps over 200 epochs)
- **EMA**: Exponential moving average of weights (decay=0.999), loaded after training for sampling

### EMA (Exponential Moving Average)

After each optimizer step:

```
ema_weight = 0.999 * ema_weight + 0.001 * current_weight
```

Training weights oscillate; EMA smooths them. The DDPM sampling chain (100 sequential denoising steps) amplifies any noise in the weights, so EMA is critical for sample quality. After training, EMA weights replace the training weights for inference.

### Zero Initialization Strategy

- `out_proj` weights and biases initialized to zero → model starts predicting zero noise
- AdaLN modulation layers initialized to zero → each block starts as identity
- Combined: initial loss ≈ 1.0 (MSE of predicting 0 vs N(0,1) noise), then decreases as the model learns

## Project Structure

| File | Description |
|---|---|
| `AE.py` | Convolutional autoencoder (`ConvAE`), AE training loop |
| `transformer.py` | `RoPE`, `MHA`, `TransformerBlock`, `Transformer`, custom `LayerNorm`, `Dropout` |
| `diffusion.py` | `SinusoidalTimeEmbedding`, `DiTBlock` (AdaLN-Zero), `DiffusionNet`, cosine schedule, training/sampling |
| `main.ipynb` | End-to-end: train AE → compute latent stats → train diffusion → generate samples |

## Quick Start

```bash
uv sync
```

Then open `main.ipynb` and run all cells.

## Latent Dimension: Why 6 Channels, Not 64

The original implementation used `latent_channels=64`, giving 49x64 = 3,136 latent dimensions — a **4x expansion** of the 28x28x1 = 784 input pixels. The diffusion model (1.4M params) could denoise real images from heavy noise (starting near the data manifold) but failed to generate recognizable digits from pure Gaussian noise. The problem: navigating from random noise onto a thin data manifold in 3,136 dimensions is far harder than in a compressed space.

Reducing to `latent_channels=6` gives 49x6 = 294 latent dimensions — a **~2.7x compression** of the input. This is enough capacity for the AE to reconstruct well, while making the diffusion model's job tractable: it only needs to learn the reverse process in 294 dimensions instead of 3,136.

The key insight is that latent diffusion requires actual **compression**, not just a change of representation. This aligns with how production models work — e.g., NVIDIA's Cosmos uses a plain AE (not a VAE) with strong spatial compression, and the diffusion model works well because the latent space is genuinely lower-dimensional than the input.

**Diagnostic that revealed the issue**: encoding real images, noising them to t=75 (heavy noise), and running the reverse process produced clear digits — proving the diffusion model had learned accurate denoising. But sampling from pure noise (t=T) produced fragmented output. This gap between "denoise from near the manifold" vs. "navigate to the manifold from scratch" is the hallmark of an insufficiently compressed latent space.

## Key References

- **DDPM**: Ho et al. 2020 — *Denoising Diffusion Probabilistic Models*
- **Improved DDPM**: Nichol & Dhariwal 2021 — cosine schedule, improved training
- **Latent Diffusion**: Rombach et al. 2022 — diffusion in autoencoder latent space
- **DiT**: Peebles & Xie 2023 — *Scalable Diffusion Models with Transformers* (AdaLN-Zero)
- **RoPE**: Su et al. 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*

## Dependencies

Managed via `pyproject.toml` (Python >= 3.12):

- torch
- torchvision
- matplotlib
- numpy
- notebook
