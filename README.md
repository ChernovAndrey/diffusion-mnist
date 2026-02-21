# Latent Diffusion for MNIST

A minimal implementation of a denoising diffusion probabilistic model (DDPM) that operates in the latent space of a convolutional autoencoder. The denoiser is a transformer with rotary position embeddings (RoPE) and a cosine noise schedule.

## Architecture

The pipeline has three stages:

1. **Convolutional Autoencoder** — compresses 28x28 grayscale images into 49 tokens of dimension 64 (spatial layout 7x7) via two strided convolutions. The latent space is unconstrained (no activation on the bottleneck).

2. **Transformer Denoiser** — 4 layers, 4 heads, pre-norm (RMSNorm), RoPE, SwiGLU MLP. A learnable timestep embedding is prepended as an extra token and stripped from the output.

3. **Diffusion Process** — 100 timesteps, cosine schedule (`alpha_bar`), epsilon-prediction objective. Sampling uses the standard DDPM reverse process.

## Project Structure

| File | Description |
|---|---|
| `AE.py` | Convolutional autoencoder (`ConvAE`) |
| `transformer.py` | Transformer, RoPE, RMSNorm, SwiGLU |
| `diffusion.py` | `DiffusionNet` wrapper, cosine schedule, training/sampling loops |
| `main.ipynb` | End-to-end notebook: train AE, train diffusion, generate samples |

## Quick Start

```bash
uv sync
```

Then open `main.ipynb` and run all cells.

## Dependencies

Managed via `pyproject.toml` (Python >= 3.12):

- torch
- torchvision
- matplotlib
- numpy
- notebook
