import torch
from torch import nn
from transformer import Transformer
import torch.nn.functional as F


class DiffusionNet(nn.Module):
    def __init__(self, token_dim, hidden_size, n_layers, n_heads, dropout_rate, mlp_size, T):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, hidden_size)
        self.t_embed = nn.Embedding(T, hidden_size)
        self.transformer = Transformer(hidden_size, n_layers, n_heads, dropout_rate, mlp_size)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, token_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, N, token_dim], t: [B]
        x = self.in_proj(x)              # [B, N, hidden]
        te = self.t_embed(t).unsqueeze(1) # [B, 1, hidden]
        x = torch.cat([te, x], dim=1)    # [B, N+1, hidden]
        x = self.transformer(x)
        x = x[:, 1:, :]                  # strip time token
        return self.out_proj(self.final_norm(x))  # [B, N, token_dim]


def cosine_alpha_bar(T: int, s: float = 0.008, device=None):
    steps = torch.arange(T + 1, dtype=torch.float32, device=device) / T
    f = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    return f / f[0]  # [T+1]


def compute_alpha_bar(T: int, device=None):
    alpha_bar_full = cosine_alpha_bar(T, device=device)  # [T+1]
    betas = 1 - (alpha_bar_full[1:] / alpha_bar_full[:-1])  # [T]
    betas = betas.clamp(0, 0.999)
    alphas = 1 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)  # [T]
    return betas, alphas, alpha_bar


@torch.no_grad()
def compute_latent_stats(ae, dataloader, device='cpu'):
    """Compute per-channel mean and std of AE latents over the training set."""
    ae.to(device)
    ae.eval()
    all_z = []
    for x_img, _ in dataloader:
        z = ae.encode(x_img.to(device))  # [B, 49, 64]
        all_z.append(z)
    all_z = torch.cat(all_z, dim=0)       # [N_total, 49, 64]
    mean = all_z.mean(dim=(0, 1))          # [64]
    std = all_z.std(dim=(0, 1))            # [64]
    return mean, std


def corrupt(x: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor):
    """
    x: [B, ...]
    t: [B] int64 timesteps in [0, T-1]
    alpha_bar: [T]
    """
    eps = torch.randn_like(x)
    a_bar = alpha_bar[t].view(x.size(0), *([1] * (x.ndim - 1)))  # [B,1,1,...]
    zt = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * eps
    return zt, eps


@torch.no_grad()
def sample(diff_model, ae, n_samples, T, n_tokens, token_dim, latent_mean, latent_std, device='cpu'):
    """DDPM reverse sampling: noise → latent tokens → decoded images.

    Implements the reverse diffusion process to generate images:
    1. Start from pure Gaussian noise z_T ~ N(0, I) in latent token space
    2. For t = T-1 down to 0: denoise one step using the trained DiffusionNet
    3. Decode the final clean latent z_0 with the AE decoder

    Args:
        diff_model: trained DiffusionNet
        ae: trained ConvAE (frozen, used only for decoding)
        n_samples: number of images to generate
        T: number of diffusion timesteps (must match training T)
        n_tokens: number of tokens per sample (ae.n_tokens = 49)
        token_dim: dimension per token (ae.token_dim = 64)
        device: 'cpu', 'cuda', or 'mps'

    Returns:
        images: [n_samples, 1, 28, 28] tensor with pixel values in [0, 1]

    TODO (you implement):
        1. Precompute betas, alphas, alpha_bar using compute_alpha_bar(T, device)
        2. Sample initial noise: z_t = torch.randn(n_samples, n_tokens, token_dim)
        3. Reverse loop from t = T-1 down to 0:
           a. Predict noise: eps_pred = diff_model(z_t, t_batch)
           b. Compute DDPM mean:
              coeff = beta_t / sqrt(1 - alpha_bar_t)
              mean = (1 / sqrt(alpha_t)) * (z_t - coeff * eps_pred)
           c. If t > 0: add noise z_{t-1} = mean + sqrt(beta_t) * N(0, I)
              If t == 0: z_0 = mean (no noise at final step)
        4. Decode: images = sigmoid(ae.decode(z_0))
        5. Return images
    """
    diff_model.eval()
    ae.eval()
    betas, alphas, alpha_bar = compute_alpha_bar(T, device=device)
    z = torch.randn(n_samples, n_tokens, token_dim).to(device)
    for i in range(T-1, -1, -1):
        eps_pred = diff_model(z, torch.full((n_samples,), i, device=device, dtype=torch.long))
        coeff = betas[i] / torch.sqrt(1- alpha_bar[i])
        mean = (1 / torch.sqrt(alphas[i])) * (z - coeff * eps_pred)
        if i > 0:
            z = mean + torch.sqrt(betas[i]) * torch.randn_like(mean)
        else:
            z = mean

    # denormalize back to AE latent space
    z = z * latent_std.to(device) + latent_mean.to(device)
    return torch.sigmoid(ae.decode(z))

def train_diffusion(diff_model, ae, n_epochs, train_dataloader, T, lr: float,
                    latent_mean=None, latent_std=None, device="cpu"):
    diff_model.to(device)
    ae.to(device)
    ae.eval()  # freeze AE behavior

    # compute latent stats if not provided
    if latent_mean is None or latent_std is None:
        latent_mean, latent_std = compute_latent_stats(ae, train_dataloader, device)
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    # precompute schedule once
    betas, alphas, alpha_bar = compute_alpha_bar(T, device=device)

    opt = torch.optim.AdamW(diff_model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        diff_model.train()
        total_loss = 0.0
        total_items = 0

        for x_img, _ in train_dataloader:
            x = x_img.to(device)  # [B, 1, 28, 28]

            with torch.no_grad():
                z0 = ae.encode(x)  # [B, N, D]  (tokens)
                z0 = (z0 - latent_mean) / latent_std  # normalize

            B = z0.size(0)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            zt, eps = corrupt(z0, t, alpha_bar)     # both [B, N, D]
            eps_pred = diff_model(zt, t)            # [B, N, D]

            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            total_items += B

        print(f"epoch {epoch}/{n_epochs} done - avg loss={total_loss / total_items:.4f}")