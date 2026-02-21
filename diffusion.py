import torch
from torch import nn
from transformer import Transformer
import torch.nn.functional as F


class DiffusionNet(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads, dropout_rate, mlp_size, T):
        super().__init__()
        self.hidden_size = hidden_size
        self.t_embed = nn.Embedding(T, hidden_size)

        # produce gamma and beta from t embedding
        self.film = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
        )

        self.transformer = Transformer(hidden_size, n_layers, n_heads, dropout_rate, mlp_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D], t: [B]
        te = self.t_embed(t)  # [B,D]
        gb = self.film(te)  # [B,2D]
        gamma, beta = gb.chunk(2, dim=-1)  # each [B,D]

        gamma = gamma.unsqueeze(1)  # [B,1,D]
        beta = beta.unsqueeze(1)  # [B,1,D]

        x = x * (1.0 + gamma) + beta
        return self.transformer(x)


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



def train_diffusion(diff_model, ae, n_epochs, train_dataloader, T, lr: float, device="cpu"):
    diff_model.to(device)
    ae.to(device)
    ae.eval()  # freeze AE behavior

    # precompute schedule once
    betas, alphas, alpha_bar = compute_alpha_bar(T, device=device)

    opt = torch.optim.AdamW(diff_model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        diff_model.train()
        total_loss = 0.0
        total_items = 0

        for x_img, _ in train_dataloader:
            x = x_img.to(device).view(x_img.size(0), -1)  # [B, 784]

            with torch.no_grad():
                z0 = ae.encode(x)  # [B, N, D]  (tokens)

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
