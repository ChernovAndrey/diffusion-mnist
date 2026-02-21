import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ConvAE(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_h = 7
        self.latent_w = 7
        self.n_tokens = self.latent_h * self.latent_w   # 49
        self.token_dim = latent_channels                  # 64

        # [B, 1, 28, 28] → [B, 32, 14, 14] → [B, C, 7, 7]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=2, padding=1),
            # no activation — latent space should be unconstrained for diffusion
        )

        # [B, C, 7, 7] → [B, 32, 14, 14] → [B, 1, 28, 28]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # no activation — output is logits for BCEWithLogitsLoss
        )

    def encode(self, x):
        # x: [B, 1, 28, 28]
        z = self.encoder(x)                              # [B, C, 7, 7]
        B, C, H, W = z.shape
        return z.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 49, C]

    def decode(self, z_tokens):
        # z_tokens: [B, 49, C]
        B = z_tokens.size(0)
        z = z_tokens.reshape(B, self.latent_h, self.latent_w, self.latent_channels)
        z = z.permute(0, 3, 1, 2)                        # [B, C, 7, 7]
        return self.decoder(z)                            # [B, 1, 28, 28]

    def forward(self, x):
        # x: [B, 1, 28, 28]
        z_tokens = self.encode(x)                         # [B, 49, 64]
        recon = self.decode(z_tokens)                     # [B, 1, 28, 28]
        return recon, z_tokens


def train_AE(model: ConvAE, n_epochs: int, train_dataloader: DataLoader, lr: float, device='cpu'):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for x, _ in train_dataloader:
            x = x.to(device)                              # [B, 1, 28, 28]
            recon, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(recon, x, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_items += bs

        avg_epoch = total_loss / total_items
        print(f"epoch {i}/{n_epochs} done - avg loss={avg_epoch:.4f}")