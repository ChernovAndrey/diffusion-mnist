import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, in_dim=784, n_tokens=16, token_dim=8, hidden_dim=64):
        super().__init__()
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.latent_dim = n_tokens * token_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def encode(self, x):
        z = self.encoder(x)                  # [B, latent_dim]
        return z.view(x.size(0), self.n_tokens, self.token_dim)

    def decode(self, z_tokens):
        z = z_tokens.view(z_tokens.size(0), -1)
        return self.decoder(z)

    def forward(self, x):
        # x expected shape: [B, 784]
        z_tokens = self.encode(x)        # [B, N, D]
        logits = self.decode(z_tokens)   # [B, 784]
        return logits, z_tokens

def train_AE(model: AE, n_epochs: int, train_dataloader: DataLoader, lr: float, device='cpu'):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for x, _ in train_dataloader:
            x = x.view(x.size(0), -1).to(device)
            logits, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, x, reduction="mean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = x.size(0)
            total_loss += loss.item() * bs  # accumulate per-sample
            total_items += bs

        avg_epoch = total_loss / total_items
        print(f"epoch {i}/{n_epochs} done - avg loss={avg_epoch:.4f}")
