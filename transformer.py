import torch
from torch import nn


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("p must be in [0, 1).")
        self.p = p
        self.inverse_p = 1.0 - p

    def forward(self, x: torch.Tensor):
        if self.training and self.p != 0.0:
            mask = (torch.rand_like(x) < self.inverse_p).to(x.dtype)
            return x * mask / self.inverse_p
        else:
            return x


class LayerNorm(nn.Module):
    def __init__(self, d: int, affine: bool = True):
        super().__init__()
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(d))
            self.beta = nn.Parameter(torch.zeros(d))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mu) / torch.sqrt(var + 1e-5)
        if self.affine:
            out = self.gamma * out + self.beta
        return out


class MHA(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.O = nn.Linear(hidden_size, hidden_size)

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.d_heads = hidden_size // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x dim: [B, N, D]
        B, N, _ = x.shape
        q_proj = self.Q(x)  # [B, N, D]
        k_proj = self.K(x)  # [B, N, D]
        v_proj = self.V(x)  # [B, N, D]

        q_proj = q_proj.view(B, N, self.n_heads, self.d_heads).transpose(1, 2)  # [B, nh, N, dh]
        k_proj = k_proj.view(B, N, self.n_heads, self.d_heads).transpose(1, 2)  # [B, nh, N, dh]
        v_proj = v_proj.view(B, N, self.n_heads, self.d_heads).transpose(1, 2)  # [B, nh, N, dh]

        attn_score = torch.matmul(q_proj, k_proj.transpose(-1, -2))  # [B, nh, N, N]
        attn_score = torch.softmax(attn_score / (self.d_heads ** 0.5), dim=-1)

        out = torch.matmul(attn_score, v_proj)  # [B, nh, N, dh]
        out = out.transpose(1, 2)  # [B, N, nh, dh]
        out = out.contiguous().view(B, N, -1)
        o_proj = self.O(out)
        return o_proj


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout_rate, mlp_size):
        super().__init__()
        self.MHA = MHA(hidden_size, n_heads)
        self.LN1 = LayerNorm(hidden_size)
        self.LN2 = LayerNorm(hidden_size)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.LN1(x)
        out = self.MHA(out)
        out = self.dropout1(out)
        out = out + x

        out2 = self.LN2(out)
        out2 = self.ffn(out2)
        out2 = self.dropout2(out2)

        return out2 + out


class Transformer(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int, n_heads: int, dropout_rate: float, mlp_size: int):
        super().__init__()

        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size

        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_size, n_heads, dropout_rate, mlp_size)
             for _ in range(self.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x