"""
ECAPA Embedding Mapper: pre-surgery -> post-surgery speaker embedding.

Models:
  EcapaMapper   – residual MLP (project-up -> res blocks -> project-down + skip)
  LinearMapper  – single residual linear layer (out = x + W*x + b)
                  much lower capacity; often better with very few training samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class EcapaMapper(nn.Module):
    """
    Residual MLP: pre_ecapa (192) -> predicted post_ecapa (192).

    Architecture:
        project up -> n_blocks residual blocks -> project back down
        final output: x + delta(x)  (residual from input)

    Defaults tuned for ~20 training samples: small hidden_dim, strong dropout.
    """

    def __init__(self, emb_dim=192, hidden_dim=64, n_blocks=1, dropout=0.5):
        super().__init__()
        self.proj_in  = nn.Linear(emb_dim, hidden_dim)
        self.blocks   = nn.ModuleList(
            [ResBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.proj_out = nn.Linear(hidden_dim, emb_dim)
        # Zero-init -> starts as identity
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return x + self.proj_out(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearMapper(nn.Module):
    """
    Residual linear mapper: out = x + W*x + b.

    Lowest possible capacity — often the right choice with < 30 training samples.
    37K parameters vs ~25K for small EcapaMapper, but strictly linear so
    it cannot overfit non-linear structure.
    """

    def __init__(self, emb_dim=192):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        # Zero-init -> starts as identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + self.linear(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
