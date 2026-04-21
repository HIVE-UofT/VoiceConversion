"""
Two small MLPs for the ECAPA-space approach:

  bridge_net   : ECAPA emb (192-d) → FreeVC speaker emb (256-d)
                 Learned regression that maps the metric-space embedding into
                 FreeVC's generator conditioning space.

  shift_ecapa  : ECAPA pre emb (192-d) → ECAPA post emb (192-d)
                 Residual MLP that learns the pre→post surgery direction
                 directly in the ECAPA embedding space.

Zero-initialised residual outputs so both start as identity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ShiftEcapa(nn.Module):
    """Residual MLP: pre ECAPA emb → post ECAPA emb (both L2-normed)."""

    def __init__(self, emb_dim=192, hidden_dim=128, n_blocks=2, dropout=0.3):
        super().__init__()
        self.proj_in = nn.Linear(emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_dim, emb_dim)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        out = x + self.alpha * self.proj_out(h)
        return F.normalize(out, dim=-1)


class BridgeEcapaToFreeVC(nn.Module):
    """MLP: ECAPA emb (192) → FreeVC speaker emb (256)."""

    def __init__(self, in_dim=192, out_dim=256, hidden_dim=256, n_blocks=2, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        out = self.proj_out(h)
        return F.normalize(out, dim=-1)  # FreeVC speaker embs are expected on unit sphere
