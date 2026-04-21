"""
Small residual MLP that learns the pre -> post surgery direction in
FreeVC's speaker-embedding space (256-d). This is the only learnable
component sitting on top of the frozen FreeVC foundation model.

Inputs/outputs are L2-normalised 256-d speaker embeddings.
Output is residual: post_spk_pred = pre_spk + alpha * delta(pre_spk).
Zero-initialised output layer so training starts as identity.
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


class SurgeryShift(nn.Module):
    """Residual MLP that predicts post speaker embedding from pre embedding.
    Starts as identity (zero-init output). Operates on L2-normalised embeddings."""

    def __init__(self, emb_dim=256, hidden_dim=128, n_blocks=2, dropout=0.3):
        super().__init__()
        self.proj_in = nn.Linear(emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(hidden_dim, emb_dim)
        # Zero-init so the residual starts at 0, i.e., the identity map.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        # Small learnable scale on the residual so it ramps up smoothly.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: (B, emb_dim), assumed L2-normalised
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        delta = self.proj_out(h)
        out = x + self.alpha * delta
        # Re-normalise so the output stays on the unit sphere (ECAPA/SE space convention).
        out = F.normalize(out, dim=-1)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
