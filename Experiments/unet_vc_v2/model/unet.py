"""
Residual 1D U-Net for WavLM Feature-Space Voice Conversion (v2).

Changes from v1:
    - Smaller hidden_dim (64 vs 128) to reduce overfitting with 28 patients
    - Higher dropout (0.4 vs 0.25)
    - Otherwise same architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + GELU, with optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size=3, residual=True, dropout=0.0):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.act(self.norm2(self.conv2(h)))
        if self.residual:
            h = h + x
        return h


class ResUNet1D(nn.Module):
    """
    Lightweight residual 1D U-Net for WavLM feature transforms.

    Args:
        feat_dim: WavLM feature dimension (1024)
        hidden_dim: internal channel width (default 64 — smaller than v1's 128)
        n_levels: number of encoder/decoder levels (default 2)
        dropout: dropout rate (default 0.4 — higher than v1's 0.25)
    """

    def __init__(self, feat_dim=1024, hidden_dim=64, n_levels=2, dropout=0.4):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_levels = n_levels

        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, 1)
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)

        # Learnable residual scaling (start small)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_dim
        for i in range(n_levels):
            self.encoders.append(ConvBlock(ch, ch, dropout=dropout))
            self.downsamples.append(
                nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
            )
            ch = ch * 2

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch, dropout=dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_levels):
            self.upsamples.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
            )
            self.decoders.append(ConvBlock(ch, ch // 2, residual=False, dropout=dropout))
            ch = ch // 2

    def forward(self, x):
        T_orig = x.shape[-1]

        divisor = 2 ** self.n_levels
        pad_len = (divisor - T_orig % divisor) % divisor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='reflect')

        h = self.input_proj(x)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        delta = self.output_proj(h)
        out = x + self.alpha * delta
        return out[..., :T_orig]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
