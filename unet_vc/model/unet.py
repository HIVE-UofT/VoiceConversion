"""
Residual 1D U-Net for WavLM Feature-Space Voice Conversion.

Architecture:
    Input (B, 1024, T)
    → project to hidden dim (256)
    → encoder (downsample T by 2x at each level)
    → bottleneck
    → decoder (upsample + skip connections)
    → project back to 1024
    → global residual: output = input + alpha * network(input)

The residual design means the network only learns the small delta
between pre and post surgery domains, not the full mapping.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + GELU, with optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size=3, residual=True):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        if self.residual:
            h = h + x
        return h


class ResUNet1D(nn.Module):
    """
    Lightweight residual 1D U-Net for WavLM feature transforms.

    Args:
        feat_dim: WavLM feature dimension (1024)
        hidden_dim: internal channel width (default 256)
        n_levels: number of encoder/decoder levels (default 3)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, feat_dim=1024, hidden_dim=256, n_levels=3, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim

        # Project 1024 -> hidden
        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, 1)
        # Project hidden -> 1024
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)

        # Learnable residual scaling (start small so initial output ≈ identity)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_dim
        for i in range(n_levels):
            self.encoders.append(ConvBlock(ch, ch))
            self.downsamples.append(
                nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1)
            )
            ch = ch * 2

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch)
        self.dropout = nn.Dropout(dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_levels):
            self.upsamples.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
            )
            # skip connection doubles channels, then conv reduces
            self.decoders.append(ConvBlock(ch, ch // 2, residual=False))
            ch = ch // 2

    def forward(self, x):
        """
        x: (B, feat_dim, T) WavLM features
        Returns: (B, feat_dim, T) transformed features
        """
        # Project down
        h = self.input_proj(x)  # (B, hidden, T)

        # Encoder path
        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            h = enc(h)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bottleneck(h)
        h = self.dropout(h)

        # Decoder path
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = up(h)
            # Handle size mismatch from odd-length inputs
            if h.shape[-1] != skip.shape[-1]:
                h = h[..., :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        # Project back to feature dim
        delta = self.output_proj(h)  # (B, feat_dim, T)

        # Global residual
        return x + self.alpha * delta

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
