"""
Multi-Layer Quality Mapper

Maps WavLM layers 12-16 pre-surgery features to post-surgery features.
Operates on concatenated multi-layer features (5 * 1024 = 5120-dim per frame).

Architecture:
  - Layer attention: learns a soft weighting over input layers
  - ResUNet1D: transforms the weighted features (like existing unet_vc)
  - Per-layer projection heads: predict each output layer separately
  - Residual connection: output = input + alpha * delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerAttention(nn.Module):
    """Learnable soft attention over WavLM layers."""

    def __init__(self, n_layers=5):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x):
        """
        x: (B, n_layers, 1024, T)
        Returns: (B, 1024, T) — weighted combination
        """
        w = F.softmax(self.weights, dim=0)  # (n_layers,)
        return torch.einsum('blct,l->bct', x, w)


class ConvBlock(nn.Module):
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


class MultiLayerMapper(nn.Module):
    """
    Maps pre-surgery WavLM layers 12-16 to post-surgery layers 12-16.

    Pipeline:
      1. Layer attention: (B, 5, 1024, T) → (B, 1024, T)
      2. U-Net transform in 1024-d feature space
      3. Per-layer projection: (B, 1024, T) → (B, 5, 1024, T)
      4. Residual: output = input + alpha * delta

    Args:
        n_layers: number of WavLM layers (default 5 for L12-16)
        feat_dim: per-layer feature dimension (1024)
        hidden_dim: U-Net internal channels
        n_levels: U-Net depth
        dropout: dropout rate
    """

    def __init__(self, n_layers=5, feat_dim=1024, hidden_dim=128,
                 n_levels=2, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.feat_dim = feat_dim
        self.n_levels = n_levels

        # Layer attention for input
        self.layer_attn = LayerAttention(n_layers)

        # U-Net backbone (operates on 1024-d)
        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, 1)
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_dim
        for _ in range(n_levels):
            self.encoders.append(ConvBlock(ch, ch, dropout=dropout))
            self.downsamples.append(nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2

        self.bottleneck = ConvBlock(ch, ch, dropout=dropout)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for _ in range(n_levels):
            self.upsamples.append(nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1))
            self.decoders.append(ConvBlock(ch, ch // 2, residual=False, dropout=dropout))
            ch //= 2

        # Per-layer output heads
        self.layer_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feat_dim, feat_dim, 1),
                nn.GELU(),
                nn.Conv1d(feat_dim, feat_dim, 1),
            )
            for _ in range(n_layers)
        ])

        # Learnable residual scaling per layer
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        x: (B, n_layers, 1024, T)
        Returns: (B, n_layers, 1024, T)
        """
        B, L, C, T_orig = x.shape
        assert L == self.n_layers and C == self.feat_dim

        # Attend over layers → shared representation
        h = self.layer_attn(x)  # (B, 1024, T)

        # Pad for U-Net
        divisor = 2 ** self.n_levels
        pad_len = (divisor - T_orig % divisor) % divisor
        if pad_len > 0:
            h = F.pad(h, (0, pad_len), mode='reflect')
            x = F.pad(x, (0, pad_len), mode='reflect')

        # U-Net
        h = self.input_proj(h)
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

        shared_delta = self.output_proj(h)  # (B, 1024, T)

        # Per-layer heads + residual
        outputs = []
        for i in range(self.n_layers):
            layer_delta = self.layer_heads[i](shared_delta)  # (B, 1024, T)
            out_i = x[:, i] + self.alphas[i] * layer_delta
            outputs.append(out_i[..., :T_orig])

        return torch.stack(outputs, dim=1)  # (B, n_layers, 1024, T)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
