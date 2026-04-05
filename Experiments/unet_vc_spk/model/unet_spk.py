"""
Speaker-Conditioned Residual 1D U-Net for WavLM Feature-Space Voice Conversion.

Each level of the U-Net is conditioned on a speaker embedding via FiLM
(Feature-wise Linear Modulation): a linear projection of the speaker embedding
produces per-channel scale (gamma) and shift (beta) that adapt the U-Net's
internal representations to the current speaker.

Speaker embedding: mean-pooled WavLM features of the pre-surgery utterance
(1024-dim). No separate speaker encoder model needed.

The hypothesis: speakers with similar WavLM profiles respond to surgery
similarly, so the model can learn a mapping
    f(x_pre_frame, spk_emb) -> x_post_frame
that generalises to unseen speakers at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale + shift feature maps per-channel."""

    def __init__(self, spk_dim, n_channels):
        super().__init__()
        self.proj = nn.Linear(spk_dim, 2 * n_channels)

    def forward(self, x, spk):
        # spk: (B, spk_dim)
        # x:   (B, C, T)
        gamma_beta = self.proj(spk)               # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1) # each (B, C)
        gamma = gamma.unsqueeze(-1)               # (B, C, 1)
        beta  = beta.unsqueeze(-1)                # (B, C, 1)
        return (1.0 + gamma) * x + beta


class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + GELU + optional FiLM conditioning."""

    def __init__(self, in_ch, out_ch, spk_dim, kernel_size=3,
                 residual=True, dropout=0.0):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film1 = FiLM(spk_dim, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film2 = FiLM(spk_dim, out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spk):
        h = self.film1(self.act(self.norm1(self.conv1(x))), spk)
        h = self.dropout(h)
        h = self.film2(self.act(self.norm2(self.conv2(h))), spk)
        if self.residual:
            h = h + x
        return h


class SpeakerConditionedUNet(nn.Module):
    """
    Speaker-conditioned residual 1D U-Net.

    Args:
        feat_dim:   WavLM feature dimension (1024)
        spk_dim:    speaker embedding dimension (same as feat_dim by default,
                    since we use mean-pooled WavLM features as the speaker emb)
        hidden_dim: internal channel width
        n_levels:   encoder/decoder depth
        dropout:    dropout rate
    """

    def __init__(self, feat_dim=1024, spk_dim=1024, hidden_dim=128,
                 n_levels=2, dropout=0.25):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_levels = n_levels

        # Project speaker embedding to a smaller space to avoid over-parameterisation
        self.spk_proj = nn.Sequential(
            nn.Linear(spk_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        spk_h = 256

        self.input_proj  = nn.Conv1d(feat_dim, hidden_dim, 1)
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)

        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Encoder
        self.encoders    = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = hidden_dim
        for _ in range(n_levels):
            self.encoders.append(ConvBlock(ch, ch, spk_h, dropout=dropout))
            self.downsamples.append(
                nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch, spk_h, dropout=dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        for _ in range(n_levels):
            self.upsamples.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1))
            self.decoders.append(
                ConvBlock(ch, ch // 2, spk_h, residual=False, dropout=dropout))
            ch //= 2

    def forward(self, x, spk_emb):
        """
        Args:
            x:       (B, feat_dim, T)  — per-frame WavLM features
            spk_emb: (B, spk_dim)      — mean-pooled WavLM features of the utterance
        Returns:
            (B, feat_dim, T)
        """
        T_orig = x.shape[-1]

        divisor = 2 ** self.n_levels
        pad_len = (divisor - T_orig % divisor) % divisor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), mode='reflect')

        spk = self.spk_proj(spk_emb)   # (B, 256)

        h = self.input_proj(x)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            h = enc(h, spk)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h, spk)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h, spk)

        delta = self.output_proj(h)
        out = x + self.alpha * delta
        return out[..., :T_orig]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
