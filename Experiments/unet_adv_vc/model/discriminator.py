"""
PatchGAN-style 1D discriminator for WavLM feature sequences.

Operates on (B, 1024, T) feature tensors and outputs per-patch
real/fake scores. Uses spectral normalization for stable GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator1D(nn.Module):
    """
    1D PatchGAN discriminator on WavLM features.

    Classifies overlapping patches of the feature sequence as real (post-surgery)
    or fake (converted). Uses spectral normalization for Lipschitz constraint.

    Output: (B, 1, T') where T' < T due to strided convolutions.
    Each output element covers a receptive field of ~30 input frames.
    """

    def __init__(self, feat_dim=1024, hidden_dim=256, n_layers=3):
        super().__init__()

        layers = []
        in_ch = feat_dim

        for i in range(n_layers):
            out_ch = min(hidden_dim * (2 ** i), 512)
            layers.append(spectral_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            ))
            layers.append(nn.LeakyReLU(0.2))
            in_ch = out_ch

        # Final 1-channel output (patch scores)
        layers.append(spectral_norm(
            nn.Conv1d(in_ch, 1, kernel_size=3, stride=1, padding=1)
        ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, 1024, T) — WavLM feature sequence
        Returns: (B, 1, T') — per-patch real/fake logits
        """
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
