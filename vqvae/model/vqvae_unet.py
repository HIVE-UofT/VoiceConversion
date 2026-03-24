"""
VQ-UNet for Voice Conversion — WavLM Feature Space (Experiment 6).

Hybrid architecture combining:
  - U-Net skip connections (from UNet-VC) for preserving fine-grained detail
  - VQ bottleneck (from VQVAE Exp5) for content abstraction
  - FiLM-conditioned skip connections for quality-aware detail modulation
  - Explicit content/quality disentanglement with all Exp5 losses

The key insight: Exp5's VQVAE destroyed too much information (SpkSim dropped
from 0.661 baseline to 0.394). Skip connections let fine-grained detail flow
around the bottleneck, but FiLM layers gate this detail based on the quality
vector — preserving disentanglement while improving reconstruction.

Architecture:
  WavLM (1024, T)
    → EncBlock1 (1024→256, T) ─── skip1 → FiLM(quality) → DecBlock1
    → EncBlock2 (256→128, T/2) ── skip2 → FiLM(quality) → DecBlock2
    → VQ bottleneck (128→code_dim→128, T/4)
    → quality vector (64-dim, from separate encoder)
    → DecBlock2 + FiLM(skip2) → (128→256, T/2)
    → DecBlock1 + FiLM(skip1) → (256→1024, T)

For conversion: encode content from pre-surgery, inject post-surgery quality
vector (which modulates skip connections via FiLM), decode, vocode with HiFi-GAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqvae import (
    ProductVectorQuantizer,
    VectorQuantizerHead,
    GradientReversal,
    gradient_reversal,
)


class ResBlock1d(nn.Module):
    """Residual block for 1D feature maps with dropout."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Learns scale (gamma) and shift (beta) from a conditioning vector,
    then applies: output = gamma * input + beta

    This allows the quality vector to modulate skip connection features
    without directly injecting quality info into the content path.
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, channels)
        self.beta_fc = nn.Linear(cond_dim, channels)
        # Initialize near-identity: gamma≈1, beta≈0
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)

    def forward(self, x, cond):
        """
        x: (B, C, T) feature maps
        cond: (B, cond_dim) conditioning vector
        """
        gamma = self.gamma_fc(cond).unsqueeze(2)  # (B, C, 1)
        beta = self.beta_fc(cond).unsqueeze(2)    # (B, C, 1)
        return gamma * x + beta


class UNetEncoder(nn.Module):
    """
    Two-level encoder with strided convolutions.
    Returns intermediate features for skip connections.

    Input:  (B, 1024, T)
    Output: (B, 128, T/4), skips=[(B, 256, T), (B, 128, T/2)]
    """
    def __init__(self, feat_dim=1024, dropout=0.1):
        super().__init__()
        # Level 1: 1024 → 256, keep T
        self.enc1 = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResBlock1d(256, dropout),
        )
        # Downsample T → T/2
        self.down1 = nn.Conv1d(256, 128, kernel_size=4, stride=2, padding=1)

        # Level 2: 128, T/2
        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock1d(128, dropout),
        )
        # Downsample T/2 → T/4
        self.down2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Level 1
        h1 = self.enc1(x)          # (B, 256, T)
        h = self.down1(h1)         # (B, 128, T/2)

        # Level 2
        h2 = self.enc2(h)          # (B, 128, T/2)
        h = self.down2(h2)         # (B, 128, T/4)

        return h, [h1, h2]


class UNetDecoder(nn.Module):
    """
    Two-level decoder with transposed convolutions and FiLM-conditioned skips.

    Input:  (B, code_dim + quality_dim, T/4), skips, quality_vector
    Output: (B, 1024, T)
    """
    def __init__(self, feat_dim=1024, code_dim=64, quality_dim=64, dropout=0.1):
        super().__init__()
        input_dim = code_dim + quality_dim

        # Bottleneck processing
        self.bottleneck = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock1d(128, dropout),
        )

        # Upsample T/4 → T/2
        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        # FiLM on skip2 (128-dim)
        self.film2 = FiLMLayer(128, quality_dim)
        # Combine: 128 (upsampled) + 128 (FiLM skip) = 256 → 256
        self.dec2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResBlock1d(256, dropout),
        )

        # Upsample T/2 → T
        self.up1 = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)
        # FiLM on skip1 (256-dim)
        self.film1 = FiLMLayer(256, quality_dim)
        # Combine: 256 (upsampled) + 256 (FiLM skip) = 512 → 1024
        self.dec1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResBlock1d(512, dropout),
            nn.Conv1d(512, feat_dim, kernel_size=1),
        )

    def forward(self, content, quality, skips):
        """
        content: (B, code_dim, T/4) quantized content
        quality: (B, quality_dim) quality vector
        skips: [skip1 (B, 256, T), skip2 (B, 128, T/2)]
        """
        B, _, T4 = content.shape
        skip1, skip2 = skips

        # Broadcast quality across time and concat with content
        q_expanded = quality.unsqueeze(2).expand(-1, -1, T4)  # (B, quality_dim, T/4)
        h = torch.cat([content, q_expanded], dim=1)           # (B, code_dim + quality_dim, T/4)

        # Bottleneck
        h = self.bottleneck(h)  # (B, 128, T/4)

        # Level 2: upsample + FiLM skip
        h = self.up2(h)                              # (B, 128, T/2)
        h = self._match_time(h, skip2)
        skip2_mod = self.film2(skip2, quality)        # (B, 128, T/2) — quality-modulated
        h = torch.cat([h, skip2_mod], dim=1)          # (B, 256, T/2)
        h = self.dec2(h)                              # (B, 256, T/2)

        # Level 1: upsample + FiLM skip
        h = self.up1(h)                              # (B, 256, T)
        h = self._match_time(h, skip1)
        skip1_mod = self.film1(skip1, quality)        # (B, 256, T) — quality-modulated
        h = torch.cat([h, skip1_mod], dim=1)          # (B, 512, T)
        h = self.dec1(h)                              # (B, 1024, T)

        return h

    def _match_time(self, x, target):
        if x.shape[2] > target.shape[2]:
            x = x[:, :, :target.shape[2]]
        elif x.shape[2] < target.shape[2]:
            x = F.pad(x, (0, target.shape[2] - x.shape[2]))
        return x


class VoiceQualityEncoder1D(nn.Module):
    """
    Encodes voice quality from WavLM features into a fixed-size vector.

    Input:  (B, 1024, T) WavLM features
    Output: (B, quality_dim)
    """
    def __init__(self, feat_dim=1024, quality_dim=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, quality_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.conv(x)           # (B, 64, T/4)
        h = h.mean(dim=2)          # (B, 64) global average pool
        return self.proj(h)         # (B, quality_dim)


class ContentBottleneck(nn.Module):
    """
    Projects encoder output to VQ code dimension.

    Input:  (B, 128, T/4) from UNet encoder
    Output: (B, code_dim, T/4) ready for VQ
    """
    def __init__(self, in_dim=128, code_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, code_dim, kernel_size=1),
            nn.GroupNorm(min(8, code_dim), code_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier1D(nn.Module):
    """
    GRU-based adversarial classifier on content features.

    Input:  (B, code_dim, T') content features
    Output: (B, 1) surgery prediction logit
    """
    def __init__(self, code_dim=64, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(code_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, content):
        h = content.permute(0, 2, 1)          # (B, T', code_dim)
        output, _ = self.gru(h)               # (B, T', hidden*2)
        h_fwd = output[:, -1, :64]
        h_bwd = output[:, 0, 64:]
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)
        return self.fc(h_cat)


class VQUNetWavLM(nn.Module):
    """
    VQ-UNet operating on WavLM features for voice conversion.

    Combines U-Net skip connections with VQ content bottleneck and
    FiLM-conditioned quality modulation.

    Components:
      - unet_encoder: WavLM → encoder features + skip connections
      - content_proj: encoder output → VQ input dimension
      - vq: Product VQ (4 heads x 32 codes each)
      - quality_encoder: WavLM → voice quality vector (64-dim)
      - unet_decoder: quantized content + quality + FiLM(skips) → WavLM features
    """
    def __init__(self, feat_dim=1024, code_dim=64, num_codes=32, num_heads=4,
                 quality_dim=64, commitment_weight=0.25, ema_decay=0.99,
                 entropy_weight=0.5, dropout=0.1, content_noise_std=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.content_noise_std = content_noise_std

        self.unet_encoder = UNetEncoder(feat_dim=feat_dim, dropout=dropout)
        self.content_proj = ContentBottleneck(in_dim=128, code_dim=code_dim, dropout=dropout)
        self.vq = ProductVectorQuantizer(
            num_codes=num_codes, code_dim=code_dim, num_heads=num_heads,
            commitment_weight=commitment_weight, ema_decay=ema_decay,
            entropy_weight=entropy_weight,
        )
        self.quality_encoder = VoiceQualityEncoder1D(
            feat_dim=feat_dim, quality_dim=quality_dim, dropout=dropout)
        self.unet_decoder = UNetDecoder(
            feat_dim=feat_dim, code_dim=code_dim, quality_dim=quality_dim,
            dropout=dropout)

    def forward(self, x):
        """
        x: (B, 1024, T) WavLM features
        Returns: recon (B, 1024, T), vq_loss, perplexity, content_z
        """
        # Encode
        enc_out, skips = self.unet_encoder(x)  # (B, 128, T/4), skips
        content_z = self.content_proj(enc_out)  # (B, code_dim, T/4)

        # Add noise to content during training
        if self.training and self.content_noise_std > 0:
            content_z_noisy = content_z + torch.randn_like(content_z) * self.content_noise_std
        else:
            content_z_noisy = content_z

        # Quantize
        content_q, vq_loss, perplexity = self.vq(content_z_noisy)

        # Quality
        quality = self.quality_encoder(x)  # (B, quality_dim)

        # Decode with FiLM-conditioned skips
        recon = self.unet_decoder(content_q, quality, skips)
        recon = self._match_time(recon, x)

        return recon, vq_loss, perplexity, content_z

    def convert(self, source_features, target_quality):
        """
        Voice conversion: content from source, quality vector directly provided.
        Skip connections come from the source — but FiLM modulates them with target quality.

        source_features: (B, 1024, T) WavLM features from source speaker
        target_quality: (B, quality_dim) pre-computed quality vector
        Returns: (B, 1024, T) converted WavLM features
        """
        with torch.no_grad():
            enc_out, skips = self.unet_encoder(source_features)
            content_z = self.content_proj(enc_out)
            content_q, _, _ = self.vq(content_z)
            converted = self.unet_decoder(content_q, target_quality, skips)
            converted = self._match_time(converted, source_features)
        return converted

    def _match_time(self, recon, target):
        """Crop or pad to match target time dimension."""
        if recon.shape[2] > target.shape[2]:
            recon = recon[:, :, :target.shape[2]]
        elif recon.shape[2] < target.shape[2]:
            pad_t = target.shape[2] - recon.shape[2]
            recon = F.pad(recon, (0, pad_t))
        return recon

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
