"""
VQVAE for Voice Conversion — WavLM Feature Space (Experiment 5).

Instead of operating on raw mel-spectrograms (Exp 1-4), this model operates
on pre-extracted WavLM features (1024-dim, 50fps). The key advantages:
  - WavLM already encodes rich content + speaker info (pretrained on 1000s of hours)
  - VQ only needs to disentangle, not also learn representations from scratch
  - HiFi-GAN vocoder (from knn-vc) handles audio reconstruction
  - Much easier task for the small CUCO dataset (~28 files per domain)

Architecture:
  - ContentEncoder1D:  WavLM features → quantized content (VQ strips quality)
  - VoiceQualityEncoder1D: WavLM features → quality vector (captures resonance/nasality)
  - Decoder1D: quantized content + quality → reconstructed WavLM features
  - DomainClassifier: GRU on content codes → adversarial disentanglement

For conversion: encode content from pre-surgery, inject post-surgery quality, decode,
then vocode with HiFi-GAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse VQ components from existing model
from .vqvae import (
    ProductVectorQuantizer,
    VectorQuantizerHead,
    GradientReversal,
    gradient_reversal,
)


class ResBlock1d(nn.Module):
    """Residual block for 1D feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class ContentEncoder1D(nn.Module):
    """
    Encodes WavLM features into content representation.
    Downsamples time by 4x.

    Input:  (B, 1024, T) WavLM features
    Output: (B, code_dim, T/4)
    """
    def __init__(self, feat_dim=1024, code_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1024, T) → (B, 512, T/2)
            nn.Conv1d(feat_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResBlock1d(512),

            # (B, 512, T/2) → (B, 256, T/4)
            nn.Conv1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock1d(256),

            # (B, 256, T/4) → (B, code_dim, T/4)
            nn.Conv1d(256, code_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class VoiceQualityEncoder1D(nn.Module):
    """
    Encodes voice quality from WavLM features into a fixed-size vector.

    Input:  (B, 1024, T) WavLM features
    Output: (B, quality_dim)
    """
    def __init__(self, feat_dim=1024, quality_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
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


class Decoder1D(nn.Module):
    """
    Reconstructs WavLM features from quantized content + voice quality.
    Upsamples time by 4x.

    Input:  content (B, code_dim, T/4), quality (B, quality_dim)
    Output: (B, 1024, T) reconstructed WavLM features
    """
    def __init__(self, feat_dim=1024, code_dim=64, quality_dim=32,
                 quality_dropout_rate=0.3, content_noise_std=0.1):
        super().__init__()
        self.quality_dropout_rate = quality_dropout_rate
        self.content_noise_std = content_noise_std
        input_dim = code_dim + quality_dim

        self.net = nn.Sequential(
            # (B, input_dim, T/4) → (B, 256, T/4)
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock1d(256),

            # (B, 256, T/4) → (B, 512, T/2)
            nn.ConvTranspose1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResBlock1d(512),

            # (B, 512, T/2) → (B, 1024, T)
            nn.ConvTranspose1d(512, feat_dim, kernel_size=4, stride=2, padding=1),
        )
        # No sigmoid — WavLM features are not bounded to [0, 1]

    def forward(self, content, quality):
        B, _, T = content.shape

        # Content noise
        if self.training and self.content_noise_std > 0:
            content = content + torch.randn_like(content) * self.content_noise_std

        # Quality dropout
        if self.training and self.quality_dropout_rate > 0:
            mask = (torch.rand(B, 1, device=quality.device) > self.quality_dropout_rate).float()
            quality = quality * mask

        # Broadcast quality across time
        q_expanded = quality.unsqueeze(2).expand(-1, -1, T)  # (B, quality_dim, T)
        h = torch.cat([content, q_expanded], dim=1)          # (B, code_dim + quality_dim, T)

        return self.net(h)


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


class VQVAEWavLM(nn.Module):
    """
    VQVAE operating on WavLM features for voice conversion.

    Components:
      - content_encoder: WavLM → continuous content features (T/4 temporal resolution)
      - vq: Product VQ (4 heads × 16 codes each)
      - quality_encoder: WavLM → voice quality vector
      - decoder: quantized content + quality → reconstructed WavLM features (4x upsample)
    """
    def __init__(self, feat_dim=1024, code_dim=64, num_codes=16, num_heads=4,
                 quality_dim=32, commitment_weight=0.25, ema_decay=0.95,
                 entropy_weight=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.content_encoder = ContentEncoder1D(feat_dim=feat_dim, code_dim=code_dim)
        self.vq = ProductVectorQuantizer(
            num_codes=num_codes, code_dim=code_dim, num_heads=num_heads,
            commitment_weight=commitment_weight, ema_decay=ema_decay,
            entropy_weight=entropy_weight,
        )
        self.quality_encoder = VoiceQualityEncoder1D(feat_dim=feat_dim, quality_dim=quality_dim)
        self.decoder = Decoder1D(feat_dim=feat_dim, code_dim=code_dim, quality_dim=quality_dim)

    def forward(self, x):
        """
        x: (B, 1024, T) WavLM features
        Returns: recon (B, 1024, T), vq_loss, perplexity, content_z
        """
        content_z = self.content_encoder(x)
        content_q, vq_loss, perplexity = self.vq(content_z)
        quality = self.quality_encoder(x)
        recon = self.decoder(content_q, quality)
        recon = self._match_time(recon, x)
        return recon, vq_loss, perplexity, content_z

    def convert(self, source_features, target_quality):
        """
        Voice conversion: content from source, quality vector directly provided.

        source_features: (B, 1024, T) WavLM features from source speaker
        target_quality: (B, quality_dim) pre-computed quality vector
        Returns: (B, 1024, T) converted WavLM features
        """
        with torch.no_grad():
            content_z = self.content_encoder(source_features)
            content_q, _, _ = self.vq(content_z)
            converted = self.decoder(content_q, target_quality)
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
