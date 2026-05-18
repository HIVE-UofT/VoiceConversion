"""
DLA-VC: Dual Layer Adapter Voice Conversion for Surgical Speech.

Combines ideas from AdaptVC (ICASSP 2025) with paired-training disentanglement:
  - Learned layer-weighted adapters on frozen WavLM-Large (24 layers)
    -> Content adapter learns which layers encode linguistic content
    -> Quality adapter learns which layers encode voice quality (resonance, nasality)
  - Product VQ bottleneck on content (forces discrete abstraction)
  - U-Net decoder with FiLM-conditioned skip connections (quality modulation)
  - Paired training: directly supervise pre->post transformation
  - Reconstruction target: WavLM layer 6 features (compatible with knn-vc HiFi-GAN)

Key novelty vs AdaptVC:
  - Dual adapters for content vs QUALITY (not speaker) — surgery-aware
  - Paired training with cross-reconstruction + cycle + adversarial losses
  - VQ + FiLM U-Net decoder in WavLM feature space (not mel)
  - Works with ~28 files per domain (vs AdaptVC's 245 hours)

Key novelty vs our VQVAE Exp5/6:
  - Learned layer selection (different WavLM layers for content vs quality)
  - Better feature routing reduces information loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────
# Layer Adapter (from AdaptVC)
# ──────────────────────────────────────────────────────

class LayerAdapter(nn.Module):
    """
    Learned softmax-weighted sum over WavLM hidden layers.

    Inspired by AdaptVC: a single Linear(num_layers, 1) with softmax normalization.
    Different adapters learn to weight different layers — content adapter
    emphasizes linguistic layers (mid-to-late), quality adapter emphasizes
    acoustic layers (early).

    Input:  (B, num_layers, C, T) stacked hidden states
    Output: (B, C, T) weighted sum
    """
    def __init__(self, num_layers=24):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, hidden_states):
        """
        hidden_states: (B, num_layers, C, T)
        Returns: (B, C, T) weighted sum
        """
        w = F.softmax(self.weight, dim=0)  # (num_layers,)
        # Weighted sum: (B, num_layers, C, T) x (num_layers,) → (B, C, T)
        return torch.einsum('blct,l->bct', hidden_states, w)


# ──────────────────────────────────────────────────────
# VQ Components (adapted from vqvae)
# ──────────────────────────────────────────────────────

class VectorQuantizerHead(nn.Module):
    """Single VQ head with EMA codebook updates."""
    def __init__(self, num_codes=32, head_dim=16, commitment_weight=0.25,
                 ema_decay=0.99):
        super().__init__()
        self.num_codes = num_codes
        self.head_dim = head_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay

        self.register_buffer('codebook', torch.randn(num_codes, head_dim))
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_sum', self.codebook.clone())
        self.register_buffer('_initialized', torch.tensor(False))

    def _init_codebook(self, z_flat):
        if self._initialized.item():
            return
        n = z_flat.shape[0]
        if n >= self.num_codes:
            indices = torch.randperm(n, device=z_flat.device)[:self.num_codes]
            self.codebook.data.copy_(z_flat[indices])
        else:
            repeats = (self.num_codes // n) + 1
            self.codebook.data.copy_(z_flat.repeat(repeats, 1)[:self.num_codes])
        self.ema_sum.data.copy_(self.codebook.data.clone())
        self.ema_count.data.fill_(1.0)
        self._initialized.fill_(True)

    def forward(self, z_flat):
        self._init_codebook(z_flat)
        dists = torch.cdist(z_flat, self.codebook)
        indices = dists.argmin(dim=1)
        quantized = self.codebook[indices]

        # Straight-through estimator
        z_q = z_flat + (quantized - z_flat).detach()

        # Losses
        commit_loss = F.mse_loss(z_flat, quantized.detach())
        loss = self.commitment_weight * commit_loss

        # EMA update
        if self.training:
            one_hot = F.one_hot(indices, self.num_codes).float()
            count = one_hot.sum(0)
            self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
            sum_vecs = one_hot.t() @ z_flat
            self.ema_sum.mul_(self.ema_decay).add_(sum_vecs, alpha=1 - self.ema_decay)
            n = self.ema_count.clamp(min=1e-5)
            self.codebook.data.copy_(self.ema_sum / n.unsqueeze(1))

        # Perplexity
        avg_probs = one_hot.mean(0) if self.training else F.one_hot(indices, self.num_codes).float().mean(0)
        return z_q, loss, avg_probs


class ProductVectorQuantizer(nn.Module):
    """Product VQ: multiple independent heads for exponential combinations."""
    def __init__(self, num_codes=32, code_dim=64, num_heads=4,
                 commitment_weight=0.25, ema_decay=0.99, entropy_weight=0.5):
        super().__init__()
        assert code_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = code_dim // num_heads
        self.entropy_weight = entropy_weight

        self.heads = nn.ModuleList([
            VectorQuantizerHead(num_codes, self.head_dim, commitment_weight, ema_decay)
            for _ in range(num_heads)
        ])

    def forward(self, z):
        B, C, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, C)  # (B*T, C)

        chunks = z_flat.chunk(self.num_heads, dim=1)
        quantized_parts, total_loss, all_probs = [], 0.0, []

        for head, chunk in zip(self.heads, chunks):
            z_q, loss, probs = head(chunk)
            quantized_parts.append(z_q)
            total_loss += loss
            all_probs.append(probs)

        quantized = torch.cat(quantized_parts, dim=1)
        quantized = quantized.reshape(B, T, C).permute(0, 2, 1)
        total_loss /= self.num_heads

        # Entropy regularization
        avg_perplexity = 0
        entropy_loss = 0
        for probs in all_probs:
            avg_probs = probs.clamp(min=1e-10)
            entropy = -(avg_probs * avg_probs.log()).sum()
            max_entropy = torch.log(torch.tensor(float(self.heads[0].num_codes)))
            entropy_loss += (max_entropy - entropy)
            perplexity = torch.exp(entropy)
            avg_perplexity += perplexity

        entropy_loss /= self.num_heads
        avg_perplexity /= self.num_heads
        total_loss += self.entropy_weight * entropy_loss

        return quantized, total_loss, avg_perplexity


# ──────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: quality vector controls skip connections."""
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, channels)
        self.beta_fc = nn.Linear(cond_dim, channels)
        nn.init.ones_(self.gamma_fc.bias)
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(2)
        beta = self.beta_fc(cond).unsqueeze(2)
        return gamma * x + beta


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ──────────────────────────────────────────────────────
# Encoders
# ──────────────────────────────────────────────────────

class ContentEncoder1D(nn.Module):
    """
    Adapter-weighted WavLM features → content for VQ.
    Downsamples time by 4x.

    Input:  (B, 1024, T)
    Output: (B, code_dim, T/4)
    """
    def __init__(self, feat_dim=1024, code_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            ResBlock1d(256, dropout),
            nn.Conv1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock1d(128, dropout),
            nn.Conv1d(128, code_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class VoiceQualityEncoder1D(nn.Module):
    """
    Adapter-weighted WavLM features → quality vector.

    Input:  (B, 1024, T)
    Output: (B, quality_dim)
    """
    def __init__(self, feat_dim=1024, quality_dim=64, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(128, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.proj = nn.Sequential(nn.Linear(64, quality_dim), nn.Tanh())

    def forward(self, x):
        h = self.conv(x).mean(dim=2)
        return self.proj(h)


# ──────────────────────────────────────────────────────
# U-Net Decoder with FiLM
# ──────────────────────────────────────────────────────

class UNetEncoder(nn.Module):
    """Two-level encoder producing skip connections."""
    def __init__(self, feat_dim=1024, dropout=0.1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(feat_dim, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(), nn.Dropout(dropout),
            ResBlock1d(256, dropout),
        )
        self.down1 = nn.Conv1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, 128), nn.GELU(), ResBlock1d(128, dropout),
        )
        self.down2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        h1 = self.enc1(x)
        h = self.down1(h1)
        h2 = self.enc2(h)
        h = self.down2(h2)
        return h, [h1, h2]


class UNetDecoder(nn.Module):
    """Two-level decoder with FiLM-conditioned skip connections."""
    def __init__(self, feat_dim=1024, code_dim=64, quality_dim=64, dropout=0.1):
        super().__init__()
        input_dim = code_dim + quality_dim

        self.bottleneck = nn.Sequential(
            nn.Conv1d(input_dim, 128, 3, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(), ResBlock1d(128, dropout),
        )
        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.film2 = FiLMLayer(128, quality_dim)
        self.dec2 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256), nn.GELU(), nn.Dropout(dropout),
            ResBlock1d(256, dropout),
        )
        self.up1 = nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1)
        self.film1 = FiLMLayer(256, quality_dim)
        self.dec1 = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=1),
            nn.GroupNorm(8, 512), nn.GELU(),
            ResBlock1d(512, dropout),
            nn.Conv1d(512, feat_dim, kernel_size=1),
        )

    def forward(self, content, quality, skips):
        B, _, T4 = content.shape
        skip1, skip2 = skips

        q_expanded = quality.unsqueeze(2).expand(-1, -1, T4)
        h = torch.cat([content, q_expanded], dim=1)
        h = self.bottleneck(h)

        h = self._match(self.up2(h), skip2)
        h = torch.cat([h, self.film2(F.instance_norm(skip2), quality)], dim=1)
        h = self.dec2(h)

        h = self._match(self.up1(h), skip1)
        h = torch.cat([h, self.film1(F.instance_norm(skip1), quality)], dim=1)
        h = self.dec1(h)
        return h

    def _match(self, x, ref):
        if x.shape[2] > ref.shape[2]:
            return x[:, :, :ref.shape[2]]
        elif x.shape[2] < ref.shape[2]:
            return F.pad(x, (0, ref.shape[2] - x.shape[2]))
        return x


# ──────────────────────────────────────────────────────
# Domain Classifier
# ──────────────────────────────────────────────────────

class DomainClassifier1D(nn.Module):
    """GRU-based adversarial classifier on content codes."""
    def __init__(self, code_dim=64, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(code_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(inplace=True), nn.Linear(32, 1),
        )

    def forward(self, content):
        h = content.permute(0, 2, 1)
        output, _ = self.gru(h)
        h_cat = torch.cat([output[:, -1, :64], output[:, 0, 64:]], dim=1)
        return self.fc(h_cat)


# ──────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────

class DLAVCModel(nn.Module):
    """
    DLA-VC: Dual Layer Adapter model for surgical voice conversion.

    WavLM-Large (frozen) → 24 hidden states
      ├─ Content Adapter (learned layer weights) → Content Encoder → Product VQ
      └─ Quality Adapter (learned layer weights) → Quality Encoder → quality_vec
                                                                        │
    VQ content + quality_vec → U-Net Decoder (FiLM skips) → WavLM layer 6 features

    The adapters learn WHICH WavLM layers are most useful for content vs quality.
    The decoder reconstructs layer 6 features (compatible with knn-vc HiFi-GAN).
    """
    def __init__(self, feat_dim=1024, code_dim=64, num_codes=32, num_heads=4,
                 quality_dim=64, num_wavlm_layers=24,
                 commitment_weight=0.25, ema_decay=0.99, entropy_weight=0.5,
                 dropout=0.1, content_noise_std=0.1,
                 use_residual_output=False, wavlm_layer_idx=5,
                 alpha_init=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.content_noise_std = content_noise_std
        self.num_wavlm_layers = num_wavlm_layers
        # When True, the decoder predicts a *delta* and the model output is
        # anchor + alpha * delta (UNet-VC's residual trick). The anchor is the
        # input WavLM layer-6 features (i.e. pre features at conversion time).
        # alpha is a learnable scalar that scales the predicted shift.
        self.use_residual_output = use_residual_output
        self.wavlm_layer_idx = wavlm_layer_idx  # 0-indexed; layer 6 → index 5
        if use_residual_output:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # Adapters (the key novelty from AdaptVC)
        self.content_adapter = LayerAdapter(num_wavlm_layers)
        self.quality_adapter = LayerAdapter(num_wavlm_layers)

        # U-Net encoder (on content adapter output) → skip connections
        self.unet_encoder = UNetEncoder(feat_dim=feat_dim, dropout=dropout)

        # Content bottleneck → VQ
        self.content_proj = nn.Sequential(
            nn.Conv1d(128, code_dim, kernel_size=1),
            nn.GroupNorm(min(8, code_dim), code_dim),
            nn.GELU(),
        )
        self.vq = ProductVectorQuantizer(
            num_codes=num_codes, code_dim=code_dim, num_heads=num_heads,
            commitment_weight=commitment_weight, ema_decay=ema_decay,
            entropy_weight=entropy_weight,
        )

        # Quality encoder (on quality adapter output)
        self.quality_encoder = VoiceQualityEncoder1D(
            feat_dim=feat_dim, quality_dim=quality_dim, dropout=dropout)

        # Pre → post quality mapper (learns the "surgery direction" in quality space)
        # At inference, given a test patient's pre audio, predict their post quality
        # vector per-patient instead of using a population-averaged vector.
        self.q_shift = nn.Sequential(
            nn.Linear(quality_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, quality_dim),
        )

        # Decoder
        self.unet_decoder = UNetDecoder(
            feat_dim=feat_dim, code_dim=code_dim, quality_dim=quality_dim,
            dropout=dropout)

    def forward(self, hidden_states, target_features):
        """
        hidden_states: (B, num_layers, 1024, T) — all WavLM hidden states
        target_features: (B, 1024, T) — WavLM layer 6 features (reconstruction target)

        Returns: recon, vq_loss, perplexity, content_z, quality
        """
        # Apply adapters
        content_feats = self.content_adapter(hidden_states)  # (B, 1024, T)
        quality_feats = self.quality_adapter(hidden_states)  # (B, 1024, T)

        # Content path: UNet encoder → project → VQ
        enc_out, skips = self.unet_encoder(content_feats)
        content_z = self.content_proj(enc_out)

        if self.training and self.content_noise_std > 0:
            content_z = content_z + torch.randn_like(content_z) * self.content_noise_std

        content_q, vq_loss, perplexity = self.vq(content_z)

        # Quality path
        quality = self.quality_encoder(quality_feats)

        # Decode → either layer-6 features directly, or a delta added to the
        # input layer-6 anchor (residual-output mode, UNet-VC style).
        delta = self.unet_decoder(content_q, quality, skips)
        delta = self._match_time(delta, target_features)
        if self.use_residual_output:
            # Anchor on the input layer-6 features (target_features at training
            # time IS the input layer 6 because forward is called with paired
            # hidden_all/target_all from the same audio).
            recon = target_features + self.alpha * delta
        else:
            recon = delta

        return recon, vq_loss, perplexity, content_z, quality

    def encode_content(self, hidden_states):
        """Extract quantized content from hidden states."""
        content_feats = self.content_adapter(hidden_states)
        enc_out, skips = self.unet_encoder(content_feats)
        content_z = self.content_proj(enc_out)
        content_q, _, _ = self.vq(content_z)
        return content_q, skips

    def encode_quality(self, hidden_states):
        """Extract quality vector from hidden states."""
        quality_feats = self.quality_adapter(hidden_states)
        return self.quality_encoder(quality_feats)

    def convert(self, source_hidden_states, target_quality):
        """
        Voice conversion: content from source, quality from target.
        Skip connections from source, FiLM-modulated by target quality.

        With residual-output mode, the decoder predicts a delta that is added
        to the source's layer-6 features (anchor = pre features), so the
        model only has to learn the pre→post shift instead of full post-feature
        regeneration. Same trick UNet-VC uses.
        """
        with torch.no_grad():
            content_q, skips = self.encode_content(source_hidden_states)
            delta = self.unet_decoder(content_q, target_quality, skips)
            if self.use_residual_output:
                anchor = source_hidden_states[:, self.wavlm_layer_idx]  # (B, 1024, T)
                delta = self._match_time(delta, anchor)
                converted = anchor + self.alpha * delta
            else:
                converted = delta
        return converted

    def predict_post_quality(self, pre_hidden_states):
        """Per-patient post quality prediction: encode_quality + q_shift.
        Use this at inference instead of a population-averaged post quality."""
        with torch.no_grad():
            q_pre = self.encode_quality(pre_hidden_states)
            return self.q_shift(q_pre)

    def get_adapter_weights(self):
        """Return learned adapter weights for visualization."""
        return {
            'content': F.softmax(self.content_adapter.weight, dim=0).detach().cpu().numpy(),
            'quality': F.softmax(self.quality_adapter.weight, dim=0).detach().cpu().numpy(),
        }

    def _match_time(self, recon, target):
        if recon.shape[2] > target.shape[2]:
            return recon[:, :, :target.shape[2]]
        elif recon.shape[2] < target.shape[2]:
            return F.pad(recon, (0, target.shape[2] - recon.shape[2]))
        return recon

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
