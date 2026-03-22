"""
VQVAE for Voice Conversion (pre/post tonsillectomy).

Architecture:
  - ContentEncoder:  mel → quantized content tokens (VQ bottleneck strips voice quality)
    Downsamples time by 8x (T/8) for tighter bottleneck.
  - VoiceQualityEncoder: mel → fixed-size quality vector (captures resonance/nasality)
  - Decoder: quantized content + quality vector → reconstructed mel

For conversion: encode pre-surgery content, inject post-surgery quality, decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerHead(nn.Module):
    """
    Single VQ head with EMA codebook updates.
    Used as a building block for ProductVectorQuantizer.

    Inputs:  (N, head_dim) flat features for this head
    Outputs: (N, head_dim) quantized features, VQ loss, avg_probs
    """
    def __init__(self, num_codes=16, head_dim=16, commitment_weight=0.25,
                 ema_decay=0.95, dead_code_threshold=2, reset_every=50):
        super().__init__()
        self.num_codes = num_codes
        self.head_dim = head_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.reset_every = reset_every

        self.register_buffer('codebook', torch.randn(num_codes, head_dim))
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_sum', self.codebook.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes))
        self.register_buffer('forward_count', torch.tensor(0, dtype=torch.long))
        self._initialized = False

    def _init_codebook(self, z_flat):
        if self._initialized:
            return
        n = z_flat.shape[0]
        if n >= self.num_codes:
            indices = torch.randperm(n, device=z_flat.device)[:self.num_codes]
            self.codebook.data.copy_(z_flat[indices])
        else:
            repeats = (self.num_codes // n) + 1
            expanded = z_flat.repeat(repeats, 1)[:self.num_codes]
            self.codebook.data.copy_(expanded)
        self.ema_sum.data.copy_(self.codebook.data.clone())
        self.ema_count.data.fill_(1.0)
        self._initialized = True

    def _reset_dead_codes(self, z_flat):
        dead_mask = self.usage_count < self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        n = z_flat.shape[0]
        replace_indices = torch.randint(0, n, (n_dead,), device=z_flat.device)
        noise = torch.randn_like(z_flat[replace_indices]) * 0.02
        self.codebook.data[dead_mask] = z_flat[replace_indices].detach() + noise
        self.ema_sum.data[dead_mask] = self.codebook.data[dead_mask].clone()
        self.ema_count.data[dead_mask] = 1.0
        self.usage_count.zero_()

    def forward(self, z_flat):
        if self.training:
            self._init_codebook(z_flat)

        # Nearest codebook entry
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        z_q_flat = self.codebook[indices]

        # EMA update + dead code tracking
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()
                count = one_hot.sum(dim=0)
                summed = one_hot.t() @ z_flat

                self.ema_count.mul_(self.ema_decay).add_(count, alpha=1 - self.ema_decay)
                self.ema_sum.mul_(self.ema_decay).add_(summed, alpha=1 - self.ema_decay)

                n = self.ema_count.sum()
                count_stable = (
                    (self.ema_count + 1e-5)
                    / (n + self.num_codes * 1e-5)
                    * n
                )
                self.codebook.data.copy_(self.ema_sum / count_stable.unsqueeze(1))

                self.usage_count += count
                self.forward_count += 1
                if self.forward_count % self.reset_every == 0:
                    self._reset_dead_codes(z_flat)

        # Commitment loss + straight-through
        commitment_loss = F.mse_loss(z_q_flat.detach(), z_flat)
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        vq_loss = self.commitment_weight * commitment_loss

        # Per-code usage probabilities (for entropy regularization)
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.num_codes).float()
            avg_probs = one_hot.mean(dim=0)

        return z_q_flat, vq_loss, avg_probs


class ProductVectorQuantizer(nn.Module):
    """
    Product VQ: multiple independent codebook heads, each quantizing a slice of the
    feature dimension. Concatenated outputs give exponentially more combinations
    (num_codes^num_heads) while each head is small enough to avoid collapse.

    Includes codebook entropy regularization to encourage uniform code usage.

    Inputs:  (B, D, T) continuous features, D = num_heads * head_dim
    Outputs: (B, D, T) quantized features, total VQ loss, average perplexity
    """
    def __init__(self, num_codes=16, code_dim=64, num_heads=4,
                 commitment_weight=0.25, ema_decay=0.95,
                 entropy_weight=0.1):
        super().__init__()
        assert code_dim % num_heads == 0, f"code_dim ({code_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = code_dim // num_heads
        self.num_codes = num_codes
        self.entropy_weight = entropy_weight

        self.heads = nn.ModuleList([
            VectorQuantizerHead(
                num_codes=num_codes,
                head_dim=self.head_dim,
                commitment_weight=commitment_weight,
                ema_decay=ema_decay,
            )
            for _ in range(num_heads)
        ])

    def forward(self, z):
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, D)  # (B*T, D)

        # Split across heads
        z_splits = z_flat.chunk(self.num_heads, dim=1)  # list of (B*T, head_dim)

        z_q_parts = []
        total_vq_loss = 0.0
        total_entropy_loss = 0.0
        all_perplexities = []

        for head, z_part in zip(self.heads, z_splits):
            z_q_part, vq_loss, avg_probs = head(z_part)
            z_q_parts.append(z_q_part)
            total_vq_loss = total_vq_loss + vq_loss

            # Entropy regularization: maximize entropy of code usage
            # Higher entropy = more uniform usage = less collapse
            entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
            max_entropy = torch.log(torch.tensor(self.num_codes, dtype=torch.float32, device=z.device))
            # Loss = (max_entropy - actual_entropy), minimizing this maximizes entropy
            total_entropy_loss = total_entropy_loss + (max_entropy - entropy)

            perplexity = torch.exp(entropy)
            all_perplexities.append(perplexity)

        # Concatenate heads back
        z_q_flat = torch.cat(z_q_parts, dim=1)  # (B*T, D)
        z_q = z_q_flat.reshape(B, T, D).permute(0, 2, 1)  # (B, D, T)

        # Total loss = commitment + entropy regularization
        total_loss = total_vq_loss + self.entropy_weight * total_entropy_loss

        # Average perplexity across heads
        avg_perplexity = torch.stack(all_perplexities).mean()

        return z_q, total_loss, avg_perplexity


class ContentEncoder(nn.Module):
    """
    Encodes mel-spectrogram into content representation.
    Downsamples time by 8x (tighter bottleneck than previous 4x).

    Input:  (B, 1, 80, T) mel-spectrogram
    Output: (B, code_dim, T') where T' = T // 8
    """
    def __init__(self, code_dim=64):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # (B, 1, 80, T) → (B, 32, 40, T/2)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (B, 32, 40, T/2) → (B, 64, 20, T/4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (B, 64, 20, T/4) → (B, 128, 10, T/8)  ← extra temporal stride
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Collapse frequency dimension: 128 * 10 → code_dim
        self.proj = nn.Sequential(
            nn.Linear(128 * 10, code_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h = self.conv_stack(x)           # (B, 128, 10, T//8)
        B, C, F, T = h.shape
        h = h.permute(0, 3, 1, 2)       # (B, T, 128, 10)
        h = h.reshape(B, T, C * F)      # (B, T, 1280)
        h = self.proj(h)                 # (B, T, code_dim)
        return h.permute(0, 2, 1)       # (B, code_dim, T)


class VoiceQualityEncoder(nn.Module):
    """
    Encodes voice quality (resonance, nasality) into a fixed-size vector.

    Input:  (B, 1, 80, T) mel-spectrogram
    Output: (B, quality_dim) voice quality embedding
    """
    def __init__(self, quality_dim=32):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Sequential(
            nn.Linear(64, quality_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.conv_stack(x)
        h = h.mean(dim=[2, 3])
        return self.proj(h)


class ResBlock2d(nn.Module):
    """Residual block for 2D feature maps."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Decoder(nn.Module):
    """
    Reconstructs mel-spectrogram from quantized content + voice quality vector.
    Upsamples time by 8x to match the encoder's 8x downsampling.

    Quality dropout: randomly zeros out quality vector during training.
    Content code dropout: adds noise to quantized content to degrade sequential patterns.

    Input:  content (B, code_dim, T'), quality (B, quality_dim)
    Output: (B, 1, 80, T) reconstructed mel-spectrogram, where T = T' * 8
    """
    def __init__(self, code_dim=64, quality_dim=32, quality_dropout_rate=0.3,
                 content_noise_std=0.1):
        super().__init__()
        self.quality_dropout_rate = quality_dropout_rate
        self.content_noise_std = content_noise_std
        input_dim = code_dim + quality_dim

        # 1D processing on content timeline
        self.pre_conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Reshape to 2D: 256 → 128 * 10
        self.reshape_proj = nn.Sequential(
            nn.Linear(256, 128 * 10),
            nn.ReLU(inplace=True),
        )

        # Upsample: 8x temporal, restore frequency
        # Stage 1: (B, 128, 10, T') → (B, 64, 20, T'*2)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResBlock2d(64)

        # Stage 2: (B, 64, 20, T'*2) → (B, 32, 40, T'*4)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResBlock2d(32)

        # Stage 3: (B, 32, 40, T'*4) → (B, 1, 80, T'*8)
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), output_padding=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, content, quality):
        B, _, T = content.shape

        # Content code dropout: add noise to degrade sequential patterns
        if self.training and self.content_noise_std > 0:
            content = content + torch.randn_like(content) * self.content_noise_std

        # Quality dropout
        if self.training and self.quality_dropout_rate > 0:
            mask = (torch.rand(B, 1, device=quality.device) > self.quality_dropout_rate).float()
            quality = quality * mask

        # Broadcast quality across time
        q_expanded = quality.unsqueeze(2).expand(-1, -1, T)
        h = torch.cat([content, q_expanded], dim=1)

        # 1D temporal processing
        h = self.pre_conv(h)                          # (B, 256, T)

        # Reshape to 2D
        h = h.permute(0, 2, 1)                        # (B, T, 256)
        h = self.reshape_proj(h)                       # (B, T, 1280)
        h = h.reshape(B, T, 128, 10)                  # (B, T, 128, 10)
        h = h.permute(0, 2, 3, 1)                     # (B, 128, 10, T)

        # Upsample with residual refinement (3 stages = 8x)
        h = self.deconv1(h)                            # (B, 64, 20, T*2)
        h = self.res1(h)
        h = self.deconv2(h)                            # (B, 32, 40, T*4)
        h = self.res2(h)
        mel = self.deconv3(h)                          # (B, 1, ~80, T*8)

        return mel


class DomainClassifier(nn.Module):
    """
    GRU-based adversarial classifier on content features.
    Uses a recurrent layer to detect sequence-level surgery information,
    not just per-frame (closes the steganography loophole).

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
        # (B, code_dim, T') → (B, T', code_dim)
        h = content.permute(0, 2, 1)
        output, _ = self.gru(h)           # (B, T', hidden*2)
        # Use last hidden state from both directions
        h_fwd = output[:, -1, :64]        # last timestep, forward
        h_bwd = output[:, 0, 64:]         # first timestep, backward
        h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)
        return self.fc(h_cat)             # (B, 1)


class GradientReversal(torch.autograd.Function):
    """Reverses gradients during backward pass (for adversarial disentanglement)."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class VQVAE(nn.Module):
    """
    Full VQVAE for voice conversion.

    Components:
      - content_encoder: mel → continuous content features (T/8 temporal resolution)
      - vq: Product VQ (4 heads × 16 codes each = 65K effective combinations)
      - quality_encoder: mel → voice quality vector
      - decoder: quantized content + quality → reconstructed mel (8x upsample)
    """
    def __init__(self, code_dim=64, num_codes=16, num_heads=4, quality_dim=32,
                 commitment_weight=0.25, ema_decay=0.95, entropy_weight=0.1):
        super().__init__()
        self.content_encoder = ContentEncoder(code_dim=code_dim)
        self.vq = ProductVectorQuantizer(
            num_codes=num_codes, code_dim=code_dim, num_heads=num_heads,
            commitment_weight=commitment_weight, ema_decay=ema_decay,
            entropy_weight=entropy_weight,
        )
        self.quality_encoder = VoiceQualityEncoder(quality_dim=quality_dim)
        self.decoder = Decoder(code_dim=code_dim, quality_dim=quality_dim)

    def forward(self, x):
        content_z = self.content_encoder(x)
        content_q, vq_loss, perplexity = self.vq(content_z)
        quality = self.quality_encoder(x)
        recon = self.decoder(content_q, quality)
        recon = self._match_size(recon, x)
        return recon, vq_loss, perplexity, content_z

    def convert(self, source_mel, target_mel):
        """Voice conversion: content from source, quality from target."""
        with torch.no_grad():
            content_z = self.content_encoder(source_mel)
            content_q, _, _ = self.vq(content_z)
            quality = self.quality_encoder(target_mel)
            converted = self.decoder(content_q, quality)
            converted = self._match_size(converted, source_mel)
        return converted

    def _match_size(self, recon, target):
        """Crop or pad reconstruction to match target spatial dimensions."""
        if recon.shape[2] > target.shape[2]:
            recon = recon[:, :, :target.shape[2], :]
        elif recon.shape[2] < target.shape[2]:
            pad_f = target.shape[2] - recon.shape[2]
            recon = F.pad(recon, (0, 0, 0, pad_f))

        if recon.shape[3] > target.shape[3]:
            recon = recon[:, :, :, :target.shape[3]]
        elif recon.shape[3] < target.shape[3]:
            pad_t = target.shape[3] - recon.shape[3]
            recon = F.pad(recon, (0, pad_t))

        return recon
