"""
MaskCycleGAN-VC: Mask CycleGAN for Non-Parallel Voice Conversion

Based on: Kaneko et al., "MaskCycleGAN-VC: Learning Non-parallel Voice
Conversion with Filling in Frames" (2021)

Architecture: 2-1-2D CNN Generator + PatchGAN Discriminator
- Domain A = pre-surgery voice
- Domain B = post-surgery voice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Generator: 2D downsample → 1D residual → 2D upsample
# ──────────────────────────────────────────────

class GLU(nn.Module):
    """Gated Linear Unit activation."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class DownsampleBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_ch * 2)
        self.glu = GLU(dim=1)

    def forward(self, x):
        return self.glu(self.norm(self.conv(x)))


class UpsampleBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch * 2, kernel_size, stride, padding, output_padding)
        self.norm = nn.InstanceNorm2d(out_ch * 2)
        self.glu = GLU(dim=1)

    def forward(self, x):
        return self.glu(self.norm(self.conv(x)))


class ResidualBlock1D(nn.Module):
    """1D residual block with GLU for the bottleneck."""
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size, padding=padding),
            nn.InstanceNorm1d(channels * 2),
            GLU(dim=1),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    2-1-2D CNN Generator for MaskCycleGAN-VC.

    Input: mel-spectrogram (B, 1, 80, T) where T = num_frames
    Output: converted mel-spectrogram (B, 1, 80, T)
    """
    def __init__(self, n_mels=80, n_res_blocks=3, base_channels=128):
        super().__init__()
        c = base_channels  # 128 (small dataset) vs 256 (large dataset)

        # --- 2D Downsampling ---
        # (B, 2, 80, T) → (B, c, 20, T/4)
        self.down = nn.Sequential(
            nn.Conv2d(2, c, (5, 15), (1, 1), (2, 7)),  # 2 channels: mel + mask
            GLU(dim=1),
            DownsampleBlock2D(c // 2, c // 2, (4, 8), (2, 2), (1, 3)),   # (40, T/2)
            DownsampleBlock2D(c // 2, c, (4, 8), (2, 2), (1, 3)),        # (20, T/4)
        )

        # --- Reshape 2D→1D ---
        bottleneck = 128
        self.to_1d = nn.Sequential(
            nn.Conv1d(c * 20, bottleneck, 1),
            nn.InstanceNorm1d(bottleneck),
        )

        # --- 1D Residual Blocks ---
        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(bottleneck) for _ in range(n_res_blocks)]
        )

        # --- Reshape 1D→2D ---
        self.to_2d = nn.Sequential(
            nn.Conv1d(bottleneck, c * 20, 1),
            nn.InstanceNorm1d(c * 20),
        )

        # --- 2D Upsampling ---
        self.up = nn.Sequential(
            UpsampleBlock2D(c, c // 2, (4, 8), (2, 2), (1, 3), (0, 0)),   # (40, T/2)
            UpsampleBlock2D(c // 2, c // 4, (4, 8), (2, 2), (1, 3), (0, 0)),  # (80, T)
        )
        self.final_conv = nn.Conv2d(c // 4, 1, (5, 15), (1, 1), (2, 7))

    def forward(self, mel, mask=None):
        """
        Args:
            mel: (B, 1, 80, T) input mel-spectrogram (possibly masked)
            mask: (B, 1, 80, T) binary mask (1=keep, 0=masked). If None, all ones.
        """
        if mask is None:
            mask = torch.ones_like(mel)

        x = torch.cat([mel, mask], dim=1)  # (B, 2, 80, T)
        T = x.shape[3]

        # Pad T to be divisible by 4
        pad_t = (4 - T % 4) % 4
        if pad_t > 0:
            x = F.pad(x, (0, pad_t))

        # 2D downsample
        h = self.down(x)  # (B, C, 20, T/4)
        B, C, F_dim, T_down = h.shape

        # Reshape to 1D
        h = h.reshape(B, C * F_dim, T_down)
        h = self.to_1d(h)

        # 1D residual processing
        h = self.res_blocks(h)

        # Reshape to 2D
        h = self.to_2d(h)
        h = h.reshape(B, C, F_dim, T_down)

        # 2D upsample
        h = self.up(h)
        out = self.final_conv(h)

        # Remove padding
        if pad_t > 0:
            out = out[:, :, :, :T]

        return out


# ──────────────────────────────────────────────
# Discriminator: 2D PatchGAN
# ──────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Input: (B, 1, 80, T)
    Output: (B, 1, H', W') patch-level realness scores
    """
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            # (B, 1, 80, T)
            nn.Conv2d(1, 128, (3, 3), (1, 2), (1, 1)),
            GLU(dim=1),
            # (B, 64, 80, T/2)
            nn.Conv2d(64, 256, (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(256),
            GLU(dim=1),
            # (B, 128, 40, T/4)
            nn.Conv2d(128, 512, (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(512),
            GLU(dim=1),
            # (B, 256, 20, T/8)
            nn.Conv2d(256, 512, (3, 3), (2, 2), (1, 1)),
            nn.InstanceNorm2d(512),
            GLU(dim=1),
            # (B, 256, 10, T/16)
            nn.Conv2d(256, 1, (1, 3), (1, 1), (0, 1)),
            # (B, 1, 10, T/16)
        )

    def forward(self, x):
        return self.layers(x)


# ──────────────────────────────────────────────
# Mask generation for FIF (Filling in Frames)
# ──────────────────────────────────────────────

def generate_mask(shape, mask_ratio=0.5, device='cpu'):
    """
    Generate a temporal frame mask for the FIF task.
    Masks contiguous frames in the time dimension.

    Args:
        shape: (B, 1, F, T) shape of mel-spectrogram
        mask_ratio: fraction of time frames to mask
        device: torch device

    Returns:
        mask: (B, 1, F, T) binary mask, 1=keep 0=masked
    """
    B, _, F, T = shape
    mask = torch.ones(B, 1, F, T, device=device)
    n_masked = int(T * mask_ratio)

    for b in range(B):
        start = torch.randint(0, T - n_masked + 1, (1,)).item()
        mask[b, :, :, start:start + n_masked] = 0.0

    return mask
