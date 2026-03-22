"""
Training script for MaskCycleGAN-VC.

Domain A = pre-surgery (label=0)
Domain B = post-surgery (label=1)

Losses:
  - Adversarial (LSGAN): D tries to distinguish real vs fake
  - Cycle-consistency: A→B→A ≈ A, B→A→B ≈ B
  - Identity: G_A2B(B) ≈ B, G_B2A(A) ≈ A  (low weight for subtle domains)
  - Multi-resolution STFT: preserves spectral/harmonic detail
  - FIF (Filling in Frames): masking forces generator to learn temporal structure
    (training signal comes from cycle + adversarial on masked input, not a direct L1)

Usage:
  python scripts/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

from model.mask_cyclegan import Generator, Discriminator, generate_mask


# ──────────────────────────────────────────────
# Dataset: separate pre/post into two domains
# ──────────────────────────────────────────────

class DomainDataset(Dataset):
    """Loads mel-spectrograms for a single domain (pre OR post surgery).

    Includes data augmentation for small datasets:
    - Random time crop (instead of always taking first target_len frames)
    - Random frequency masking (SpecAugment-style)
    - Random amplitude scaling
    """
    def __init__(self, pkl_path, label, target_len=400, augment=False):
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        self.data = [d for d in all_data if d['label'] == label]
        self.target_len = target_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel_spectrogram'].copy()  # (80, T)

        # Random time crop vs fixed crop
        if mel.shape[1] > self.target_len:
            if self.augment:
                start = np.random.randint(0, mel.shape[1] - self.target_len + 1)
                mel = mel[:, start:start + self.target_len]
            else:
                mel = mel[:, :self.target_len]
        elif mel.shape[1] < self.target_len:
            pad = self.target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')

        if self.augment:
            # Random frequency masking (1-5 mel bins)
            if np.random.rand() > 0.5:
                n_mask = np.random.randint(1, 6)
                f_start = np.random.randint(0, mel.shape[0] - n_mask)
                mel[f_start:f_start + n_mask, :] = 0.0

            # Random amplitude scaling (±10%)
            if np.random.rand() > 0.5:
                scale = 0.9 + np.random.rand() * 0.2
                mel = np.clip(mel * scale, 0, 1)

        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, 80, T)


class PairedDomainLoader:
    """Yields random pairs (A, B) from two domain datasets, cycling the shorter one."""
    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        for _ in range(max(len(self.loader_a), len(self.loader_b))):
            try:
                a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a = next(iter_a)
            try:
                b = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b = next(iter_b)
            # Match batch sizes (in case last batch differs)
            min_b = min(a.shape[0], b.shape[0])
            yield a[:min_b], b[:min_b]

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def adversarial_loss_d(d_real, d_fake):
    """LSGAN discriminator loss."""
    return torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2)


def adversarial_loss_g(d_fake):
    """LSGAN generator loss."""
    return torch.mean((d_fake - 1) ** 2)


def cycle_consistency_loss(real, reconstructed):
    """L1 cycle loss."""
    return F.l1_loss(reconstructed, real)


def identity_loss(real, same):
    """L1 identity loss."""
    return F.l1_loss(same, real)


def spectral_convergence_loss(pred, target):
    """Spectral convergence: Frobenius norm of difference / Frobenius norm of target."""
    return torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-7)


def log_stft_magnitude_loss(pred, target):
    """Log STFT magnitude loss."""
    return F.l1_loss(torch.log(pred + 1e-7), torch.log(target + 1e-7))


def multi_resolution_stft_loss(pred, target, fft_sizes=(256, 512, 1024)):
    """
    Multi-resolution STFT loss on mel-spectrograms.
    Computes spectral convergence + log magnitude at multiple FFT resolutions
    applied along the time axis of the mel-spectrogram.

    Args:
        pred: (B, 1, 80, T) predicted mel
        target: (B, 1, 80, T) target mel
    """
    pred_2d = pred.squeeze(1)     # (B, 80, T)
    target_2d = target.squeeze(1)

    total_loss = 0.0
    for fft_size in fft_sizes:
        hop = fft_size // 4
        win = fft_size

        # Pad time dimension if needed
        T = pred_2d.shape[-1]
        if T < fft_size:
            pad_len = fft_size - T
            pred_pad = F.pad(pred_2d, (0, pad_len))
            target_pad = F.pad(target_2d, (0, pad_len))
        else:
            pred_pad = pred_2d
            target_pad = target_2d

        # STFT along time axis for each mel bin, then average
        # Reshape: (B, 80, T) → (B*80, T)
        B, M, T_pad = pred_pad.shape
        pred_flat = pred_pad.reshape(B * M, T_pad)
        target_flat = target_pad.reshape(B * M, T_pad)

        window = torch.hann_window(win, device=pred.device)
        pred_stft = torch.stft(pred_flat, fft_size, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        target_stft = torch.stft(target_flat, fft_size, hop_length=hop, win_length=win,
                                 window=window, return_complex=True)

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        sc_loss = spectral_convergence_loss(pred_mag, target_mag)
        mag_loss = log_stft_magnitude_loss(pred_mag, target_mag)
        total_loss += sc_loss + mag_loss

    return total_loss / len(fft_sizes)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train():
    # --- Config ---
    DATA_ROOT = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data"
    TRAIN_PKL = os.path.join(DATA_ROOT, "train_dataset.pkl")
    VAL_PKL = os.path.join(DATA_ROOT, "val_dataset.pkl")
    CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    BATCH_SIZE = 4         # smaller batch for small dataset
    EPOCHS = 500           # more epochs since dataset is small
    LR_G = 2e-4            # generator learning rate
    LR_D = 2e-4            # equal LR — D needs to stay strong
    N_D_STEPS = 2          # train D twice per G step to prevent collapse
    LAMBDA_CYCLE = 10.0
    LAMBDA_IDENTITY = 0.5  # low weight: subtle domain diff, don't discourage change
    LAMBDA_STFT = 5.0      # multi-resolution STFT to preserve spectral detail
    MASK_RATIO = 0.25
    TARGET_LEN = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data (with augmentation for training) ---
    ds_a_train = DomainDataset(TRAIN_PKL, label=0, target_len=TARGET_LEN, augment=True)
    ds_b_train = DomainDataset(TRAIN_PKL, label=1, target_len=TARGET_LEN, augment=True)
    ds_a_val = DomainDataset(VAL_PKL, label=0, target_len=TARGET_LEN, augment=False)
    ds_b_val = DomainDataset(VAL_PKL, label=1, target_len=TARGET_LEN, augment=False)

    print(f"Train: {len(ds_a_train)} pre-surgery, {len(ds_b_train)} post-surgery segments")
    print(f"Val:   {len(ds_a_val)} pre-surgery, {len(ds_b_val)} post-surgery segments")

    loader_a = DataLoader(ds_a_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    loader_b = DataLoader(ds_b_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    paired_loader = PairedDomainLoader(loader_a, loader_b)

    val_loader_a = DataLoader(ds_a_val, batch_size=4, shuffle=False)
    val_loader_b = DataLoader(ds_b_val, batch_size=4, shuffle=False)

    # --- Models (reduced capacity for small dataset) ---
    G_A2B = Generator(base_channels=64).to(device)  # pre → post
    G_B2A = Generator(base_channels=64).to(device)  # post → pre
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # --- Optimizers ---
    opt_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
        lr=LR_G, betas=(0.5, 0.999)
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_A.parameters(), D_B.parameters()),
        lr=LR_D, betas=(0.5, 0.999)
    )

    # Linear LR decay after halfway
    def lr_lambda(epoch):
        if epoch < EPOCHS // 2:
            return 1.0
        return 1.0 - (epoch - EPOCHS // 2) / (EPOCHS // 2 + 1)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # --- Logging ---
    history = {
        'g_loss': [], 'g_loss_val': [],
        'd_loss': [], 'cycle_loss': [],
        'identity_loss': [], 'stft_loss': [],
    }
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        G_A2B.train(); G_B2A.train()
        D_A.train(); D_B.train()

        epoch_g, epoch_d, epoch_cyc, epoch_id, epoch_stft = 0, 0, 0, 0, 0
        n_batches = 0

        pbar = tqdm(paired_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for real_A, real_B in pbar:
            real_A = real_A.to(device)  # (B, 1, 80, T)
            real_B = real_B.to(device)

            # ─── Generate masks for FIF ───
            mask_A = generate_mask(real_A.shape, mask_ratio=MASK_RATIO, device=device)
            mask_B = generate_mask(real_B.shape, mask_ratio=MASK_RATIO, device=device)
            masked_A = real_A * mask_A
            masked_B = real_B * mask_B
            ones_A = torch.ones_like(real_A)
            ones_B = torch.ones_like(real_B)

            # ═══════════════════════════════
            # Train Discriminators (N_D_STEPS per G step)
            # ═══════════════════════════════
            for _ in range(N_D_STEPS):
                with torch.no_grad():
                    fake_B_d = G_A2B(masked_A, mask_A)
                    fake_A_d = G_B2A(masked_B, mask_B)

                opt_D.zero_grad()
                loss_d_B = adversarial_loss_d(D_B(real_B), D_B(fake_B_d))
                loss_d_A = adversarial_loss_d(D_A(real_A), D_A(fake_A_d))
                loss_D = (loss_d_A + loss_d_B) * 0.5
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(D_A.parameters(), D_B.parameters()), 1.0)
                opt_D.step()

            # ═══════════════════════════════
            # Train Generators
            # ═══════════════════════════════
            opt_G.zero_grad()

            # Forward conversion with masked input (FIF task)
            fake_B = G_A2B(masked_A, mask_A)    # pre→post (masked input)
            fake_A = G_B2A(masked_B, mask_B)    # post→pre (masked input)

            # Cycle: fake back to original (unmasked)
            recon_A = G_B2A(fake_B, ones_B)     # fake_B → back to A
            recon_B = G_A2B(fake_A, ones_A)     # fake_A → back to B

            # Identity (unmasked)
            idt_A = G_B2A(real_A, ones_A)       # G_B2A(A) ≈ A
            idt_B = G_A2B(real_B, ones_B)       # G_A2B(B) ≈ B

            # Adversarial loss
            loss_g_adv = adversarial_loss_g(D_B(fake_B)) + adversarial_loss_g(D_A(fake_A))

            # Cycle loss
            loss_cyc = cycle_consistency_loss(real_A, recon_A) + cycle_consistency_loss(real_B, recon_B)

            # Identity loss (low weight — subtle domain difference)
            loss_idt = identity_loss(real_A, idt_A) + identity_loss(real_B, idt_B)

            # Multi-resolution STFT loss on cycle reconstruction (preserves harmonics)
            loss_stft = multi_resolution_stft_loss(recon_A, real_A) + multi_resolution_stft_loss(recon_B, real_B)

            loss_G = (loss_g_adv
                      + LAMBDA_CYCLE * loss_cyc
                      + LAMBDA_IDENTITY * loss_idt
                      + LAMBDA_STFT * loss_stft)
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), 1.0)
            opt_G.step()

            # Logging
            epoch_g += loss_G.item()
            epoch_d += loss_D.item()
            epoch_cyc += loss_cyc.item()
            epoch_id += loss_idt.item()
            epoch_stft += loss_stft.item()
            n_batches += 1

            pbar.set_postfix({
                'G': f'{loss_G.item():.3f}',
                'D': f'{loss_D.item():.3f}',
                'cyc': f'{loss_cyc.item():.3f}',
                'stft': f'{loss_stft.item():.3f}',
            })

        # Average losses
        history['g_loss'].append(epoch_g / n_batches)
        history['d_loss'].append(epoch_d / n_batches)
        history['cycle_loss'].append(epoch_cyc / n_batches)
        history['identity_loss'].append(epoch_id / n_batches)
        history['stft_loss'].append(epoch_stft / n_batches)

        sched_G.step()
        sched_D.step()

        # ─── Validation ───
        G_A2B.eval(); G_B2A.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            val_iter_a = iter(val_loader_a)
            val_iter_b = iter(val_loader_b)
            for _ in range(min(len(val_loader_a), len(val_loader_b))):
                va = next(val_iter_a).to(device)
                vb = next(val_iter_b).to(device)
                min_bs = min(va.shape[0], vb.shape[0])
                va, vb = va[:min_bs], vb[:min_bs]

                ones_va = torch.ones_like(va)
                ones_vb = torch.ones_like(vb)
                fake_vb = G_A2B(va, ones_va)
                fake_va = G_B2A(vb, ones_vb)
                recon_va = G_B2A(fake_vb, ones_vb)
                recon_vb = G_A2B(fake_va, ones_va)

                val_cyc = F.l1_loss(recon_va, va) + F.l1_loss(recon_vb, vb)
                val_loss += val_cyc.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        history['g_loss_val'].append(avg_val)

        print(f"Epoch {epoch+1} | G: {history['g_loss'][-1]:.4f} | D: {history['d_loss'][-1]:.4f} | "
              f"Cyc: {history['cycle_loss'][-1]:.4f} | Val Cyc: {avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'G_A2B': G_A2B.state_dict(),
                'G_B2A': G_B2A.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, 'best_mask_cyclegan.pth'))
            print(f"  ✓ Saved best model (val cyc: {avg_val:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'G_A2B': G_A2B.state_dict(),
                'G_B2A': G_B2A.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'mask_cyclegan_epoch{epoch+1}.pth'))

        # ─── Visualize every 10 epochs ───
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_a = next(iter(val_loader_a))[:2].to(device)
                sample_b = next(iter(val_loader_b))[:2].to(device)
                ones_sa = torch.ones_like(sample_a)
                ones_sb = torch.ones_like(sample_b)
                conv_b = G_A2B(sample_a, ones_sa)
                conv_a = G_B2A(sample_b, ones_sb)

                fig, axes = plt.subplots(2, 3, figsize=(15, 6))
                axes[0, 0].set_title("Pre-surgery (real)")
                axes[0, 0].imshow(sample_a[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[0, 1].set_title("→ Post-surgery (fake)")
                axes[0, 1].imshow(conv_b[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[0, 2].set_title("Post-surgery (real)")
                axes[0, 2].imshow(sample_b[0, 0].cpu().numpy(), aspect='auto', origin='lower')

                axes[1, 0].set_title("Post-surgery (real)")
                axes[1, 0].imshow(sample_b[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[1, 1].set_title("→ Pre-surgery (fake)")
                axes[1, 1].imshow(conv_a[0, 0].cpu().numpy(), aspect='auto', origin='lower')
                axes[1, 2].set_title("Pre-surgery (real)")
                axes[1, 2].imshow(sample_a[0, 0].cpu().numpy(), aspect='auto', origin='lower')

                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f'conversion_epoch{epoch+1}.png'), dpi=100)
                plt.close()

    # ─── Final loss plots ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['g_loss'], label='G train')
    axes[0, 0].plot(history['g_loss_val'], label='G val (cycle)')
    axes[0, 0].set_title('Generator Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['d_loss'], label='D loss')
    axes[0, 1].set_title('Discriminator Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['cycle_loss'], label='Cycle')
    axes[1, 0].plot(history['identity_loss'], label='Identity')
    axes[1, 0].set_title('Cycle & Identity Loss'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['stft_loss'], label='STFT')
    axes[1, 1].set_title('Multi-Res STFT Loss'); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"\nTraining complete. Best val cycle loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()
