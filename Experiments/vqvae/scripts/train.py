"""
Training script for VQVAE Voice Conversion.

Disentangles voice into:
  - Content (quantized via VQ codebook) — what is being said
  - Voice quality (continuous vector) — how it sounds (resonance, nasality)

Losses:
  - Reconstruction: L1 on mel-spectrogram
  - VQ commitment: keeps encoder output close to codebook
  - Adversarial disentanglement: content encoder must NOT encode surgery status
  - Quality classification: quality encoder MUST encode surgery status
  - Multi-resolution STFT: preserves spectral detail
  - Cycle (cross-reconstruction): swap quality between domains, re-encode,
    verify content preserved and quality matches target

Domain A = pre-surgery (label=0)
Domain B = post-surgery (label=1)

Usage:
  python scripts/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.vqvae import VQVAE, DomainClassifier, gradient_reversal


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class MelDataset(Dataset):
    """Loads mel-spectrograms with labels (0=pre, 1=post surgery)."""
    def __init__(self, pkl_path, target_len=400, augment=False, label_filter=None):
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        if label_filter is not None:
            self.data = [d for d in all_data if d['label'] == label_filter]
        else:
            self.data = all_data
        self.target_len = target_len
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel_spectrogram'].copy()  # (80, T)
        label = self.data[idx]['label']  # 0 or 1

        # Time crop/pad
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
            # Random frequency masking
            if np.random.rand() > 0.5:
                n_mask = np.random.randint(1, 6)
                f_start = np.random.randint(0, mel.shape[0] - n_mask)
                mel[f_start:f_start + n_mask, :] = 0.0

            # Random amplitude scaling (+-10%)
            if np.random.rand() > 0.5:
                scale = 0.9 + np.random.rand() * 0.2
                mel = np.clip(mel * scale, 0, 1)

        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 80, T)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return mel_tensor, label_tensor


class PairedDomainLoader:
    """Yields (pre_batch, post_batch) pairs, cycling the shorter domain."""
    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        for _ in range(max(len(self.loader_a), len(self.loader_b))):
            try:
                a_mel, a_lab = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a_mel, a_lab = next(iter_a)
            try:
                b_mel, b_lab = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b_mel, b_lab = next(iter_b)
            # Match batch sizes
            min_bs = min(a_mel.shape[0], b_mel.shape[0])
            yield a_mel[:min_bs], a_lab[:min_bs], b_mel[:min_bs], b_lab[:min_bs]

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def spectral_convergence_loss(pred, target):
    return torch.norm(target - pred, p='fro') / (torch.norm(target, p='fro') + 1e-7)


def log_stft_magnitude_loss(pred, target):
    return F.l1_loss(torch.log(pred + 1e-7), torch.log(target + 1e-7))


def multi_resolution_stft_loss(pred, target, fft_sizes=(256, 512, 1024)):
    """Multi-resolution STFT loss on mel-spectrograms."""
    pred_2d = pred.squeeze(1)     # (B, 80, T)
    target_2d = target.squeeze(1)

    total_loss = 0.0
    for fft_size in fft_sizes:
        hop = fft_size // 4
        win = fft_size

        T = pred_2d.shape[-1]
        if T < fft_size:
            pad_len = fft_size - T
            pred_pad = F.pad(pred_2d, (0, pad_len))
            target_pad = F.pad(target_2d, (0, pad_len))
        else:
            pred_pad = pred_2d
            target_pad = target_2d

        B, M, T_pad = pred_pad.shape
        pred_flat = pred_pad.reshape(B * M, T_pad)
        target_flat = target_pad.reshape(B * M, T_pad)

        window = torch.hann_window(win, device=pred.device)
        pred_stft = torch.stft(pred_flat, fft_size, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        target_stft = torch.stft(target_flat, fft_size, hop_length=hop, win_length=win,
                                 window=window, return_complex=True)

        sc = spectral_convergence_loss(pred_stft.abs(), target_stft.abs())
        mag = log_stft_magnitude_loss(pred_stft.abs(), target_stft.abs())
        total_loss += sc + mag

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

    # Model hyperparameters
    CODE_DIM = 64
    NUM_CODES = 16             # per-head codebook size (4 heads × 16 = 65K combinations)
    NUM_HEADS = 4              # product VQ heads
    QUALITY_DIM = 32
    COMMITMENT_WEIGHT = 0.25   # lower: let encoder explore freely (was 1.0)
    EMA_DECAY = 0.95           # faster codebook adaptation (was 0.99)
    ENTROPY_WEIGHT = 0.1       # entropy regularization for uniform code usage

    # Training hyperparameters
    BATCH_SIZE = 4
    EPOCHS = 400
    LR = 2e-4
    LR_ADV = 2e-4

    # Loss weights
    LAMBDA_RECON = 1.0
    LAMBDA_VQ = 1.0            # includes commitment + entropy regularization
    LAMBDA_STFT = 2.0
    LAMBDA_ADV = 1.0
    LAMBDA_QUAL_CLS = 2.0
    LAMBDA_CYCLE = 5.0

    TARGET_LEN = 400

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data: domain-specific loaders for paired batches ---
    ds_pre_train = MelDataset(TRAIN_PKL, target_len=TARGET_LEN, augment=True, label_filter=0)
    ds_post_train = MelDataset(TRAIN_PKL, target_len=TARGET_LEN, augment=True, label_filter=1)
    ds_val = MelDataset(VAL_PKL, target_len=TARGET_LEN, augment=False)

    print(f"Train: {len(ds_pre_train)} pre-surgery, {len(ds_post_train)} post-surgery")
    print(f"Val:   {len(ds_val)} segments")

    loader_pre = DataLoader(ds_pre_train, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=2)
    loader_post = DataLoader(ds_post_train, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2)
    paired_loader = PairedDomainLoader(loader_pre, loader_post)

    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Models ---
    model = VQVAE(code_dim=CODE_DIM, num_codes=NUM_CODES, num_heads=NUM_HEADS,
                  quality_dim=QUALITY_DIM, commitment_weight=COMMITMENT_WEIGHT,
                  ema_decay=EMA_DECAY, entropy_weight=ENTROPY_WEIGHT).to(device)
    domain_cls = DomainClassifier(code_dim=CODE_DIM).to(device)
    quality_cls = nn.Linear(QUALITY_DIM, 1).to(device)

    # --- Optimizers ---
    opt_model = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    opt_adv = torch.optim.Adam(domain_cls.parameters(), lr=LR_ADV, betas=(0.9, 0.999))
    opt_qual = torch.optim.Adam(quality_cls.parameters(), lr=LR_ADV, betas=(0.9, 0.999))

    sched_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=1e-6)
    sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=EPOCHS, eta_min=1e-6)

    # --- Logging ---
    history = {
        'recon_loss': [], 'vq_loss': [], 'stft_loss': [],
        'adv_loss': [], 'qual_cls_loss': [], 'cycle_loss': [],
        'perplexity': [], 'val_recon': [],
    }
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        domain_cls.train()
        quality_cls.train()

        ep_recon, ep_vq, ep_stft, ep_adv, ep_qual, ep_cycle, ep_perp = 0, 0, 0, 0, 0, 0, 0
        n_batches = 0

        pbar = tqdm(paired_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for mel_pre, lab_pre, mel_post, lab_post in pbar:
            mel_pre = mel_pre.to(device)     # (B, 1, 80, T) pre-surgery
            mel_post = mel_post.to(device)   # (B, 1, 80, T) post-surgery
            lab_pre = lab_pre.to(device)     # (B,) all 0s
            lab_post = lab_post.to(device)   # (B,) all 1s

            # Combine into one batch for reconstruction + adversarial losses
            mel = torch.cat([mel_pre, mel_post], dim=0)       # (2B, 1, 80, T)
            labels = torch.cat([lab_pre, lab_post], dim=0)    # (2B,)

            # ═══════════════════════════════
            # Step 1: Train adversarial domain classifier
            # ═══════════════════════════════
            with torch.no_grad():
                content_z_all = model.content_encoder(mel)

            opt_adv.zero_grad()
            adv_pred = domain_cls(content_z_all.detach())
            loss_adv_cls = F.binary_cross_entropy_with_logits(adv_pred.squeeze(1), labels)
            loss_adv_cls.backward()
            opt_adv.step()

            # ═══════════════════════════════
            # Step 2: Train VQVAE + quality classifier
            # ═══════════════════════════════
            opt_model.zero_grad()
            opt_qual.zero_grad()

            # --- Self-reconstruction ---
            recon, vq_loss, perplexity, content_z = model(mel)
            loss_recon = F.l1_loss(recon, mel)
            loss_stft = multi_resolution_stft_loss(recon, mel)

            # --- Adversarial disentanglement ---
            content_reversed = gradient_reversal(content_z, alpha=LAMBDA_ADV)
            adv_pred_gr = domain_cls(content_reversed)
            loss_adv_g = F.binary_cross_entropy_with_logits(adv_pred_gr.squeeze(1), labels)

            # --- Quality classification ---
            quality = model.quality_encoder(mel)
            qual_pred = quality_cls(quality)
            loss_qual = F.binary_cross_entropy_with_logits(qual_pred.squeeze(1), labels)

            # ═══════════════════════════════
            # Step 3: Cycle (cross-reconstruction) loss
            #
            # Take pre content + post quality → cross_recon
            # Re-encode cross_recon → verify:
            #   - content codes match original pre content codes
            #   - quality matches post quality
            # Same in reverse direction.
            # ═══════════════════════════════
            B = mel_pre.shape[0]

            # Encode each domain separately
            content_z_pre = model.content_encoder(mel_pre)          # (B, D, T')
            content_q_pre, _, _ = model.vq(content_z_pre)           # quantized
            quality_pre = model.quality_encoder(mel_pre)             # (B, quality_dim)

            content_z_post = model.content_encoder(mel_post)
            content_q_post, _, _ = model.vq(content_z_post)
            quality_post = model.quality_encoder(mel_post)

            # Cross-reconstruct: pre content + post quality
            cross_pre2post = model.decoder(content_q_pre, quality_post)
            cross_pre2post = model._match_size(cross_pre2post, mel_pre)

            # Cross-reconstruct: post content + pre quality
            cross_post2pre = model.decoder(content_q_post, quality_pre)
            cross_post2pre = model._match_size(cross_post2pre, mel_post)

            # Re-encode the cross-reconstructed outputs
            re_content_z_a2b = model.content_encoder(cross_pre2post)
            re_content_q_a2b, _, _ = model.vq(re_content_z_a2b)
            re_quality_a2b = model.quality_encoder(cross_pre2post)

            re_content_z_b2a = model.content_encoder(cross_post2pre)
            re_content_q_b2a, _, _ = model.vq(re_content_z_b2a)
            re_quality_b2a = model.quality_encoder(cross_post2pre)

            # Cycle content loss: re-encoded content should match original content
            loss_cycle_content = (F.l1_loss(re_content_q_a2b, content_q_pre.detach())
                                + F.l1_loss(re_content_q_b2a, content_q_post.detach()))

            # Cycle quality loss: re-encoded quality should match the target quality
            loss_cycle_quality = (F.l1_loss(re_quality_a2b, quality_post.detach())
                                + F.l1_loss(re_quality_b2a, quality_pre.detach()))

            loss_cycle = loss_cycle_content + loss_cycle_quality

            # ═══════════════════════════════
            # Total loss
            # ═══════════════════════════════
            loss_total = (LAMBDA_RECON * loss_recon
                         + LAMBDA_VQ * vq_loss
                         + LAMBDA_STFT * loss_stft
                         + loss_adv_g
                         + LAMBDA_QUAL_CLS * loss_qual
                         + LAMBDA_CYCLE * loss_cycle)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step()
            opt_qual.step()

            # Logging
            ep_recon += loss_recon.item()
            ep_vq += vq_loss.item()
            ep_stft += loss_stft.item()
            ep_adv += loss_adv_cls.item()
            ep_qual += loss_qual.item()
            ep_cycle += loss_cycle.item()
            ep_perp += perplexity.item()
            n_batches += 1

            pbar.set_postfix({
                'recon': f'{loss_recon.item():.3f}',
                'cycle': f'{loss_cycle.item():.3f}',
                'perp': f'{perplexity.item():.0f}',
                'adv': f'{loss_adv_cls.item():.3f}',
            })

        # Average
        for key, val in [('recon_loss', ep_recon), ('vq_loss', ep_vq),
                         ('stft_loss', ep_stft), ('adv_loss', ep_adv),
                         ('qual_cls_loss', ep_qual), ('cycle_loss', ep_cycle),
                         ('perplexity', ep_perp)]:
            history[key].append(val / n_batches)

        sched_model.step()
        sched_adv.step()

        # ─── Validation ───
        model.eval()
        val_recon = 0
        n_val = 0
        with torch.no_grad():
            for mel_v, _ in val_loader:
                mel_v = mel_v.to(device)
                recon_v, _, _, _ = model(mel_v)
                val_recon += F.l1_loss(recon_v, mel_v).item()
                n_val += 1

        avg_val = val_recon / max(n_val, 1)
        history['val_recon'].append(avg_val)

        print(f"Epoch {epoch+1} | Recon: {history['recon_loss'][-1]:.4f} | "
              f"VQ: {history['vq_loss'][-1]:.4f} | Perp: {history['perplexity'][-1]:.0f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Adv: {history['adv_loss'][-1]:.4f} | Qual: {history['qual_cls_loss'][-1]:.4f} | "
              f"Val Recon: {avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'domain_cls': domain_cls.state_dict(),
                'quality_cls': quality_cls.state_dict(),
                'opt_model': opt_model.state_dict(),
                'opt_adv': opt_adv.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, 'best_vqvae.pth'))
            print(f"  -> Saved best model (val recon: {avg_val:.4f})")

        # Periodic checkpoints
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'domain_cls': domain_cls.state_dict(),
                'quality_cls': quality_cls.state_dict(),
                'opt_model': opt_model.state_dict(),
                'opt_adv': opt_adv.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'vqvae_epoch{epoch+1}.pth'))

        # ─── Visualize every 10 epochs ───
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Get one pre-surgery and one post-surgery sample from validation
                pre_samples, post_samples = [], []
                for mel_v, lab_v in val_loader:
                    for i in range(mel_v.shape[0]):
                        if lab_v[i].item() == 0 and len(pre_samples) < 1:
                            pre_samples.append(mel_v[i:i+1])
                        elif lab_v[i].item() == 1 and len(post_samples) < 1:
                            post_samples.append(mel_v[i:i+1])
                        if len(pre_samples) >= 1 and len(post_samples) >= 1:
                            break
                    if len(pre_samples) >= 1 and len(post_samples) >= 1:
                        break

                pre_mel = pre_samples[0].to(device)
                post_mel = post_samples[0].to(device)

                # Reconstruction
                recon_pre, _, _, _ = model(pre_mel)
                recon_post, _, _, _ = model(post_mel)

                # Conversion: pre content + post quality
                converted_pre2post = model.convert(pre_mel, post_mel)
                # Conversion: post content + pre quality
                converted_post2pre = model.convert(post_mel, pre_mel)

                fig, axes = plt.subplots(3, 4, figsize=(20, 10))
                titles = [
                    ["Pre (real)", "Pre (recon)", "Post (real)", "Post (recon)"],
                    ["Pre (real)", "Pre->Post (converted)", "Post (real)", "Post->Pre (converted)"],
                    ["Diff: real-recon (pre)", "Diff: real-recon (post)",
                     "Diff: pre real vs converted", "Diff: post real vs converted"],
                ]
                images = [
                    [pre_mel, recon_pre, post_mel, recon_post],
                    [pre_mel, converted_pre2post, post_mel, converted_post2pre],
                    [pre_mel - recon_pre, post_mel - recon_post,
                     post_mel - converted_pre2post, pre_mel - converted_post2pre],
                ]

                for row in range(3):
                    for col in range(4):
                        ax = axes[row, col]
                        img = images[row][col][0, 0].cpu().numpy()
                        if row == 2:
                            ax.imshow(img, aspect='auto', origin='lower', cmap='RdBu', vmin=-0.5, vmax=0.5)
                        else:
                            ax.imshow(img, aspect='auto', origin='lower', vmin=0, vmax=1)
                        ax.set_title(titles[row][col], fontsize=9)

                plt.tight_layout()
                plt.savefig(os.path.join(PLOT_DIR, f'vqvae_epoch{epoch+1}.png'), dpi=100)
                plt.close()

    # ─── Final loss plots ───
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    axes[0, 0].plot(history['recon_loss'], label='Train')
    axes[0, 0].plot(history['val_recon'], label='Val')
    axes[0, 0].set_title('Reconstruction Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['vq_loss'], label='VQ')
    axes[0, 1].set_title('VQ Commitment Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[0, 2].plot(history['perplexity'], label='Avg Perplexity')
    axes[0, 2].axhline(y=NUM_CODES, color='r', linestyle='--', label=f'Max per head ({NUM_CODES})')
    axes[0, 2].set_title(f'Codebook Perplexity ({NUM_HEADS} heads × {NUM_CODES} codes)'); axes[0, 2].legend(); axes[0, 2].grid(True)

    axes[0, 3].plot(history['cycle_loss'], label='Cycle')
    axes[0, 3].set_title('Cycle (Cross-Recon) Loss'); axes[0, 3].legend(); axes[0, 3].grid(True)

    axes[1, 0].plot(history['stft_loss'], label='STFT')
    axes[1, 0].set_title('Multi-Res STFT Loss'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['adv_loss'], label='Adv (content)')
    axes[1, 1].axhline(y=0.693, color='r', linestyle='--', label='Random (0.693)')
    axes[1, 1].set_title('Adversarial Loss (want ~0.693)'); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].plot(history['qual_cls_loss'], label='Quality cls')
    axes[1, 2].set_title('Quality Classification Loss (want low)'); axes[1, 2].legend(); axes[1, 2].grid(True)

    axes[1, 3].axis('off')  # empty subplot

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"\nTraining complete. Best val recon loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    train()
