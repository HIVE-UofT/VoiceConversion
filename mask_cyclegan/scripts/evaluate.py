"""
Evaluate MaskCycleGAN-VC on the held-out test set.

Loads test_dataset.pkl, converts mel-spectrograms through the generators,
and computes objective metrics:
  - Mel Cepstral Distortion (MCD): lower = more similar spectral shape
  - F0 Correlation: higher = better pitch tracking
  - Cycle Consistency (L1): lower = better reconstruction A→B→A
  - Identity Preservation (L1): lower = generator preserves same-domain input

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint checkpoints/mask_cyclegan_epoch300.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import pickle
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

from model.mask_cyclegan import Generator


# ──────────────────────────────────────────────
# Audio params (must match training/inference)
# ──────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
TARGET_LEN = 400


# ──────────────────────────────────────────────
# Dataset (same as train.py, no augmentation)
# ──────────────────────────────────────────────

class DomainDataset(Dataset):
    def __init__(self, pkl_path, label, target_len=TARGET_LEN):
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        self.data = [d for d in all_data if d['label'] == label]
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel_spectrogram'].copy()  # (80, T)
        if mel.shape[1] > self.target_len:
            mel = mel[:, :self.target_len]
        elif mel.shape[1] < self.target_len:
            pad = self.target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, 80, T)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_mcd(ref_mel, synth_mel):
    """Mel Cepstral Distortion between two mel-spectrograms (numpy, dB scale)."""
    min_len = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel = ref_mel[:, :min_len]
    synth_mel = synth_mel[:, :min_len]

    ref_mfcc = librosa.feature.mfcc(S=ref_mel, n_mfcc=13)
    synth_mfcc = librosa.feature.mfcc(S=synth_mel, n_mfcc=13)

    min_len = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - synth_mfcc[:, :min_len]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))
    return mcd


def compute_f0_corr(mel_a, mel_b, sr=SAMPLE_RATE):
    """F0 correlation between two mel-spectrograms (via Griffin-Lim reconstruction)."""
    # Denormalize: [0,1] → dB → power
    mel_a_db = mel_a * 80 - 80
    mel_b_db = mel_b * 80 - 80
    mel_a_power = librosa.db_to_power(mel_a_db)
    mel_b_power = librosa.db_to_power(mel_b_db)

    y_a = librosa.feature.inverse.mel_to_audio(mel_a_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=32)
    y_b = librosa.feature.inverse.mel_to_audio(mel_b_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=32)

    f0_a, _, _ = librosa.pyin(y_a, fmin=50, fmax=500, sr=sr)
    f0_b, _, _ = librosa.pyin(y_b, fmin=50, fmax=500, sr=sr)

    min_len = min(len(f0_a), len(f0_b))
    f0_a = f0_a[:min_len]
    f0_b = f0_b[:min_len]
    valid = ~np.isnan(f0_a) & ~np.isnan(f0_b)

    if valid.sum() < 10:
        return float('nan')
    return np.corrcoef(f0_a[valid], f0_b[valid])[0, 1]


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def evaluate():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MaskCycleGAN-VC on test set")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_mask_cyclegan.pth'))
    parser.add_argument('--test_pkl', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/test_dataset.pkl")
    parser.add_argument('--skip_f0', action='store_true', help='Skip F0 correlation (slow)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models (base_channels=64 matches retrained checkpoints)
    G_A2B = Generator(base_channels=64).to(device)
    G_B2A = Generator(base_channels=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    G_A2B.load_state_dict(checkpoint['G_A2B'])
    G_B2A.load_state_dict(checkpoint['G_B2A'])
    G_A2B.eval()
    G_B2A.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load test data
    ds_a = DomainDataset(args.test_pkl, label=0)
    ds_b = DomainDataset(args.test_pkl, label=1)
    loader_a = DataLoader(ds_a, batch_size=1, shuffle=False)
    loader_b = DataLoader(ds_b, batch_size=1, shuffle=False)
    print(f"Test set: {len(ds_a)} pre-surgery, {len(ds_b)} post-surgery segments\n")

    # ─── Metrics storage ───
    mcd_a2b = []     # MCD of converted vs real post-surgery
    mcd_b2a = []     # MCD of converted vs real pre-surgery
    cycle_a = []     # L1 of A→B→A reconstruction
    cycle_b = []     # L1 of B→A→B reconstruction
    idt_a = []       # L1 of G_B2A(A) vs A (identity)
    idt_b = []       # L1 of G_A2B(B) vs B (identity)
    f0_a2b = []      # F0 corr: converted B vs real B
    f0_b2a = []      # F0 corr: converted A vs real A

    # Collect all real post-surgery mels for MCD reference
    real_b_mels = []
    for batch in loader_b:
        real_b_mels.append(batch.squeeze(0).squeeze(0).numpy())
    real_a_mels = []
    for batch in loader_a:
        real_a_mels.append(batch.squeeze(0).squeeze(0).numpy())

    # Re-create loaders after consuming them
    loader_a = DataLoader(ds_a, batch_size=1, shuffle=False)
    loader_b = DataLoader(ds_b, batch_size=1, shuffle=False)

    # ─── Evaluate A→B direction (pre→post) ───
    print("Evaluating A→B (pre-surgery → post-surgery)...")
    with torch.no_grad():
        for i, real_A in enumerate(loader_a):
            real_A = real_A.to(device)
            ones = torch.ones_like(real_A)

            # Convert A→B
            fake_B = G_A2B(real_A, ones)

            # Cycle A→B→A
            recon_A = G_B2A(fake_B, ones)
            cyc_loss = F.l1_loss(recon_A, real_A).item()
            cycle_a.append(cyc_loss)

            # Identity: G_B2A(A) should ≈ A
            idt = G_B2A(real_A, ones)
            idt_loss = F.l1_loss(idt, real_A).item()
            idt_a.append(idt_loss)

            # MCD: converted mel vs best-matching real post-surgery mel
            fake_B_np = fake_B.squeeze().cpu().numpy()
            # Denormalize to dB for MCD
            fake_B_db = fake_B_np * 80 - 80
            file_mcds = []
            for rb in real_b_mels:
                rb_db = rb * 80 - 80
                mcd = compute_mcd(rb_db, fake_B_db)
                file_mcds.append(mcd)
            best_mcd = min(file_mcds)
            mcd_a2b.append(best_mcd)

            # F0 correlation (converted vs best-matching real)
            if not args.skip_f0:
                best_idx = np.argmin(file_mcds)
                f0_c = compute_f0_corr(fake_B_np, real_b_mels[best_idx])
                f0_a2b.append(f0_c)

            print(f"  [{i+1}/{len(ds_a)}] MCD={best_mcd:.2f}  Cycle={cyc_loss:.4f}  Idt={idt_loss:.4f}")

    # ─── Evaluate B→A direction (post→pre) ───
    print("\nEvaluating B→A (post-surgery → pre-surgery)...")
    with torch.no_grad():
        for i, real_B in enumerate(loader_b):
            real_B = real_B.to(device)
            ones = torch.ones_like(real_B)

            # Convert B→A
            fake_A = G_B2A(real_B, ones)

            # Cycle B→A→B
            recon_B = G_A2B(fake_A, ones)
            cyc_loss = F.l1_loss(recon_B, real_B).item()
            cycle_b.append(cyc_loss)

            # Identity: G_A2B(B) should ≈ B
            idt = G_A2B(real_B, ones)
            idt_loss = F.l1_loss(idt, real_B).item()
            idt_b.append(idt_loss)

            # MCD: converted mel vs best-matching real pre-surgery mel
            fake_A_np = fake_A.squeeze().cpu().numpy()
            fake_A_db = fake_A_np * 80 - 80
            file_mcds = []
            for ra in real_a_mels:
                ra_db = ra * 80 - 80
                mcd = compute_mcd(ra_db, fake_A_db)
                file_mcds.append(mcd)
            best_mcd = min(file_mcds)
            mcd_b2a.append(best_mcd)

            if not args.skip_f0:
                best_idx = np.argmin(file_mcds)
                f0_c = compute_f0_corr(fake_A_np, real_a_mels[best_idx])
                f0_b2a.append(f0_c)

            print(f"  [{i+1}/{len(ds_b)}] MCD={best_mcd:.2f}  Cycle={cyc_loss:.4f}  Idt={idt_loss:.4f}")

    # ─── Summary ───
    print(f"\n{'='*55}")
    print(f"  MaskCycleGAN-VC Test Set Evaluation")
    print(f"{'='*55}")

    print(f"\n  A→B (pre→post) — {len(mcd_a2b)} samples:")
    print(f"    MCD (lower=better):        {np.mean(mcd_a2b):.2f} ± {np.std(mcd_a2b):.2f}")
    print(f"    Cycle L1 (lower=better):   {np.mean(cycle_a):.4f} ± {np.std(cycle_a):.4f}")
    print(f"    Identity L1 (lower=better): {np.mean(idt_a):.4f} ± {np.std(idt_a):.4f}")

    print(f"\n  B→A (post→pre) — {len(mcd_b2a)} samples:")
    print(f"    MCD (lower=better):        {np.mean(mcd_b2a):.2f} ± {np.std(mcd_b2a):.2f}")
    print(f"    Cycle L1 (lower=better):   {np.mean(cycle_b):.4f} ± {np.std(cycle_b):.4f}")
    print(f"    Identity L1 (lower=better): {np.mean(idt_b):.4f} ± {np.std(idt_b):.4f}")

    if not args.skip_f0:
        valid_f0_a2b = [f for f in f0_a2b if not np.isnan(f)]
        valid_f0_b2a = [f for f in f0_b2a if not np.isnan(f)]
        if valid_f0_a2b:
            print(f"\n  F0 Correlation (higher=better):")
            print(f"    A→B: {np.mean(valid_f0_a2b):.3f} ± {np.std(valid_f0_a2b):.3f}  ({len(valid_f0_a2b)}/{len(f0_a2b)} valid)")
        if valid_f0_b2a:
            print(f"    B→A: {np.mean(valid_f0_b2a):.3f} ± {np.std(valid_f0_b2a):.3f}  ({len(valid_f0_b2a)}/{len(f0_b2a)} valid)")

    print(f"\n  Overall:")
    all_mcd = mcd_a2b + mcd_b2a
    all_cycle = cycle_a + cycle_b
    print(f"    MCD:   {np.mean(all_mcd):.2f} ± {np.std(all_mcd):.2f}")
    print(f"    Cycle: {np.mean(all_cycle):.4f} ± {np.std(all_cycle):.4f}")
    print(f"{'='*55}")


if __name__ == '__main__':
    evaluate()
