"""
Evaluate VQVAE Voice Conversion on the held-out test set.

Metrics:
  - Reconstruction L1: how well the model reconstructs input mel
  - Mel Cepstral Distortion (MCD): spectral distance between converted and real target
  - F0 Correlation: pitch tracking between source and converted
  - Content Preservation MCD: source vs converted (content should be preserved)
  - Quality Disentanglement: accuracy of domain classifier on content vs quality

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --checkpoint checkpoints/vqvae_epoch200.pth
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader

from model.vqvae import VQVAE, DomainClassifier


# ──────────────────────────────────────────────
# Audio params (must match dataset_processing.py)
# ──────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
TARGET_LEN = 400


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class MelDataset(Dataset):
    def __init__(self, pkl_path, target_len=TARGET_LEN, label_filter=None):
        with open(pkl_path, 'rb') as f:
            all_data = pickle.load(f)
        if label_filter is not None:
            self.data = [d for d in all_data if d['label'] == label_filter]
        else:
            self.data = all_data
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel = self.data[idx]['mel_spectrogram'].copy()
        label = self.data[idx]['label']
        if mel.shape[1] > self.target_len:
            mel = mel[:, :self.target_len]
        elif mel.shape[1] < self.target_len:
            pad = self.target_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
        return torch.from_numpy(mel).float().unsqueeze(0), torch.tensor(label, dtype=torch.float32)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_mcd(ref_mel, synth_mel):
    """Mel Cepstral Distortion (dB scale mels)."""
    min_len = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel = ref_mel[:, :min_len]
    synth_mel = synth_mel[:, :min_len]
    ref_mfcc = librosa.feature.mfcc(S=ref_mel, n_mfcc=13)
    synth_mfcc = librosa.feature.mfcc(S=synth_mel, n_mfcc=13)
    min_len = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - synth_mfcc[:, :min_len]
    return np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))


def compute_f0_corr(mel_a, mel_b, sr=SAMPLE_RATE):
    """F0 correlation via Griffin-Lim reconstruction."""
    mel_a_db = mel_a * 80 - 80
    mel_b_db = mel_b * 80 - 80
    mel_a_power = librosa.db_to_power(mel_a_db)
    mel_b_power = librosa.db_to_power(mel_b_db)

    y_a = librosa.feature.inverse.mel_to_audio(mel_a_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=32)
    y_b = librosa.feature.inverse.mel_to_audio(mel_b_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=32)

    f0_a, _, _ = librosa.pyin(y_a, fmin=50, fmax=500, sr=sr)
    f0_b, _, _ = librosa.pyin(y_b, fmin=50, fmax=500, sr=sr)

    min_len = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:min_len], f0_b[:min_len]
    valid = ~np.isnan(f0_a) & ~np.isnan(f0_b)
    if valid.sum() < 10:
        return float('nan')
    return np.corrcoef(f0_a[valid], f0_b[valid])[0, 1]


def compute_avg_quality(model, pkl_path, label, device, max_samples=50):
    """Compute average quality vector for a domain."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    domain_data = [d for d in data if d['label'] == label]
    if len(domain_data) > max_samples:
        indices = np.random.RandomState(42).choice(len(domain_data), max_samples, replace=False)
        domain_data = [domain_data[i] for i in indices]

    qualities = []
    with torch.no_grad():
        for d in domain_data:
            mel = d['mel_spectrogram'].copy()
            if mel.shape[1] > TARGET_LEN:
                mel = mel[:, :TARGET_LEN]
            elif mel.shape[1] < TARGET_LEN:
                pad = TARGET_LEN - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')
            mel_t = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
            q = model.quality_encoder(mel_t)
            qualities.append(q)
    return torch.stack(qualities).mean(dim=0)


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate VQVAE on test set")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_vqvae.pth'))
    parser.add_argument('--test_pkl', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/test_dataset.pkl")
    parser.add_argument('--train_pkl', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/train_dataset.pkl")
    parser.add_argument('--skip_f0', action='store_true', help='Skip F0 correlation (slow)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = VQVAE(code_dim=64, num_codes=16, num_heads=4, quality_dim=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?') + 1})")

    # Load test data
    ds_pre = MelDataset(args.test_pkl, label_filter=0)
    ds_post = MelDataset(args.test_pkl, label_filter=1)
    ds_all = MelDataset(args.test_pkl, label_filter=None)
    loader_pre = DataLoader(ds_pre, batch_size=1, shuffle=False)
    loader_post = DataLoader(ds_post, batch_size=1, shuffle=False)
    loader_all = DataLoader(ds_all, batch_size=1, shuffle=False)
    print(f"Test set: {len(ds_pre)} pre-surgery, {len(ds_post)} post-surgery segments\n")

    # Compute average quality vectors from training data
    print("Computing average quality vectors from training set...")
    avg_quality_pre = compute_avg_quality(model, args.train_pkl, label=0, device=device)
    avg_quality_post = compute_avg_quality(model, args.train_pkl, label=1, device=device)
    print(f"  Pre-surgery quality:  {avg_quality_pre.squeeze().cpu().numpy()[:4]}...")
    print(f"  Post-surgery quality: {avg_quality_post.squeeze().cpu().numpy()[:4]}...")
    quality_distance = F.l1_loss(avg_quality_pre, avg_quality_post).item()
    print(f"  Quality vector L1 distance: {quality_distance:.4f}\n")

    # ─── Metric 1: Reconstruction quality ───
    print("1. Reconstruction quality...")
    recon_losses = []
    with torch.no_grad():
        for mel, _ in loader_all:
            mel = mel.to(device)
            recon, _, perp, _ = model(mel)
            recon_losses.append(F.l1_loss(recon, mel).item())
    print(f"   Recon L1: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}")

    # ─── Metric 2: Disentanglement — can content predict surgery? ───
    print("\n2. Disentanglement check...")
    content_preds, quality_preds, true_labels = [], [], []
    with torch.no_grad():
        for mel, label in loader_all:
            mel = mel.to(device)
            content_z = model.content_encoder(mel)
            quality = model.quality_encoder(mel)

            # Content should NOT predict surgery (accuracy ~50% = good)
            # Quality SHOULD predict surgery (accuracy ~100% = good)
            content_preds.append(content_z.mean(dim=[1, 2]).cpu())  # rough global feature
            quality_preds.append(quality.cpu())
            true_labels.append(label)

    # Simple linear probe on content features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    content_feats = torch.cat(content_preds).numpy()
    quality_feats = torch.cat(quality_preds).numpy()
    labels_np = torch.cat(true_labels).numpy()

    if len(np.unique(labels_np)) > 1:
        # Content probe (want ~50%)
        clf_content = LogisticRegression(max_iter=200).fit(content_feats, labels_np)
        content_acc = accuracy_score(labels_np, clf_content.predict(content_feats))

        # Quality probe (want ~100%)
        clf_quality = LogisticRegression(max_iter=200).fit(quality_feats, labels_np)
        quality_acc = accuracy_score(labels_np, clf_quality.predict(quality_feats))

        print(f"   Content domain accuracy: {content_acc:.1%} (want ~50% = disentangled)")
        print(f"   Quality domain accuracy: {quality_acc:.1%} (want ~100% = informative)")
    else:
        print("   Skipped — only one class in test set")

    # ─── Metric 3: Conversion quality (A→B) ───
    print("\n3. Conversion quality: Pre → Post surgery...")
    mcd_a2b, f0_a2b, content_pres_a2b = [], [], []

    # Collect real post mels for MCD reference
    real_post_mels = []
    for mel, _ in loader_post:
        real_post_mels.append(mel.squeeze(0).squeeze(0).numpy())

    with torch.no_grad():
        for i, (mel, _) in enumerate(loader_pre):
            mel = mel.to(device)
            converted = model.convert(mel, torch.zeros(1))  # dummy, we'll use avg quality
            # Actually use avg quality
            content_z = model.content_encoder(mel)
            content_q, _, _ = model.vq(content_z)
            converted = model.decoder(content_q, avg_quality_post)
            converted = model._match_size(converted, mel)

            conv_np = converted[0, 0].cpu().numpy()
            src_np = mel[0, 0].cpu().numpy()

            # MCD vs best-matching real post-surgery
            conv_db = conv_np * 80 - 80
            file_mcds = [compute_mcd(rb * 80 - 80, conv_db) for rb in real_post_mels]
            mcd_a2b.append(min(file_mcds))

            # Content preservation: MCD between source and converted (lower=better preserved)
            src_db = src_np * 80 - 80
            content_pres_a2b.append(compute_mcd(src_db, conv_db))

            # F0 correlation
            if not args.skip_f0:
                f0_c = compute_f0_corr(src_np, conv_np)
                f0_a2b.append(f0_c)

            print(f"   [{i+1}/{len(ds_pre)}] MCD(target)={min(file_mcds):.2f}  MCD(content)={content_pres_a2b[-1]:.2f}")

    # ─── Metric 4: Conversion quality (B→A) ───
    print("\n4. Conversion quality: Post → Pre surgery...")
    mcd_b2a, f0_b2a, content_pres_b2a = [], [], []

    real_pre_mels = []
    for mel, _ in loader_pre:
        real_pre_mels.append(mel.squeeze(0).squeeze(0).numpy())

    # Re-create loader
    loader_post = DataLoader(ds_post, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (mel, _) in enumerate(loader_post):
            mel = mel.to(device)
            content_z = model.content_encoder(mel)
            content_q, _, _ = model.vq(content_z)
            converted = model.decoder(content_q, avg_quality_pre)
            converted = model._match_size(converted, mel)

            conv_np = converted[0, 0].cpu().numpy()
            src_np = mel[0, 0].cpu().numpy()

            conv_db = conv_np * 80 - 80
            file_mcds = [compute_mcd(ra * 80 - 80, conv_db) for ra in real_pre_mels]
            mcd_b2a.append(min(file_mcds))

            src_db = src_np * 80 - 80
            content_pres_b2a.append(compute_mcd(src_db, conv_db))

            if not args.skip_f0:
                f0_c = compute_f0_corr(src_np, conv_np)
                f0_b2a.append(f0_c)

            print(f"   [{i+1}/{len(ds_post)}] MCD(target)={min(file_mcds):.2f}  MCD(content)={content_pres_b2a[-1]:.2f}")

    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  VQVAE Voice Conversion — Test Set Evaluation")
    print(f"{'='*60}")

    print(f"\n  Reconstruction:")
    print(f"    L1 loss: {np.mean(recon_losses):.4f} +/- {np.std(recon_losses):.4f}")

    print(f"\n  Disentanglement:")
    if len(np.unique(labels_np)) > 1:
        print(f"    Content domain acc: {content_acc:.1%} (ideal: ~50%)")
        print(f"    Quality domain acc: {quality_acc:.1%} (ideal: ~100%)")
    print(f"    Quality vector distance (pre vs post): {quality_distance:.4f}")

    print(f"\n  A->B (pre->post) — {len(mcd_a2b)} samples:")
    print(f"    MCD to target (lower=better):      {np.mean(mcd_a2b):.2f} +/- {np.std(mcd_a2b):.2f}")
    print(f"    Content MCD (lower=preserved):     {np.mean(content_pres_a2b):.2f} +/- {np.std(content_pres_a2b):.2f}")

    print(f"\n  B->A (post->pre) — {len(mcd_b2a)} samples:")
    print(f"    MCD to target (lower=better):      {np.mean(mcd_b2a):.2f} +/- {np.std(mcd_b2a):.2f}")
    print(f"    Content MCD (lower=preserved):     {np.mean(content_pres_b2a):.2f} +/- {np.std(content_pres_b2a):.2f}")

    if not args.skip_f0:
        valid_a2b = [f for f in f0_a2b if not np.isnan(f)]
        valid_b2a = [f for f in f0_b2a if not np.isnan(f)]
        print(f"\n  F0 Correlation (higher=better content preservation):")
        if valid_a2b:
            print(f"    A->B: {np.mean(valid_a2b):.3f} +/- {np.std(valid_a2b):.3f}  ({len(valid_a2b)}/{len(f0_a2b)} valid)")
        if valid_b2a:
            print(f"    B->A: {np.mean(valid_b2a):.3f} +/- {np.std(valid_b2a):.3f}  ({len(valid_b2a)}/{len(f0_b2a)} valid)")

    print(f"\n  Overall:")
    all_mcd = mcd_a2b + mcd_b2a
    all_content = content_pres_a2b + content_pres_b2a
    print(f"    Target MCD:  {np.mean(all_mcd):.2f} +/- {np.std(all_mcd):.2f}")
    print(f"    Content MCD: {np.mean(all_content):.2f} +/- {np.std(all_content):.2f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    evaluate()