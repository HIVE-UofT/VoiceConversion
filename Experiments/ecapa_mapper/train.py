"""
Train and evaluate EcapaMapper: pre -> post surgery.

Feature extractor: librosa acoustic features (NOT ECAPA speaker embeddings).
ECAPA was trained to be invariant to exactly what surgery changes; these
features instead capture what matters for tonsillectomy:

  - MFCCs (C0–C19):     spectral envelope = vocal tract shape (40 dims)
  - Delta-MFCCs:         rate-of-change dynamics              (40 dims)
  - Spectral contrast:   harmonics-vs-noise per sub-band ≈ HNR (14 dims)
  - Spectral centroid/bandwidth/rolloff/flatness/ZCR/RMS:    (12 dims)

All features are temporal means + stds, which removes linguistic content
while preserving voice quality characteristics.

Target: Session 2 (immediate post-surgery).

Usage:
    python train.py
    python train.py --model_type linear
    python train.py --n_crops 5
"""

import argparse
import copy
import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from model import EcapaMapper, LinearMapper

SAMPLE_RATE = 16000
CUCO_BASE   = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"

LR           = 1e-4
WEIGHT_DECAY = 0.1
EPOCHS       = 500
PATIENCE     = 60
HIDDEN_DIM   = 128
N_BLOCKS     = 1
DROPOUT      = 0.5
NOISE_SCALE  = 0.02
N_AUG        = 5
MIXUP_ALPHA  = 0.2
K_FOLDS      = 5
N_CROPS      = 10
CROP_SEC     = 3.0


# ─────────────────────────────────────────────────────────────
# Feature extraction  (librosa, content-free)
# ─────────────────────────────────────────────────────────────

def _crop_features(audio: np.ndarray, sr: int) -> torch.Tensor:
    """
    Compute a content-free voice-characteristic feature vector from one crop.

    Features (all computed as temporal mean + std → removes linguistic content):
      MFCCs C0-C19          : 20 × 2 =  40 dims   (spectral envelope / vocal tract)
      Delta-MFCCs C0-C19    : 20 × 2 =  40 dims   (temporal dynamics)
      Spectral contrast      :  7 × 2 =  14 dims   (harmonics vs. noise ≈ HNR)
      Spectral centroid      :  1 × 2 =   2 dims
      Spectral bandwidth     :  1 × 2 =   2 dims
      Spectral rolloff       :  1 × 2 =   2 dims
      Spectral flatness      :  1 × 2 =   2 dims
      Zero-crossing rate     :  1 × 2 =   2 dims
      RMS energy             :  1 × 2 =   2 dims
                                       ─────────
                                         106 dims
    """
    feats = []

    # MFCC + delta
    mfcc  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    dmfcc = librosa.feature.delta(mfcc)
    for mat in (mfcc, dmfcc):
        feats.extend(mat.mean(axis=1).tolist())
        feats.extend(mat.std(axis=1).tolist())

    # Spectral contrast (7 bands): voice quality / harmonicity
    sc = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    feats.extend(sc.mean(axis=1).tolist())
    feats.extend(sc.std(axis=1).tolist())

    # Aggregate spectral statistics
    for arr in (
        librosa.feature.spectral_centroid(y=audio, sr=sr)[0],
        librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0],
        librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0],
        librosa.feature.spectral_flatness(y=audio)[0],
        librosa.feature.zero_crossing_rate(audio)[0],
        librosa.feature.rms(y=audio)[0],
    ):
        feats.extend([float(arr.mean()), float(arr.std())])

    return torch.tensor(feats, dtype=torch.float32)   # (106,)


def extract_features_multicrop(wav_path, n_crops, crop_sec):
    """
    Load audio, split into n_crops evenly-spaced windows, return a list
    of (106,) feature tensors.
    """
    sig, sr = torchaudio.load(wav_path)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        sig = torchaudio.functional.resample(sig, sr, SAMPLE_RATE)

    audio       = sig.squeeze().numpy()
    total       = len(audio)
    crop_frames = int(crop_sec * SAMPLE_RATE)

    if n_crops == 1 or total <= crop_frames:
        return [_crop_features(audio, SAMPLE_RATE)]

    starts = np.linspace(0, total - crop_frames, n_crops).astype(int)
    return [_crop_features(audio[s : s + crop_frames], SAMPLE_RATE)
            for s in starts]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def mixup(X, Y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], lam * Y + (1 - lam) * Y[idx]


def make_model(model_type, feat_dim):
    if model_type == 'linear':
        return LinearMapper(emb_dim=feat_dim)
    return EcapaMapper(emb_dim=feat_dim, hidden_dim=HIDDEN_DIM,
                       n_blocks=N_BLOCKS, dropout=DROPOUT)


def kfold_split(n, k, seed):
    shuffled = np.random.default_rng(seed).permutation(n).tolist()
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else n
        val   = sorted(shuffled[start:end])
        tr    = sorted(shuffled[:start] + shuffled[end:])
        folds.append((tr, val))
    return folds


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_one_fold(X_tr, Y_tr, X_val, Y_val, model_type, feat_dim,
                   fold_i, ckpt_path):
    model     = make_model(model_type, feat_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_val_loss    = float('inf')
    patience_counter = 0
    best_state       = copy.deepcopy(model.state_dict())
    best_epoch       = 1

    for epoch in range(1, EPOCHS + 1):
        model.train()

        X_parts, Y_parts = [X_tr], [Y_tr]
        for _ in range(N_AUG):
            X_parts.append(X_tr + torch.randn_like(X_tr) * NOISE_SCALE)
            Y_parts.append(Y_tr)
        if MIXUP_ALPHA > 0:
            Xm, Ym = mixup(X_tr, Y_tr, MIXUP_ALPHA)
            X_parts.append(Xm)
            Y_parts.append(Ym)

        pred = model(torch.cat(X_parts))
        loss = (1.0 - F.cosine_similarity(pred, torch.cat(Y_parts))).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = (1.0 - F.cosine_similarity(
                model(X_val), Y_val)).mean().item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            best_epoch       = epoch
            best_state       = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Fold {fold_i}: early stop epoch {epoch} "
                      f"(best={best_epoch})")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    return model, best_epoch


def train_final(X_cv, Y_cv, model_type, feat_dim, n_epochs, ckpt_path):
    model     = make_model(model_type, feat_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        model.train()
        X_parts, Y_parts = [X_cv], [Y_cv]
        for _ in range(N_AUG):
            X_parts.append(X_cv + torch.randn_like(X_cv) * NOISE_SCALE)
            Y_parts.append(Y_cv)
        if MIXUP_ALPHA > 0:
            Xm, Ym = mixup(X_cv, Y_cv, MIXUP_ALPHA)
            X_parts.append(Xm)
            Y_parts.append(Ym)

        pred = model(torch.cat(X_parts))
        loss = (1.0 - F.cosine_similarity(pred, torch.cat(Y_parts))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    return model


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery',    type=str, default='Tonsill')
    parser.add_argument('--n_test',     type=int, default=5)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--output',     type=str, default=None)
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'linear'])
    parser.add_argument('--k_folds',    type=int, default=K_FOLDS)
    parser.add_argument('--n_crops',    type=int, default=N_CROPS)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), f'results_{args.surgery.lower()}')
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Acoustic Feature Mapper ({args.model_type}): "
          f"pre (ses1) -> post (ses2)")
    print(f"  Features: librosa MFCC + spectral (content-free)")
    print(f"{'='*60}")

    # ── Patient split ──
    pre_dir  = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files  = sorted(glob.glob(os.path.join(pre_dir,  "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n     = len(pre_files)
    names = [Path(f).stem for f in pre_files]

    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    test_idx = sorted(indices[:args.n_test])
    cv_idx   = sorted(indices[args.n_test:])

    print(f"\n{args.surgery}: {n} patients  |  "
          f"n_crops={args.n_crops}  crop_sec={CROP_SEC}")
    print(f"  CV pool ({len(cv_idx)}): {[names[i] for i in cv_idx]}")
    print(f"  Test   ({len(test_idx)}): {[names[i] for i in test_idx]}")

    # ── Extract features ──
    def load_features(files, label):
        print(f"\nExtracting {label} features ({args.n_crops} crops × {CROP_SEC}s)...")
        crops, means = {}, {}
        for i, f in enumerate(files):
            c = extract_features_multicrop(f, args.n_crops, CROP_SEC)
            crops[i] = c
            means[i] = torch.stack(c).mean(0)
            print(f"  {Path(f).stem}: {len(c)} crops  feat_dim={c[0].shape[0]}")
        return crops, means

    pre_crops,  mean_pre  = load_features(pre_files,  "pre (session 1)")
    post_crops, mean_post = load_features(post_files, "post (session 2)")

    feat_dim = mean_pre[0].shape[0]
    print(f"\nFeature dimension: {feat_dim}")

    # ── Feature normalisation: z-score using CV pool statistics ──
    # Fit on CV only (no test leakage)
    cv_feats = torch.stack([mean_pre[i]  for i in cv_idx] +
                           [mean_post[i] for i in cv_idx])
    feat_mean = cv_feats.mean(0)
    feat_std  = cv_feats.std(0).clamp(min=1e-8)

    def normalise(t):
        return (t - feat_mean) / feat_std

    # Apply normalisation in-place to all crops + means
    for i in range(n):
        pre_crops[i]  = [normalise(c) for c in pre_crops[i]]
        post_crops[i] = [normalise(c) for c in post_crops[i]]
        mean_pre[i]   = normalise(mean_pre[i])
        mean_post[i]  = normalise(mean_post[i])

    # ── Baseline ──
    baseline_cv   = [cosine_sim(mean_pre[i], mean_post[i]) for i in cv_idx]
    baseline_test = [cosine_sim(mean_pre[i], mean_post[i]) for i in test_idx]
    print(f"\nBaseline cosine sim (no transform, normalised mean features):")
    print(f"  CV pool: {np.mean(baseline_cv):.4f} ± {np.std(baseline_cv):.4f}")
    print(f"  Test:    {np.mean(baseline_test):.4f} ± {np.std(baseline_test):.4f}")

    X_cv_mean = torch.stack([mean_pre[i]  for i in cv_idx])
    Y_cv_mean = torch.stack([mean_post[i] for i in cv_idx])
    X_test    = torch.stack([mean_pre[i]  for i in test_idx])
    Y_test    = torch.stack([mean_post[i] for i in test_idx])

    # Mean shift
    delta_mean = (Y_cv_mean - X_cv_mean).mean(0)
    ms_test    = [cosine_sim(X_test[j] + delta_mean, Y_test[j])
                  for j in range(len(test_idx))]
    print(f"Mean shift test sim: {np.mean(ms_test):.4f}  "
          f"delta={np.mean(ms_test) - np.mean(baseline_test):+.4f}")

    # ── Build training pairs (pre crop → mean post) ──
    def build_pairs(patient_indices):
        X_list, Y_list = [], []
        for i in patient_indices:
            for pre_e in pre_crops[i]:
                X_list.append(pre_e)
                Y_list.append(mean_post[i])
        return torch.stack(X_list), torch.stack(Y_list)

    X_cv_all, Y_cv_all = build_pairs(cv_idx)

    # ── Model info ──
    n_params = make_model(args.model_type, feat_dim).count_parameters()
    print(f"\nK-fold CV ({args.k_folds} folds)  model={args.model_type}  "
          f"feat_dim={feat_dim}  hidden={HIDDEN_DIM}  blocks={N_BLOCKS}  "
          f"dropout={DROPOUT}  LR={LR}  wd={WEIGHT_DECAY}  "
          f"n_aug={N_AUG}  mixup={MIXUP_ALPHA}  noise={NOISE_SCALE}")
    print(f"Model parameters: {n_params:,}\n")
    print("Training folds...")

    folds         = kfold_split(len(cv_idx), args.k_folds, args.seed)
    fold_models   = []
    fold_epochs   = []
    fold_val_sims = []

    for fold_i, (tr_local, val_local) in enumerate(folds):
        tr_global  = [cv_idx[j] for j in tr_local]
        val_global = [cv_idx[j] for j in val_local]

        X_tr, Y_tr = build_pairs(tr_global)
        X_val = torch.stack([mean_pre[i]  for i in val_global])
        Y_val = torch.stack([mean_post[i] for i in val_global])

        ckpt  = os.path.join(args.output, f'fold_{fold_i}_mapper.pt')
        model, best_ep = train_one_fold(
            X_tr, Y_tr, X_val, Y_val, args.model_type, feat_dim, fold_i, ckpt)
        model.eval()

        with torch.no_grad():
            val_sim = F.cosine_similarity(model(X_val), Y_val).mean().item()
        val_base = np.mean([cosine_sim(mean_pre[i], mean_post[i])
                            for i in val_global])
        fold_val_sims.append(val_sim)
        fold_epochs.append(best_ep)
        print(f"  Fold {fold_i}: val_base={val_base:.4f}  val_sim={val_sim:.4f}  "
              f"delta={val_sim - val_base:+.4f}  best_epoch={best_ep}  "
              f"({len(tr_global)*args.n_crops} train / {len(val_global)} val)")
        fold_models.append(model)

    median_ep = int(np.median(fold_epochs))
    final_ep  = max(median_ep, 100)
    print(f"\n  CV mean val sim: {np.mean(fold_val_sims):.4f} "
          f"± {np.std(fold_val_sims):.4f}")
    print(f"  Fold best epochs: {fold_epochs}  → "
          f"median={median_ep}  final_model_epochs={final_ep}")

    # ── Final model ──
    print(f"\nTraining final model on {len(X_cv_all)} pairs for {final_ep} epochs...")
    ckpt_final  = os.path.join(args.output, 'final_mapper.pt')
    final_model = train_final(X_cv_all, Y_cv_all, args.model_type,
                              feat_dim, final_ep, ckpt_final)
    final_model.eval()

    # ── Ensemble ──
    all_models = fold_models + [final_model]
    with torch.no_grad():
        ensemble_pred = torch.stack([m(X_test) for m in all_models]).mean(0)

    # ── Test results ──
    print(f"\n{'='*60}")
    print(f"  Test Results  ({len(all_models)}-model ensemble)")
    print(f"{'='*60}\n--- TEST ---")

    test_sims = []
    for j, i in enumerate(test_idx):
        sim_ens  = cosine_sim(ensemble_pred[j], Y_test[j])
        sim_base = cosine_sim(X_test[j], Y_test[j])
        test_sims.append(sim_ens)
        print(f"  {names[i]:40s}  "
              f"baseline={sim_base:.4f}  "
              f"mean_shift={ms_test[j]:.4f}  "
              f"ensemble={sim_ens:.4f}  "
              f"delta={sim_ens - sim_base:+.4f}")

    mean_ens  = np.mean(test_sims)
    mean_base = np.mean(baseline_test)
    mean_ms   = np.mean(ms_test)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:   {mean_base:.4f}")
    print(f"  Mean shift: {mean_ms:.4f}  delta={mean_ms  - mean_base:+.4f}")
    print(f"  Ensemble:   {mean_ens:.4f}  delta={mean_ens - mean_base:+.4f}")
    print(f"  (CV val sim: {np.mean(fold_val_sims):.4f} "
          f"± {np.std(fold_val_sims):.4f})")
    print(f"{'='*60}")

    results = {
        'surgery':    args.surgery,
        'features':   'librosa_mfcc_spectral_106dim',
        'model_type': args.model_type,
        'target_session': '2',
        'k_folds':    args.k_folds,
        'n_crops':    args.n_crops,
        'feat_dim':   feat_dim,
        'hyperparams': {
            'lr': LR, 'hidden_dim': HIDDEN_DIM, 'n_blocks': N_BLOCKS,
            'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY,
            'noise_scale': NOISE_SCALE, 'n_aug': N_AUG,
            'mixup_alpha': MIXUP_ALPHA, 'crop_sec': CROP_SEC,
        },
        'split': {
            'cv_pool': [names[i] for i in cv_idx],
            'test':    [names[i] for i in test_idx],
        },
        'cv_val_sim': {
            'mean': float(np.mean(fold_val_sims)),
            'std':  float(np.std(fold_val_sims)),
        },
        'fold_best_epochs':   fold_epochs,
        'final_model_epochs': final_ep,
        'summary': {
            'baseline':   float(mean_base),
            'mean_shift': float(mean_ms),
            'ensemble':   float(mean_ens),
            'delta':      float(mean_ens - mean_base),
        },
        'per_patient_test': [
            {
                'name':       names[test_idx[j]],
                'baseline':   float(baseline_test[j]),
                'mean_shift': float(ms_test[j]),
                'ensemble':   float(cosine_sim(ensemble_pred[j], Y_test[j])),
            }
            for j in range(len(test_idx))
        ],
    }
    out_json = os.path.join(args.output, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")
    print("Done!")


if __name__ == '__main__':
    main()
