"""
UNet-VC — K-Fold Cross-Validation with Held-Out Test Set

Evaluation protocol:
1. Hold out N_TEST patients entirely (never seen during any training)
2. On the remaining patients, run K-fold CV:
   - Each fold: train on K-1 folds, validate on 1 fold
   - Report average val loss across folds
3. Train a final model on ALL non-test patients (train+val)
4. Inference + evaluation on the held-out test patients only

This gives an honest estimate of generalization to unseen patients.

Usage:
    python scripts/train_kfold.py
    python scripts/train_kfold.py --n_test 5 --k_folds 5
"""

import argparse
import os
import sys
import glob
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D


SAMPLE_RATE = 16000

# Training config
HIDDEN_DIM = 128
N_LEVELS = 2
DROPOUT = 0.25
BATCH_SIZE = 32
SEGMENT_LEN = 64
SEGMENT_HOP = 16
LR = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 40
COSINE_LOSS_WEIGHT = 0.5
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1


def extract_all_features(knn_vc, wav_files):
    """Extract WavLM features from a list of WAV files. Returns list of (filename, features)."""
    if not wav_files:
        raise ValueError("Empty file list passed to extract_all_features")

    results = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)  # (T, 1024)
        results.append((Path(wf).stem, features.cpu()))

    total = sum(f.shape[0] for _, f in results)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")
    return results


def _build_audio_augmenter():
    """audiomentations chain: pitch shift + time stretch + gaussian noise + gain.
    Same chain object can be re-seeded per call to produce deterministic per-pair
    transformations (so pre and post in a pair get the SAME params)."""
    from audiomentations import (Compose, PitchShift, TimeStretch,
                                  AddGaussianNoise, Gain)
    return Compose([
        PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.7),
        TimeStretch(min_rate=0.92, max_rate=1.08, leave_length_unchanged=False, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
        Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.4),
    ])


def extract_features_audio_augmented(knn_vc, wav_paths, augmenter, seed_offset, device):
    """Augment each audio file deterministically (seeded by index + seed_offset),
    then run WavLM. Returning (stem, features) like extract_all_features.

    The seed is deterministic per (file_index, seed_offset), so calling this
    on the corresponding pre/post file lists in the same order with the same
    seed_offset yields IDENTICAL augmentation parameters for matched pairs --
    which preserves the surgery direction in feature space."""
    import librosa, soundfile as sf, tempfile
    SAMPLE_RATE = 16000
    results = []
    for i, wp in enumerate(wav_paths):
        per_seed = seed_offset * 100003 + i  # large prime separation
        random.seed(per_seed); np.random.seed(per_seed)
        wav, _ = librosa.load(wp, sr=SAMPLE_RATE)
        wav_aug = augmenter(samples=wav, sample_rate=SAMPLE_RATE)
        # Easiest cross-API path: write to temp wav, hand to knn_vc.get_features
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, wav_aug, SAMPLE_RATE)
            try:
                feats = knn_vc.get_features(tmp.name)
            finally:
                os.unlink(tmp.name)
        results.append((Path(wp).stem + f"_aug{seed_offset}", feats.cpu()))
    total = sum(f.shape[0] for _, f in results)
    print(f"  [aug{seed_offset}] Total: {total} frames ({total * 0.02 / 60:.1f} min)")
    return results


def pair_frames_knn(X, Y):
    """Pair source frames (X) to target frames (Y) via cosine NN."""
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)

    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        sim = X_norm[i:i + chunk_size] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))

    indices = torch.cat(all_indices)
    return X, Y[indices]


def build_segments(pre_features_list, post_features_list, patient_indices,
                   segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    """Build paired segments for a subset of patients (by index)."""
    segments = []
    for idx in patient_indices:
        pre_feat = pre_features_list[idx]
        post_feat = post_features_list[idx]
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_feat)
        n_frames = X_paired.shape[0]
        if n_frames < segment_len:
            continue
        for start in range(0, n_frames - segment_len + 1, segment_hop):
            end = start + segment_len
            segments.append((
                X_paired[start:end].t(),
                Y_paired[start:end].t(),
            ))
    return segments


class FeatureSegmentDataset(Dataset):
    def __init__(self, segments, augment=False, noise_std=0.02, mask_prob=0.1):
        self.segments = segments
        self.augment = augment
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y = self.segments[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
            mask = torch.rand(x.shape[-1]) > self.mask_prob
            x = x * mask.unsqueeze(0)
        return x, y


def combined_loss(y_pred, y_target, cosine_weight=0.5):
    mse = F.mse_loss(y_pred, y_target)
    cos_sim = F.cosine_similarity(y_pred, y_target, dim=1).mean()
    cosine_loss = 1.0 - cos_sim
    return mse + cosine_weight * cosine_loss, mse.item(), cosine_loss.item()


def train_model(train_indices, val_indices, pre_features, post_features,
                device, output_path, tag=""):
    """Train a single model on given patient splits. Returns best val loss."""

    train_segs = build_segments(pre_features, post_features, train_indices)
    val_segs = build_segments(pre_features, post_features, val_indices)

    train_dataset = FeatureSegmentDataset(train_segs, augment=True,
                                           noise_std=AUGMENT_NOISE_STD,
                                           mask_prob=AUGMENT_MASK_PROB)
    val_dataset = FeatureSegmentDataset(val_segs, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"  {tag} Train: {len(train_dataset)} segs ({len(train_indices)} patients), "
          f"Val: {len(val_dataset)} segs ({len(val_indices)} patients)")

    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss, _, _ = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss, _, _ = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        alpha_val = model.alpha.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"    {tag} Epoch {epoch:3d}  val={val_loss:.6f}  alpha={alpha_val:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'alpha': alpha_val,
                'config': {
                    'feat_dim': 1024,
                    'hidden_dim': HIDDEN_DIM,
                    'n_levels': N_LEVELS,
                    'dropout': DROPOUT,
                },
            }, output_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    {tag} Early stopping at epoch {epoch}")
                break

    print(f"    {tag} Best val loss: {best_val_loss:.6f}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="UNet-VC — K-Fold CV with held-out test set")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n_test', type=int, default=5,
                        help='Number of patients held out for final test (ignored if --test_patients set)')
    parser.add_argument('--test_patients', type=str,
                        default="0085,0110,0122,0132,0045",
                        help='Comma-separated fixed test patient IDs (overrides --n_test random selection)')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds on non-test patients')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extra_surgeries', action='store_true',
                        help='Also train on Fess+Sept+Contr data (added to train set in every fold)')
    parser.add_argument('--n_aug', type=int, default=0,
                        help='Number of audio-domain augmentation rounds applied to '
                             'CV training audio before WavLM extraction. Each round '
                             'uses pitch shift + time stretch + gain + Gaussian noise '
                             'with the SAME parameters applied to pre and post in a '
                             'pair (preserves surgery direction). 0 disables. '
                             'Test patients are NEVER augmented '
                             '(get_all_audio_pairs already excludes them).')
    args = parser.parse_args()

    if args.output is None:
        suffix = '_multisurg' if args.extra_surgeries else ''
        args.output = os.path.join(os.path.dirname(__file__), '..', f'checkpoints_kfold{suffix}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    # Load WavLM
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from utils import get_all_audio_pairs

    fixed_ids = set(p.strip() for p in args.test_patients.split(',') if p.strip()) \
                if args.test_patients else set()

    # Collect all audio types, excluding test patients
    patient_pairs = get_all_audio_pairs("Tonsill", exclude=fixed_ids)
    all_pids = sorted(patient_pairs.keys())

    # Flatten to parallel file lists
    pid_of_file = [pid for pid in all_pids for _ in patient_pairs[pid]]
    pre_file_list  = [pre  for pid in all_pids for pre,  _   in patient_pairs[pid]]
    post_file_list = [post for pid in all_pids for _,    post in patient_pairs[pid]]

    print(f"\nExtracting pre-surgery features ({len(pre_file_list)} files)...")
    pre_data = extract_all_features(knn_vc, pre_file_list)
    print(f"\nExtracting post-surgery features ({len(post_file_list)} files)...")
    post_data = extract_all_features(knn_vc, post_file_list)

    pre_features  = [feat for _, feat in pre_data]
    post_features = [feat for _, feat in post_data]
    n_cv_files = len(pre_features)  # number of Tonsill CV files

    # ── Audio-domain augmentation rounds (CV pool only; test patients
    #    were already filtered by get_all_audio_pairs(exclude=fixed_ids)).
    # Augmented features are appended to pre_features/post_features but
    # tracked in `aug_pid_of_file` and `aug_feature_indices`, NOT in
    # `pid_of_file`. They will be added only to train folds, never val,
    # so val stays a clean measurement of original CUCO data.
    aug_feature_indices = []
    aug_pid_of_file = []
    if args.n_aug > 0:
        print(f"\n=== Audio augmentation: {args.n_aug} extra rounds "
              f"on the {n_cv_files} CV file pairs (train-only) ===")
        augmenter = _build_audio_augmenter()
        for k in range(1, args.n_aug + 1):
            print(f"\n[Aug round {k}/{args.n_aug}] re-extracting pre features...")
            aug_pre  = extract_features_audio_augmented(
                knn_vc, pre_file_list,  augmenter, seed_offset=k, device=device)
            print(f"[Aug round {k}/{args.n_aug}] re-extracting post features...")
            aug_post = extract_features_audio_augmented(
                knn_vc, post_file_list, augmenter, seed_offset=k, device=device)
            n_before = len(pre_features)
            pre_features  += [f for _, f in aug_pre]
            post_features += [f for _, f in aug_post]
            aug_feature_indices += list(range(n_before, len(pre_features)))
            # The augmented copies' patient IDs (parallel to the new indices)
            aug_pid_of_file += list(pid_of_file[:n_cv_files])
        print(f"\n  After augmentation: {len(pre_features)} pre / "
              f"{len(post_features)} post feature vectors "
              f"({n_cv_files} originals + {len(aug_feature_indices)} augmented)")

    # Extra surgery data: extract features and add to training pool
    extra_indices = []
    if args.extra_surgeries:
        extra_surgeries = ["Fess", "Sept", "Contr"]
        print(f"\nExtracting features for extra surgery data ({extra_surgeries})...")
        for surg in extra_surgeries:
            surg_pairs = get_all_audio_pairs(surg)
            extra_pre_paths = [pre for pid in sorted(surg_pairs)
                               for pre, _ in surg_pairs[pid]]
            extra_post_paths = [post for pid in sorted(surg_pairs)
                                for _, post in surg_pairs[pid]]
            n_before = len(pre_features)
            print(f"  {surg}: {len(extra_pre_paths)} files")
            extra_pre_data  = extract_all_features(knn_vc, extra_pre_paths)
            extra_post_data = extract_all_features(knn_vc, extra_post_paths)
            pre_features  += [f for _, f in extra_pre_data]
            post_features += [f for _, f in extra_post_data]
            extra_indices += list(range(n_before, len(pre_features)))
        print(f"  Total extra features: {len(extra_indices)} files")

    n_files = len(pre_features)

    # ═══ Split: K-fold at PATIENT level, then map to file indices ═══
    random.seed(args.seed)

    cv_pids = all_pids.copy()  # already excludes test patients
    random.shuffle(cv_pids)

    print(f"\n{'='*60}")
    print(f"  Train/val patients: {len(cv_pids)}, files: {n_files}")
    print(f"  Held-out test: {sorted(fixed_ids)}")
    print(f"{'='*60}")

    # Map each file index to patient; cv_indices = file-level indices for all cv patients
    cv_indices = [i for i, pid in enumerate(pid_of_file) if pid in set(cv_pids)]

    # Save split info
    split_info = {
        'seed': args.seed,
        'n_test': len(fixed_ids),
        'k_folds': args.k_folds,
        'test_patients': sorted(fixed_ids),
        'cv_patients': cv_pids,
    }
    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # ═══ K-Fold CV — fold at PATIENT level ═══
    fold_size = len(cv_pids) // args.k_folds
    patient_folds = []
    for k in range(args.k_folds):
        start = k * fold_size
        end = start + fold_size if k < args.k_folds - 1 else len(cv_pids)
        patient_folds.append(set(cv_pids[start:end]))

    # Convert patient folds to file-level index folds
    folds = []
    for pf in patient_folds:
        folds.append([i for i, pid in enumerate(pid_of_file) if pid in pf])

    cv_val_losses = []
    print(f"\n{'='*60}")
    print(f"  {args.k_folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    for k in range(args.k_folds):
        val_fold = folds[k]
        train_fold = [idx for j, fold in enumerate(folds) for idx in fold if j != k]
        # Add extra surgery data to training (never to validation)
        train_fold = train_fold + extra_indices
        # Add augmented features whose source patient is a TRAIN patient in
        # this fold (never val): pick aug indices whose patient ID is in
        # the train-patient set for this fold.
        if aug_feature_indices:
            train_pids_this_fold = set(cv_pids) - patient_folds[k]
            aug_train_for_fold = [
                ai for ai, apid in zip(aug_feature_indices, aug_pid_of_file)
                if apid in train_pids_this_fold
            ]
            train_fold = train_fold + aug_train_for_fold
        else:
            aug_train_for_fold = []

        print(f"\nFold {k+1}/{args.k_folds}: "
              f"train={len(train_fold)} files "
              f"(incl. {len(extra_indices)} extra-surg, "
              f"{len(aug_train_for_fold)} augmented), "
              f"val={len(val_fold)} files")

        ckpt_path = os.path.join(args.output, f'fold{k+1}_model.pt')
        val_loss = train_model(train_fold, val_fold, pre_features, post_features,
                               device, ckpt_path, tag=f"Fold {k+1}")
        cv_val_losses.append(val_loss)

    mean_cv = np.mean(cv_val_losses)
    std_cv = np.std(cv_val_losses)
    print(f"\n{'='*60}")
    print(f"  CV Results: {mean_cv:.6f} +/- {std_cv:.6f}")
    print(f"  Per-fold: {[f'{v:.6f}' for v in cv_val_losses]}")
    print(f"{'='*60}")

    # ═══ Final model: train on ALL cv_indices, validate on a small held-out from cv ═══
    print(f"\nTraining final model on all {len(cv_indices)} CV patients + {len(extra_indices)} extra...")

    # Use last fold as val for early stopping (just for stopping criterion)
    final_train = [idx for fold in folds[:-1] for idx in fold] + extra_indices
    if aug_feature_indices:
        final_train_pids = set(cv_pids) - patient_folds[-1]
        final_train += [
            ai for ai, apid in zip(aug_feature_indices, aug_pid_of_file)
            if apid in final_train_pids
        ]
    final_val = folds[-1]

    final_ckpt = os.path.join(args.output, 'best_model.pt')
    train_model(final_train, final_val, pre_features, post_features,
                device, final_ckpt, tag="Final")

    # Save test patient IDs for inference script
    with open(os.path.join(args.output, 'test_files.json'), 'w') as f:
        json.dump({'test_patients': sorted(fixed_ids)}, f, indent=2)

    print(f"\nDone. Final model: {final_ckpt}")
    print(f"Test patients ({len(fixed_ids)}): {sorted(fixed_ids)}")
    print(f"Run inference on test patients only:")
    print(f"  python scripts/inference_kfold.py --checkpoint {final_ckpt}")


if __name__ == '__main__':
    main()
