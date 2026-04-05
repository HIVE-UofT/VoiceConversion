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


def extract_all_features(knn_vc, wav_dir):
    """Extract WavLM features from all WAV files. Returns list of (filename, features)."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    results = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)  # (T, 1024)
        results.append((Path(wf).stem, features.cpu()))
        print(f"  {Path(wf).name}: {features.shape[0]} frames")

    total = sum(f.shape[0] for _, f in results)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")
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
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints_kfold'))
    parser.add_argument('--n_test', type=int, default=5,
                        help='Number of patients held out for final test')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of CV folds on non-test patients')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    # Load WavLM
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Extract features
    print(f"\nExtracting pre-surgery features...")
    pre_data = extract_all_features(knn_vc, args.pre_dir)
    print(f"\nExtracting post-surgery features...")
    post_data = extract_all_features(knn_vc, args.post_dir)

    assert len(pre_data) == len(post_data)
    n_patients = len(pre_data)

    pre_names = [name for name, _ in pre_data]
    pre_features = [feat for _, feat in pre_data]
    post_features = [feat for _, feat in post_data]

    # ═══ Split: held-out test + CV pool ═══
    random.seed(args.seed)
    all_indices = list(range(n_patients))
    random.shuffle(all_indices)

    test_indices = sorted(all_indices[:args.n_test])
    cv_indices = sorted(all_indices[args.n_test:])

    test_names = [pre_names[i] for i in test_indices]
    cv_names = [pre_names[i] for i in cv_indices]

    print(f"\n{'='*60}")
    print(f"  Total patients: {n_patients}")
    print(f"  Held-out test ({args.n_test}): {test_names}")
    print(f"  CV pool ({len(cv_indices)}): {len(cv_indices)} patients")
    print(f"{'='*60}")

    # Save split info
    split_info = {
        'seed': args.seed,
        'n_test': args.n_test,
        'k_folds': args.k_folds,
        'test_patients': test_names,
        'cv_patients': cv_names,
    }
    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # ═══ K-Fold CV on cv_indices ═══
    random.shuffle(cv_indices)
    fold_size = len(cv_indices) // args.k_folds
    folds = []
    for k in range(args.k_folds):
        start = k * fold_size
        end = start + fold_size if k < args.k_folds - 1 else len(cv_indices)
        folds.append(cv_indices[start:end])

    cv_val_losses = []
    print(f"\n{'='*60}")
    print(f"  {args.k_folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    for k in range(args.k_folds):
        val_fold = folds[k]
        train_fold = [idx for j, fold in enumerate(folds) for idx in fold if j != k]

        print(f"\nFold {k+1}/{args.k_folds}: "
              f"train={len(train_fold)} patients, val={len(val_fold)} patients")

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
    print(f"\nTraining final model on all {len(cv_indices)} CV patients...")

    # Use last fold as val for early stopping (just for stopping criterion)
    final_train = [idx for fold in folds[:-1] for idx in fold]
    final_val = folds[-1]

    final_ckpt = os.path.join(args.output, 'best_model.pt')
    train_model(final_train, final_val, pre_features, post_features,
                device, final_ckpt, tag="Final")

    # Save test patient names for inference script
    pre_dir = args.pre_dir
    test_wav_files = [os.path.join(pre_dir, name + '.wav') for name in test_names]
    with open(os.path.join(args.output, 'test_files.json'), 'w') as f:
        json.dump({'test_patients': test_names, 'test_wav_files': test_wav_files}, f, indent=2)

    print(f"\nDone. Final model: {final_ckpt}")
    print(f"Test patients ({args.n_test}): {test_names}")
    print(f"Run inference on test patients only:")
    print(f"  python scripts/inference_kfold.py --checkpoint {final_ckpt}")


if __name__ == '__main__':
    main()
