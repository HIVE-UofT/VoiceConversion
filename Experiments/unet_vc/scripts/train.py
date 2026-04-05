"""
UNet-VC — Nonlinear Feature-Space Voice Conversion (Training)

Trains a lightweight residual 1D U-Net to learn a nonlinear transform
in WavLM feature space: f(x_pre) ≈ x_post.

Training procedure:
1. Extract WavLM features from all pre and post surgery audio
2. Pair frames per-utterance via cosine-similarity nearest neighbors
3. Create overlapping windows (segments) per-utterance for temporal context
4. Train U-Net with MSE + cosine similarity loss on features

The residual design (output = input + alpha * network(input)) means the
network only needs to learn the small delta between domains.

Usage:
    python scripts/train.py
"""

import argparse
import os
import sys
import glob
import random
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
SEGMENT_LEN = 64       # frames per training segment (~1.3s at 50fps)
SEGMENT_HOP = 16        # smaller hop = more training samples from limited data
LR = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 40           # early stopping patience
COSINE_LOSS_WEIGHT = 0.5
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1


def extract_all_features(knn_vc, wav_dir):
    """Extract WavLM features from all WAV files. Returns list of per-file tensors."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    all_features = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)  # (T, 1024)
        all_features.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")

    total = sum(f.shape[0] for f in all_features)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")
    return all_features


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


def create_segments_per_utterance(pre_features_list, post_features_list,
                                   segment_len, segment_hop):
    """Create paired segments per-utterance, preserving temporal order."""
    segments = []

    for pre_feat, post_feat in zip(pre_features_list, post_features_list):
        # Pair frames within this utterance
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_feat)

        n_frames = X_paired.shape[0]
        if n_frames < segment_len:
            continue

        for start in range(0, n_frames - segment_len + 1, segment_hop):
            end = start + segment_len
            segments.append((
                X_paired[start:end].t(),   # (1024, seg_len)
                Y_paired[start:end].t(),   # (1024, seg_len)
            ))

    print(f"  Created {len(segments)} segments (len={segment_len}, hop={segment_hop})")
    return segments


class FeatureSegmentDataset(Dataset):
    """Dataset of paired (source, target) feature segments with augmentation."""

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
            # Gaussian noise on input features
            x = x + torch.randn_like(x) * self.noise_std
            # Random time-frame masking (zero out random frames)
            mask = torch.rand(x.shape[-1]) > self.mask_prob
            x = x * mask.unsqueeze(0)

        return x, y


def combined_loss(y_pred, y_target, cosine_weight=0.5):
    """MSE + cosine similarity loss."""
    mse = F.mse_loss(y_pred, y_target)

    # Cosine similarity over the feature dimension (dim=1), averaged over batch and time
    cos_sim = F.cosine_similarity(y_pred, y_target, dim=1).mean()
    cosine_loss = 1.0 - cos_sim

    return mse + cosine_weight * cosine_loss, mse.item(), cosine_loss.item()


def main():
    parser = argparse.ArgumentParser(description="UNet-VC — Train nonlinear feature transform")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load WavLM
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Extract features per-utterance (keep separate for per-utterance pairing)
    print(f"\nExtracting pre-surgery features...")
    features_pre_list = extract_all_features(knn_vc, args.pre_dir)

    print(f"\nExtracting post-surgery features...")
    features_post_list = extract_all_features(knn_vc, args.post_dir)

    assert len(features_pre_list) == len(features_post_list), \
        f"Mismatch: {len(features_pre_list)} pre files vs {len(features_post_list)} post files"

    # Create segments per-utterance (preserving temporal coherence)
    print(f"\nCreating paired segments per-utterance...")
    all_segments = create_segments_per_utterance(
        features_pre_list, features_post_list, SEGMENT_LEN, SEGMENT_HOP
    )

    # Split train/val by utterance index to avoid data leakage
    n_utts = len(features_pre_list)
    utt_indices = list(range(n_utts))
    random.shuffle(utt_indices)
    n_val_utts = max(1, int(0.15 * n_utts))  # ~15% utterances for val
    val_utt_set = set(utt_indices[:n_val_utts])
    train_utt_set = set(utt_indices[n_val_utts:])

    print(f"  Train utterances: {len(train_utt_set)}, Val utterances: {len(val_utt_set)}")

    # Rebuild segments split by utterance
    train_segments = []
    val_segments = []
    seg_idx = 0
    for utt_i, (pre_feat, post_feat) in enumerate(zip(features_pre_list, features_post_list)):
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_feat)
        n_frames = X_paired.shape[0]
        if n_frames < SEGMENT_LEN:
            continue
        for start in range(0, n_frames - SEGMENT_LEN + 1, SEGMENT_HOP):
            end = start + SEGMENT_LEN
            seg = (X_paired[start:end].t(), Y_paired[start:end].t())
            if utt_i in val_utt_set:
                val_segments.append(seg)
            else:
                train_segments.append(seg)

    train_dataset = FeatureSegmentDataset(train_segments, augment=True,
                                           noise_std=AUGMENT_NOISE_STD,
                                           mask_prob=AUGMENT_MASK_PROB)
    val_dataset = FeatureSegmentDataset(val_segments, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(train_dataset)} segments, Val: {len(val_dataset)} segments")

    # Model
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)  # (B, 1024, T)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss, mse_val, cos_val = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(x_batch)
                loss, _, _ = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        alpha_val = model.alpha.item()
        lr_now = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"alpha={alpha_val:.4f}  lr={lr_now:.2e}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'alpha': alpha_val,
                'config': {
                    'feat_dim': 1024,
                    'hidden_dim': HIDDEN_DIM,
                    'n_levels': N_LEVELS,
                    'dropout': DROPOUT,
                },
            }, os.path.join(args.output, 'best_model.pt'))
            print(f"  -> Saved best model (val={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to {os.path.join(args.output, 'best_model.pt')}")


if __name__ == '__main__':
    main()
