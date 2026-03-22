"""
UNet-VC — Nonlinear Feature-Space Voice Conversion (Training)

Trains a lightweight residual 1D U-Net to learn a nonlinear transform
in WavLM feature space: f(x_pre) ≈ x_post.

Training procedure:
1. Extract WavLM features from all pre and post surgery audio
2. Pair frames via cosine-similarity nearest neighbors
3. Create overlapping windows (segments) for temporal context
4. Train U-Net with MSE loss + multi-res spectral loss on features

The residual design (output = input + alpha * network(input)) means the
network only needs to learn the small delta between domains.

Usage:
    python scripts/train.py
"""

import argparse
import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D


SAMPLE_RATE = 16000

# Training config
HIDDEN_DIM = 256
N_LEVELS = 3
DROPOUT = 0.1
BATCH_SIZE = 32
SEGMENT_LEN = 64       # frames per training segment (~1.3s at 50fps)
SEGMENT_HOP = 32       # overlap for more training samples
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 30           # early stopping patience


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
    print(f"  Pairing {X.shape[0]} -> {Y.shape[0]} frames...")

    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)

    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        sim = X_norm[i:i + chunk_size] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))

    indices = torch.cat(all_indices)
    return X, Y[indices]


class FeatureSegmentDataset(Dataset):
    """Dataset of paired (source, target) feature segments."""

    def __init__(self, X_paired, Y_paired, segment_len=64, segment_hop=32):
        self.segments = []
        n_frames = X_paired.shape[0]
        for start in range(0, n_frames - segment_len + 1, segment_hop):
            end = start + segment_len
            self.segments.append((
                X_paired[start:end].t(),   # (1024, seg_len)
                Y_paired[start:end].t(),   # (1024, seg_len)
            ))
        print(f"  Created {len(self.segments)} segments (len={segment_len}, hop={segment_hop})")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


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

    # Extract features
    print(f"\nExtracting pre-surgery features...")
    features_pre_list = extract_all_features(knn_vc, args.pre_dir)
    features_pre = torch.cat(features_pre_list, dim=0)

    print(f"\nExtracting post-surgery features...")
    features_post_list = extract_all_features(knn_vc, args.post_dir)
    features_post = torch.cat(features_post_list, dim=0)

    # Pair frames
    print(f"\nPairing frames (pre -> post)...")
    X_paired, Y_paired = pair_frames_knn(features_pre, features_post)

    # Create train/val split (90/10 by frames)
    n = X_paired.shape[0]
    n_train = int(0.9 * n)

    # Shuffle paired frames before splitting (but keep pairs together)
    perm = torch.randperm(n)
    X_paired = X_paired[perm]
    Y_paired = Y_paired[perm]

    train_dataset = FeatureSegmentDataset(X_paired[:n_train], Y_paired[:n_train],
                                          SEGMENT_LEN, SEGMENT_HOP)
    val_dataset = FeatureSegmentDataset(X_paired[n_train:], Y_paired[n_train:],
                                        SEGMENT_LEN, SEGMENT_HOP)

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

    # Loss: MSE on features
    mse_loss = nn.MSELoss()

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
            loss = mse_loss(y_pred, y_batch)

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
                loss = mse_loss(y_pred, y_batch)
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
