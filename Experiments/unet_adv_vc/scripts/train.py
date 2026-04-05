"""
UNet-Adv-VC — Adversarial Residual U-Net Voice Conversion (Training)

Extends UNet-VC with a PatchGAN discriminator that pushes converted features
closer to the real post-surgery distribution. The generator (U-Net) is trained
with MSE + cosine + adversarial loss; the discriminator distinguishes real
post-surgery features from converted features.

Phase 1 (warmup): Train generator with MSE+cosine only (same as UNet-VC)
Phase 2: Add adversarial loss with the discriminator

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
from model.discriminator import PatchDiscriminator1D


SAMPLE_RATE = 16000

# Training config
HIDDEN_DIM = 128
N_LEVELS = 2
DROPOUT = 0.25
BATCH_SIZE = 32
SEGMENT_LEN = 64
SEGMENT_HOP = 16
LR_G = 5e-4
LR_D = 2e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 400
PATIENCE = 50
COSINE_LOSS_WEIGHT = 0.5
LAMBDA_ADV = 0.1           # adversarial loss weight for generator
WARMUP_EPOCHS = 50          # train without adversarial loss first
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1
D_UPDATES_PER_G = 1        # discriminator steps per generator step


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
            x = x + torch.randn_like(x) * self.noise_std
            mask = torch.rand(x.shape[-1]) > self.mask_prob
            x = x * mask.unsqueeze(0)

        return x, y


class UnpairedPostDataset(Dataset):
    """Dataset of unpaired post-surgery segments for the discriminator."""

    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


def recon_loss(y_pred, y_target, cosine_weight=0.5):
    """MSE + cosine similarity loss."""
    mse = F.mse_loss(y_pred, y_target)
    cos_sim = F.cosine_similarity(y_pred, y_target, dim=1).mean()
    cosine_loss = 1.0 - cos_sim
    return mse + cosine_weight * cosine_loss, mse.item(), cosine_loss.item()


def main():
    parser = argparse.ArgumentParser(description="UNet-Adv-VC — Train with adversarial loss")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'))
    parser.add_argument('--epochs', type=int, default=EPOCHS)
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
    print(f"\nExtracting post-surgery features...")
    features_post_list = extract_all_features(knn_vc, args.post_dir)

    assert len(features_pre_list) == len(features_post_list), \
        f"Mismatch: {len(features_pre_list)} pre files vs {len(features_post_list)} post files"

    # Split train/val by utterance
    n_utts = len(features_pre_list)
    utt_indices = list(range(n_utts))
    random.shuffle(utt_indices)
    n_val_utts = max(1, int(0.15 * n_utts))
    val_utt_set = set(utt_indices[:n_val_utts])

    # Create paired segments and unpaired post segments
    train_paired = []
    val_paired = []
    train_post_unpaired = []

    for utt_i, (pre_feat, post_feat) in enumerate(zip(features_pre_list, features_post_list)):
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_feat)
        n_frames = X_paired.shape[0]
        if n_frames < SEGMENT_LEN:
            continue

        for start in range(0, n_frames - SEGMENT_LEN + 1, SEGMENT_HOP):
            end = start + SEGMENT_LEN
            seg = (X_paired[start:end].t(), Y_paired[start:end].t())
            if utt_i in val_utt_set:
                val_paired.append(seg)
            else:
                train_paired.append(seg)

        # Also create unpaired post segments for discriminator
        if utt_i not in val_utt_set:
            n_post = post_feat.shape[0]
            for start in range(0, n_post - SEGMENT_LEN + 1, SEGMENT_HOP):
                end = start + SEGMENT_LEN
                train_post_unpaired.append(post_feat[start:end].t())  # (1024, T)

    train_dataset = FeatureSegmentDataset(train_paired, augment=True,
                                           noise_std=AUGMENT_NOISE_STD,
                                           mask_prob=AUGMENT_MASK_PROB)
    val_dataset = FeatureSegmentDataset(val_paired, augment=False)
    post_dataset = UnpairedPostDataset(train_post_unpaired)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    post_loader = DataLoader(post_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, drop_last=True)

    print(f"\nTrain paired: {len(train_dataset)}, Val: {len(val_dataset)}, "
          f"Post unpaired: {len(post_dataset)}")

    # Models
    generator = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                          dropout=DROPOUT).to(device)
    discriminator = PatchDiscriminator1D(feat_dim=1024, hidden_dim=256, n_layers=3).to(device)

    print(f"Generator parameters: {generator.count_parameters():,}")
    print(f"Discriminator parameters: {discriminator.count_parameters():,}")

    # Optimizers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=LR_G, weight_decay=WEIGHT_DECAY)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, weight_decay=WEIGHT_DECAY)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    # Training
    os.makedirs(args.output, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        use_adv = epoch > WARMUP_EPOCHS
        post_iter = iter(post_loader) if use_adv else None

        ep_recon = 0.0
        ep_g_adv = 0.0
        ep_d_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B = x_batch.shape[0]

            # ═══ Generator step ═══
            y_pred = generator(x_batch)
            loss_rec, mse_val, cos_val = recon_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
            loss_g = loss_rec

            loss_g_adv_val = 0.0
            loss_d_val = 0.0

            if use_adv:
                # Generator adversarial loss: fool discriminator
                fake_scores = discriminator(y_pred)
                loss_g_adv = F.binary_cross_entropy_with_logits(
                    fake_scores, torch.ones_like(fake_scores))
                loss_g = loss_rec + LAMBDA_ADV * loss_g_adv
                loss_g_adv_val = loss_g_adv.item()

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            # ═══ Discriminator step ═══
            if use_adv:
                for _ in range(D_UPDATES_PER_G):
                    # Get real post-surgery batch
                    try:
                        real_post = next(post_iter).to(device)
                    except StopIteration:
                        post_iter = iter(post_loader)
                        real_post = next(post_iter).to(device)

                    # Fake: converted features (detached from generator)
                    with torch.no_grad():
                        fake_post = generator(x_batch)

                    real_scores = discriminator(real_post)
                    fake_scores = discriminator(fake_post)

                    loss_d_real = F.binary_cross_entropy_with_logits(
                        real_scores, torch.ones_like(real_scores))
                    loss_d_fake = F.binary_cross_entropy_with_logits(
                        fake_scores, torch.zeros_like(fake_scores))
                    loss_d = (loss_d_real + loss_d_fake) * 0.5

                    opt_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    opt_d.step()

                    loss_d_val = loss_d.item()

            ep_recon += loss_rec.item()
            ep_g_adv += loss_g_adv_val
            ep_d_loss += loss_d_val
            n_batches += 1

        sched_g.step()
        if use_adv:
            sched_d.step()

        ep_recon /= n_batches
        ep_g_adv /= n_batches
        ep_d_loss /= n_batches

        # Validate (recon loss only)
        generator.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = generator(x_batch)
                loss, _, _ = recon_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        alpha_val = generator.alpha.item()

        phase = "[WARMUP]" if not use_adv else ""
        print(f"Epoch {epoch:3d}/{args.epochs} {phase} "
              f"recon={ep_recon:.6f}  g_adv={ep_g_adv:.4f}  d_loss={ep_d_loss:.4f}  "
              f"val={val_loss:.6f}  alpha={alpha_val:.4f}")

        # Save best (based on val recon loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
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


if __name__ == '__main__':
    main()
