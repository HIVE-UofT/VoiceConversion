"""
UNet-Adv-VC — Training with Proper Patient-Level Split

Replaces the utterance-level shuffle in train.py with a patient-level split
that excludes a fixed test set from all training data.

Phase 1 (warmup): Train generator with MSE+cosine only.
Phase 2: Add adversarial loss with the discriminator.

Usage:
    python scripts/train_split.py
    python scripts/train_split.py --test_patients 0085,0110,0122,0132,0045
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
from model.discriminator import PatchDiscriminator1D

SAMPLE_RATE = 16000
CUCO_BASE   = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"

HIDDEN_DIM        = 128
N_LEVELS          = 2
DROPOUT           = 0.25
BATCH_SIZE        = 32
SEGMENT_LEN       = 64
SEGMENT_HOP       = 16
LR_G              = 5e-4
LR_D              = 2e-4
WEIGHT_DECAY      = 1e-3
EPOCHS            = 400
PATIENCE          = 50
COSINE_LOSS_WEIGHT = 0.5
LAMBDA_ADV        = 0.1
WARMUP_EPOCHS     = 50
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1
D_UPDATES_PER_G   = 1


def extract_all_features(knn_vc, wav_files):
    all_features = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)
        all_features.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")
    total = sum(f.shape[0] for f in all_features)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")
    return all_features


def pair_frames_knn(X, Y):
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    all_indices = []
    for i in range(0, X.shape[0], 5000):
        sim = X_norm[i:i + 5000] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))
    return X, Y[torch.cat(all_indices)]


def build_segments(pre_feats, post_feats, indices, augment=False):
    paired, post_unpaired = [], []
    for idx in indices:
        X, Y = pair_frames_knn(pre_feats[idx], post_feats[idx])
        n = X.shape[0]
        if n < SEGMENT_LEN:
            continue
        for s in range(0, n - SEGMENT_LEN + 1, SEGMENT_HOP):
            paired.append((X[s:s + SEGMENT_LEN].t(), Y[s:s + SEGMENT_LEN].t()))
        n_post = post_feats[idx].shape[0]
        for s in range(0, n_post - SEGMENT_LEN + 1, SEGMENT_HOP):
            post_unpaired.append(post_feats[idx][s:s + SEGMENT_LEN].t())
    return paired, post_unpaired


class FeatureSegmentDataset(Dataset):
    def __init__(self, segments, augment=False):
        self.segments = segments
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y = self.segments[idx]
        if self.augment:
            x = x + torch.randn_like(x) * AUGMENT_NOISE_STD
            mask = torch.rand(x.shape[-1]) > AUGMENT_MASK_PROB
            x = x * mask.unsqueeze(0)
        return x, y


class UnpairedPostDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


def recon_loss(y_pred, y_target, cosine_weight=COSINE_LOSS_WEIGHT):
    mse = F.mse_loss(y_pred, y_target)
    cos_sim = F.cosine_similarity(y_pred, y_target, dim=1).mean()
    return mse + cosine_weight * (1.0 - cos_sim), mse.item(), (1.0 - cos_sim).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--pre_dir', type=str,
                        default=os.path.join(CUCO_BASE, 'Tonsill', 'Speech', '1'))
    parser.add_argument('--post_dir', type=str,
                        default=os.path.join(CUCO_BASE, 'Tonsill', 'Speech', '2'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--test_patients', type=str,
                        default="0085,0110,0122,0132,0045",
                        help='Comma-separated fixed test patient IDs to exclude from training')
    parser.add_argument('--val_fraction', type=float, default=0.15,
                        help='Fraction of non-test patients to use for validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--extra_surgeries', action='store_true',
                        help='Also train on Fess+Sept+Contr data (train only, val stays Tonsill)')
    args = parser.parse_args()

    if args.output is None:
        suffix = '_multisurg' if args.extra_surgeries else ''
        args.output = os.path.join(os.path.dirname(__file__), '..', f'checkpoints{suffix}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.output, exist_ok=True)

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from utils import get_all_audio_pairs

    fixed_ids = set(p.strip() for p in args.test_patients.split(',') if p.strip())

    # Collect all audio types (Speech + TDU + Vowels + Sustained vowels), excluding test patients
    patient_pairs = get_all_audio_pairs(args.surgery, exclude=fixed_ids)
    all_pids = sorted(patient_pairs.keys())

    random.seed(args.seed)
    shuffled_pids = all_pids.copy()
    random.shuffle(shuffled_pids)
    n_val_pids = max(1, int(args.val_fraction * len(shuffled_pids)))
    val_pids   = set(shuffled_pids[:n_val_pids])
    train_pids = set(shuffled_pids[n_val_pids:])

    # Flatten to paired file lists (all audio types, all train/val patients)
    pid_of_file = [pid for pid in sorted(all_pids) for _ in patient_pairs[pid]]
    pre_files   = [pre  for pid in sorted(all_pids) for pre,  _   in patient_pairs[pid]]
    post_files  = [post for pid in sorted(all_pids) for _,    post in patient_pairs[pid]]
    n_patients = len(pre_files)
    train_idx = [i for i, pid in enumerate(pid_of_file) if pid in train_pids]
    val_idx   = [i for i, pid in enumerate(pid_of_file) if pid in val_pids]

    # Extra surgery data (appended to training set only)
    if args.extra_surgeries:
        extra_surgeries = ["Fess", "Sept", "Contr"]
        print(f"\nAdding extra surgery data to training...")
        n_before = len(pre_files)
        for surg in extra_surgeries:
            surg_pairs = get_all_audio_pairs(surg)
            n_surg = 0
            for pid in sorted(surg_pairs):
                for pre, post in surg_pairs[pid]:
                    pre_files.append(pre)
                    post_files.append(post)
                    n_surg += 1
            print(f"  {surg}: {n_surg} file pairs")
        n_extra = len(pre_files) - n_before
        train_idx = list(train_idx) + list(range(n_before, n_before + n_extra))
        print(f"  Total extra: {n_extra} files; train_idx now {len(train_idx)}")

    print(f"\n{args.surgery}: {len(all_pids)} train/val patients, {len(pid_of_file)} tonsill files")
    if args.extra_surgeries:
        print(f"  + {n_extra} extra-surgery files added to training")
    print(f"  Train: {len(train_pids)} tonsill patients + extra, {len(train_idx)} files total")
    print(f"  Val:   {len(val_pids)} patients, {len(val_idx)} files")
    print(f"  Test:  held out: {sorted(fixed_ids)}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': sorted(fixed_ids), 'train': sorted(train_pids),
                   'val': sorted(val_pids), 'n_files': n_patients,
                   'seed': args.seed}, f, indent=2)

    # Load WavLM
    print("\nLoading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("\nExtracting pre-surgery features...")
    pre_feats  = extract_all_features(knn_vc, pre_files)
    print("\nExtracting post-surgery features...")
    post_feats = extract_all_features(knn_vc, post_files)

    print("\nBuilding training segments...")
    train_paired, train_post_unpaired = build_segments(pre_feats, post_feats, train_idx)
    print("Building validation segments...")
    val_paired, _ = build_segments(pre_feats, post_feats, val_idx)

    train_ds  = FeatureSegmentDataset(train_paired, augment=True)
    val_ds    = FeatureSegmentDataset(val_paired,   augment=False)
    post_ds   = UnpairedPostDataset(train_post_unpaired)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    post_loader  = DataLoader(post_ds,  batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    print(f"\nTrain: {len(train_ds)} segs, Val: {len(val_ds)} segs, "
          f"Post unpaired: {len(post_ds)} segs")

    generator     = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                              dropout=DROPOUT).to(device)
    discriminator = PatchDiscriminator1D(feat_dim=1024, hidden_dim=256, n_layers=3).to(device)

    print(f"Generator:     {generator.count_parameters():,} params")
    print(f"Discriminator: {discriminator.count_parameters():,} params")

    opt_g   = torch.optim.AdamW(generator.parameters(),     lr=LR_G, weight_decay=WEIGHT_DECAY)
    opt_d   = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, weight_decay=WEIGHT_DECAY)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    best_val_loss    = float('inf')
    patience_counter = 0
    ckpt_path        = os.path.join(args.output, 'best_model.pt')

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()

        use_adv   = epoch > WARMUP_EPOCHS
        post_iter = iter(post_loader) if use_adv else None

        ep_recon = ep_g_adv = ep_d_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_pred = generator(x_batch)
            loss_rec, _, _ = recon_loss(y_pred, y_batch)
            loss_g = loss_rec

            if use_adv:
                fake_scores = discriminator(y_pred)
                loss_g_adv  = F.binary_cross_entropy_with_logits(
                    fake_scores, torch.ones_like(fake_scores))
                loss_g = loss_rec + LAMBDA_ADV * loss_g_adv
                ep_g_adv += loss_g_adv.item()

            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            if use_adv:
                for _ in range(D_UPDATES_PER_G):
                    try:
                        real_post = next(post_iter).to(device)
                    except StopIteration:
                        post_iter = iter(post_loader)
                        real_post = next(post_iter).to(device)

                    with torch.no_grad():
                        fake_post = generator(x_batch)

                    real_scores = discriminator(real_post)
                    fake_scores = discriminator(fake_post)
                    loss_d = 0.5 * (
                        F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) +
                        F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
                    )
                    opt_d.zero_grad()
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    opt_d.step()
                    ep_d_loss += loss_d.item()

            ep_recon  += loss_rec.item()
            n_batches += 1

        sched_g.step()
        if use_adv:
            sched_d.step()

        # Validate
        generator.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                loss, _, _ = recon_loss(generator(x_batch), y_batch)
                val_losses.append(loss.item())

        val_loss  = np.mean(val_losses)
        alpha_val = generator.alpha.item()
        phase     = "[WARMUP]" if not use_adv else ""
        print(f"Epoch {epoch:3d}/{args.epochs} {phase}  "
              f"recon={ep_recon/n_batches:.6f}  "
              f"g_adv={ep_g_adv/n_batches:.4f}  "
              f"d_loss={ep_d_loss/n_batches:.4f}  "
              f"val={val_loss:.6f}  alpha={alpha_val:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'alpha': alpha_val,
                'config': {'feat_dim': 1024, 'hidden_dim': HIDDEN_DIM,
                           'n_levels': N_LEVELS, 'dropout': DROPOUT},
            }, ckpt_path)
            print(f"  -> Saved best (val={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
