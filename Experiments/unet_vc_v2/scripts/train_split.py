"""
UNet-VC v2 — Train/Test Split (no LOO)

Hold out N_TEST patients, train on the rest, evaluate on test set.
Uses same-patient frame pairing and v1 model/training config.

Usage:
    python scripts/train_split.py
    python scripts/train_split.py --surgery Tonsill --n_test 5
"""

import argparse
import os
import sys
import glob
import json
import random
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D

SAMPLE_RATE = 16000
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"

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


def extract_features_for_files(knn_vc, wav_files):
    results = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)
        results.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")
    return results


def pair_frames_knn(X, Y):
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        sim = X_norm[i:i + chunk_size] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))
    return X, Y[torch.cat(all_indices)]


def build_segments_cross_patient(pre_feats, post_feats, indices,
                                  segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    post_pool = torch.cat([post_feats[i] for i in indices], dim=0)
    print(f"    Post pool: {post_pool.shape[0]} frames from {len(indices)} patients")
    segments = []
    for idx in indices:
        X, Y = pair_frames_knn(pre_feats[idx], post_pool)
        n = X.shape[0]
        if n < segment_len:
            continue
        for s in range(0, n - segment_len + 1, segment_hop):
            segments.append((X[s:s+segment_len].t(), Y[s:s+segment_len].t()))
    print(f"    {len(segments)} segments")
    return segments


def build_segments_same_patient(pre_feats, post_feats, indices,
                                 segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    segments = []
    for idx in indices:
        X, Y = pair_frames_knn(pre_feats[idx], post_feats[idx])
        n = X.shape[0]
        if n < segment_len:
            continue
        for s in range(0, n - segment_len + 1, segment_hop):
            segments.append((X[s:s+segment_len].t(), Y[s:s+segment_len].t()))
    print(f"    {len(segments)} segments (same-patient)")
    return segments


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


def combined_loss(y_pred, y_target):
    mse = F.mse_loss(y_pred, y_target)
    cos_loss = 1.0 - F.cosine_similarity(y_pred, y_target, dim=1).mean()
    return mse + COSINE_LOSS_WEIGHT * cos_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    use_cross = False
    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..',
                                    f'results_{args.surgery.lower()}_same')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Pairing: {'CROSS-PATIENT' if use_cross else 'SAME-PATIENT'}")
    os.makedirs(args.output, exist_ok=True)

    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n = len(pre_files)
    names = [Path(f).stem for f in pre_files]

    # Split
    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    test_idx = sorted(indices[:args.n_test])
    cv_idx = sorted(indices[args.n_test:])

    # Further split cv into train/val (85/15)
    random.shuffle(cv_idx)
    n_val = max(1, int(0.15 * len(cv_idx)))
    val_idx = sorted(cv_idx[:n_val])
    train_idx = sorted(cv_idx[n_val:])

    test_names = [names[i] for i in test_idx]
    train_names = [names[i] for i in train_idx]
    val_names = [names[i] for i in val_idx]

    print(f"\n{args.surgery}: {n} patients")
    print(f"  Test ({len(test_idx)}):  {test_names}")
    print(f"  Train ({len(train_idx)}): {len(train_idx)} patients")
    print(f"  Val ({len(val_idx)}):   {val_names}")

    # Save split
    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': test_names, 'train': train_names, 'val': val_names,
                   'seed': args.seed}, f, indent=2)

    # Extract features
    print("\nLoading kNN-VC...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("\nExtracting pre-surgery features...")
    pre_feats = extract_features_for_files(knn_vc, pre_files)
    print("\nExtracting post-surgery features...")
    post_feats = extract_features_for_files(knn_vc, post_files)

    # Build segments
    print("\nBuilding training segments...")
    if use_cross:
        train_segs = build_segments_cross_patient(pre_feats, post_feats, train_idx)
    else:
        train_segs = build_segments_same_patient(pre_feats, post_feats, train_idx)

    print("Building validation segments...")
    val_segs = build_segments_same_patient(pre_feats, post_feats, val_idx)

    train_ds = FeatureSegmentDataset(train_segs, augment=True)
    val_ds = FeatureSegmentDataset(val_segs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} segs, Val: {len(val_ds)} segs")

    # Train
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(args.output, 'best_model.pt')
    best_val = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = combined_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(combined_loss(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch:3d}/{EPOCHS}  val={val_loss:.6f}  alpha={model.alpha.item():.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val_loss': val_loss, 'alpha': model.alpha.item(),
                'config': {'feat_dim': 1024, 'hidden_dim': HIDDEN_DIM,
                           'n_levels': N_LEVELS, 'dropout': DROPOUT},
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"  Best val: {best_val:.6f}")

    # ═══ Evaluate on test set ═══
    print(f"\n{'='*70}")
    print(f"  Evaluating on {len(test_idx)} test patients")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=0.0).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})

    def get_emb(path):
        sig, sr = torchaudio.load(path)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return ecapa.encode_batch(sig).squeeze()

    conv_dir = os.path.join(args.output, 'converted')
    os.makedirs(conv_dir, exist_ok=True)

    results_test = []
    results_train = []

    # Convert and evaluate TEST patients
    print("\n--- TEST set ---")
    for i in test_idx:
        feats = knn_vc.get_features(pre_files[i])
        with torch.no_grad():
            out = model(feats.t().unsqueeze(0).to(device)).squeeze(0).t()
        wav = knn_vc.vocode(out[None]).cpu().squeeze()
        out_path = os.path.join(conv_dir, names[i] + '.wav')
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[i])
        emb_pre = get_emb(pre_files[i])
        sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        sim_pre = F.cosine_similarity(emb_conv.unsqueeze(0), emb_pre.unsqueeze(0)).item()
        baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()
        results_test.append({'name': names[i], 'sim_post': sim_post, 'sim_pre': sim_pre, 'baseline': baseline})
        print(f"  [TEST]  {names[i]}: conv->post={sim_post:.3f}  baseline={baseline:.3f}  delta={sim_post-baseline:+.3f}")

    # Convert and evaluate TRAIN patients (overfitting check)
    print("\n--- TRAIN set (overfitting check) ---")
    for i in train_idx:
        feats = knn_vc.get_features(pre_files[i])
        with torch.no_grad():
            out = model(feats.t().unsqueeze(0).to(device)).squeeze(0).t()
        wav = knn_vc.vocode(out[None]).cpu().squeeze()
        out_path = os.path.join(conv_dir, names[i] + '_train.wav')
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[i])
        emb_pre = get_emb(pre_files[i])
        sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        sim_pre = F.cosine_similarity(emb_conv.unsqueeze(0), emb_pre.unsqueeze(0)).item()
        baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()
        results_train.append({'name': names[i], 'sim_post': sim_post, 'sim_pre': sim_pre, 'baseline': baseline})
        print(f"  [TRAIN] {names[i]}: conv->post={sim_post:.3f}  baseline={baseline:.3f}  delta={sim_post-baseline:+.3f}")

    # Summary
    test_post = [r['sim_post'] for r in results_test]
    test_base = [r['baseline'] for r in results_test]
    train_post = [r['sim_post'] for r in results_train]
    train_base = [r['baseline'] for r in results_train]

    print(f"\n{'='*70}")
    print(f"  UNet-VC v2 — {args.surgery} — SUMMARY")
    print(f"  Pairing: {'CROSS-PATIENT' if use_cross else 'SAME-PATIENT'}")
    print(f"{'='*70}")
    print(f"  TEST ({len(test_idx)} patients):")
    print(f"    Baseline:     {np.mean(test_base):.3f} +/- {np.std(test_base):.3f}")
    print(f"    Conv vs post: {np.mean(test_post):.3f} +/- {np.std(test_post):.3f}")
    print(f"    Improvement:  {np.mean(test_post) - np.mean(test_base):+.3f}")
    print(f"  TRAIN ({len(train_idx)} patients):")
    print(f"    Baseline:     {np.mean(train_base):.3f} +/- {np.std(train_base):.3f}")
    print(f"    Conv vs post: {np.mean(train_post):.3f} +/- {np.std(train_post):.3f}")
    print(f"    Improvement:  {np.mean(train_post) - np.mean(train_base):+.3f}")
    print(f"{'='*70}")

    all_results = {
        'method': f'UNet-VC v2 ({"cross" if use_cross else "same"})',
        'surgery': args.surgery,
        'test': results_test, 'train': results_train,
        'test_summary': {'baseline': float(np.mean(test_base)), 'conv_post': float(np.mean(test_post))},
        'train_summary': {'baseline': float(np.mean(train_base)), 'conv_post': float(np.mean(train_post))},
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
