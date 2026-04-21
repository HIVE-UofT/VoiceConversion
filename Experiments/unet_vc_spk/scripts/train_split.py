"""
UNet-VC Speaker-Conditioned — Train/Test Split

Hold out N_TEST patients, train on the rest, evaluate on test set.
The U-Net is conditioned on a per-utterance speaker embedding (mean-pooled
WavLM features of the pre-surgery utterance) via FiLM at every level.

No post-surgery audio is needed at inference — only pre-surgery.

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
from model.unet_spk import SpeakerConditionedUNet

SAMPLE_RATE = 16000
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"

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
IDENTITY_REG_WEIGHT = 0.3    # penalise deviation from input (don't change more than needed)
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


def build_segments(pre_feats, post_feats, indices,
                   segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    """
    Same-patient pairing. Each segment is stored with its utterance-level
    speaker embedding (mean-pooled pre features).
    Returns list of (x_seg, y_seg, spk_emb) tuples.
    """
    segments = []
    for idx in indices:
        X, Y = pair_frames_knn(pre_feats[idx], post_feats[idx])
        spk_emb = pre_feats[idx].mean(dim=0)   # (1024,) — utterance-level speaker emb
        n = X.shape[0]
        if n < segment_len:
            continue
        for s in range(0, n - segment_len + 1, segment_hop):
            segments.append((
                X[s:s+segment_len].t(),   # (1024, seg_len)
                Y[s:s+segment_len].t(),   # (1024, seg_len)
                spk_emb,                  # (1024,)
            ))
    print(f"    {len(segments)} segments")
    return segments


class SpeakerSegmentDataset(Dataset):
    def __init__(self, segments, augment=False):
        self.segments = segments
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y, spk = self.segments[idx]
        if self.augment:
            x = x + torch.randn_like(x) * AUGMENT_NOISE_STD
            mask = torch.rand(x.shape[-1]) > AUGMENT_MASK_PROB
            x = x * mask.unsqueeze(0)
        return x, y, spk


def combined_loss(y_pred, y_target, x_input):
    mse = F.mse_loss(y_pred, y_target)
    cos_loss = 1.0 - F.cosine_similarity(y_pred, y_target, dim=1).mean()
    # Identity reg: penalise deviation from input — forces model to only change what it must
    identity_reg = F.mse_loss(y_pred, x_input)
    return mse + COSINE_LOSS_WEIGHT * cos_loss + IDENTITY_REG_WEIGHT * identity_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--test_patients', type=str,
                        default="0085,0110,0122,0132,0045",
                        help='Comma-separated fixed test patient IDs (overrides --n_test)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--extra_surgeries', action='store_true',
                        help='Also train on Fess+Sept+Contr data (train only, val stays Tonsill)')
    args = parser.parse_args()

    if args.output is None:
        suffix = '_multisurg' if args.extra_surgeries else ''
        args.output = os.path.join(os.path.dirname(__file__), '..',
                                    f'results_{args.surgery.lower()}{suffix}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from utils import get_all_audio_pairs

    fixed_ids = set(p.strip() for p in args.test_patients.split(',') if p.strip()) \
                if args.test_patients else set()

    # Collect all audio types (Speech + TDU + Vowels + Sustained vowels), excluding test patients
    patient_pairs = get_all_audio_pairs(args.surgery, exclude=fixed_ids)
    all_pids = sorted(patient_pairs.keys())

    random.seed(args.seed)
    shuffled_pids = all_pids.copy()
    random.shuffle(shuffled_pids)
    n_val_pids = max(1, int(0.15 * len(shuffled_pids)))
    val_pids   = set(shuffled_pids[:n_val_pids])
    train_pids = set(shuffled_pids[n_val_pids:])

    # Flatten to paired file lists (all audio types, all train/val patients)
    pid_of_file = [pid for pid in sorted(all_pids) for _ in patient_pairs[pid]]
    pre_files   = [pre  for pid in sorted(all_pids) for pre,  _   in patient_pairs[pid]]
    post_files  = [post for pid in sorted(all_pids) for _,    post in patient_pairs[pid]]
    n = len(pre_files)
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

    print(f"\n{args.surgery}: {len(all_pids)} train/val patients, {n} tonsill files")
    if args.extra_surgeries:
        print(f"  + {n_extra} extra-surgery files added to training")
    print(f"  Train: {len(train_pids)} tonsill patients + extra, {len(train_idx)} files total")
    print(f"  Val:   {len(val_pids)} patients, {len(val_idx)} files")
    print(f"  Test:  held out: {sorted(fixed_ids)}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': sorted(fixed_ids), 'train': sorted(train_pids),
                   'val': sorted(val_pids), 'n_files': n,
                   'seed': args.seed}, f, indent=2)

    # Extract features
    print("\nLoading kNN-VC...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("\nExtracting pre-surgery features...")
    pre_feats  = extract_features_for_files(knn_vc, pre_files)
    print("\nExtracting post-surgery features...")
    post_feats = extract_features_for_files(knn_vc, post_files)

    # Build segments
    print("\nBuilding training segments...")
    train_segs = build_segments(pre_feats, post_feats, train_idx)
    print("Building validation segments...")
    val_segs   = build_segments(pre_feats, post_feats, val_idx)

    train_ds = SpeakerSegmentDataset(train_segs, augment=True)
    val_ds   = SpeakerSegmentDataset(val_segs,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\nTrain: {len(train_ds)} segs, Val: {len(val_ds)} segs")

    # Model
    model = SpeakerConditionedUNet(feat_dim=1024, spk_dim=1024,
                                    hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                                    dropout=DROPOUT).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(args.output, 'best_model.pt')
    best_val  = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb, spk in train_loader:
            xb, yb, spk = xb.to(device), yb.to(device), spk.to(device)
            loss = combined_loss(model(xb, spk), yb, xb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, spk in val_loader:
                xb, yb, spk = xb.to(device), yb.to(device), spk.to(device)
                val_losses.append(combined_loss(model(xb, spk), yb, xb).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}"
              f"  alpha={model.alpha.item():.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val_loss': val_loss, 'alpha': model.alpha.item(),
                'config': {'feat_dim': 1024, 'spk_dim': 1024,
                           'hidden_dim': HIDDEN_DIM, 'n_levels': N_LEVELS,
                           'dropout': DROPOUT},
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"Best val: {best_val:.6f}")

    # ═══ Evaluate on test set ═══
    print(f"\n{'='*70}")
    print(f"  Evaluating on {len(test_idx)} test patients")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SpeakerConditionedUNet(feat_dim=1024, spk_dim=1024,
                                    hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
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

    results_test  = []
    results_train = []

    print("\n--- TEST set ---")
    for i in test_idx:
        feats   = knn_vc.get_features(pre_files[i]).cpu()
        spk_emb = feats.mean(dim=0).unsqueeze(0).to(device)  # (1, 1024)
        with torch.no_grad():
            out = model(feats.t().unsqueeze(0).to(device), spk_emb).squeeze(0).t()
        wav = knn_vc.vocode(out[None]).cpu().squeeze()
        out_path = os.path.join(conv_dir, names[i] + '.wav')
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[i])
        emb_pre  = get_emb(pre_files[i])
        sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        baseline = F.cosine_similarity(emb_pre.unsqueeze(0),  emb_post.unsqueeze(0)).item()
        results_test.append({'name': names[i], 'sim_post': sim_post, 'baseline': baseline})
        print(f"  [TEST]  {names[i]}: conv->post={sim_post:.3f}  baseline={baseline:.3f}"
              f"  delta={sim_post-baseline:+.3f}")

    print("\n--- TRAIN set (overfitting check) ---")
    for i in train_idx:
        feats   = knn_vc.get_features(pre_files[i]).cpu()
        spk_emb = feats.mean(dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(feats.t().unsqueeze(0).to(device), spk_emb).squeeze(0).t()
        wav = knn_vc.vocode(out[None]).cpu().squeeze()
        out_path = os.path.join(conv_dir, names[i] + '_train.wav')
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[i])
        emb_pre  = get_emb(pre_files[i])
        sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        baseline = F.cosine_similarity(emb_pre.unsqueeze(0),  emb_post.unsqueeze(0)).item()
        results_train.append({'name': names[i], 'sim_post': sim_post, 'baseline': baseline})
        print(f"  [TRAIN] {names[i]}: conv->post={sim_post:.3f}  baseline={baseline:.3f}"
              f"  delta={sim_post-baseline:+.3f}")

    test_post  = [r['sim_post']  for r in results_test]
    test_base  = [r['baseline']  for r in results_test]
    train_post = [r['sim_post']  for r in results_train]
    train_base = [r['baseline']  for r in results_train]

    print(f"\n{'='*70}")
    print(f"  UNet-VC Speaker-Conditioned — {args.surgery} — SUMMARY")
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
        'method': 'UNet-VC Speaker-Conditioned',
        'surgery': args.surgery,
        'test': results_test, 'train': results_train,
        'test_summary':  {'baseline': float(np.mean(test_base)),
                          'conv_post': float(np.mean(test_post))},
        'train_summary': {'baseline': float(np.mean(train_base)),
                          'conv_post': float(np.mean(train_post))},
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
