"""
UNet Voice Mapper: pre-surgery ECAPA embedding -> post-surgery ECAPA embedding.

Treats the 192-dim ECAPA vector as a 1D signal (192 "positions", 1 channel)
and applies a residual 1D UNet with skip connections.

Difference from the MLP in train.py:
  - Conv1d operations learn relationships between neighbouring embedding dims
  - Skip connections give the decoder direct access to encoder activations
    at every scale (192 → 96 → 48 → 24 positions)
  - Same training setup: k-fold CV, multi-crop, augmentation, mixup, ensemble

Session 1 → Session 2  (same as the best MLP run)

Usage:
    python unet_mapper.py
    python unet_mapper.py --surgery Tonsill --n_test 5 --seed 42
"""

import argparse
import copy
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path

SAMPLE_RATE   = 16000
CUCO_BASE     = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"

LR            = 1e-4
WEIGHT_DECAY  = 0.1
EPOCHS        = 500
PATIENCE      = 60
BASE_CH       = 16      # channel width at first encoder level
N_LEVELS      = 3       # 192 → 96 → 48 → 24
DROPOUT       = 0.4
NOISE_SCALE   = 0.02
N_AUG         = 5
MIXUP_ALPHA   = 0.2
K_FOLDS       = 5
N_CROPS       = 10
CROP_SEC      = 3.0


# ─────────────────────────────────────────────────────────────
# UNet model
# ─────────────────────────────────────────────────────────────

class ConvBlock1D(nn.Module):
    """Two Conv1d + GroupNorm + GELU layers with optional residual."""
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        groups = lambda c: min(8, c)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups(out_ch), out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(groups(out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class UNet1D(nn.Module):
    """
    1D UNet: input shape (B, 1, 192) → output shape (B, 1, 192).

    The 192-dim ECAPA embedding is treated as a 1D signal with 1 channel.
    Encoder downsamples (192→96→48→24), decoder upsamples with skip connections.
    Final output: input + alpha * delta  (residual, starts as identity).
    """
    def __init__(self, emb_dim=192, base_ch=BASE_CH,
                 n_levels=N_LEVELS, dropout=DROPOUT):
        super().__init__()
        self.n_levels = n_levels

        # Encoder
        self.encoders   = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_ch
        self.input_proj = nn.Conv1d(1, ch, kernel_size=1)
        for _ in range(n_levels):
            self.encoders.append(ConvBlock1D(ch, ch, dropout))
            self.downsamples.append(
                nn.Conv1d(ch, ch * 2, kernel_size=4, stride=2, padding=1))
            ch *= 2

        # Bottleneck
        self.bottleneck = ConvBlock1D(ch, ch, dropout)

        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders  = nn.ModuleList()
        for _ in range(n_levels):
            self.upsamples.append(
                nn.ConvTranspose1d(ch, ch // 2, kernel_size=4, stride=2, padding=1))
            self.decoders.append(ConvBlock1D(ch, ch // 2, dropout))
            ch //= 2

        self.output_proj = nn.Conv1d(ch, 1, kernel_size=1)

        # Small init so training starts near identity
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Learnable residual scale (starts small, grows as needed)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: (B, 192) pre-surgery ECAPA embedding
        Returns:
            (B, 192) predicted post-surgery ECAPA embedding
        """
        # Treat embedding as 1D signal: (B, 1, 192)
        h = x.unsqueeze(1)
        T = h.shape[-1]

        # Pad to multiple of 2^n_levels
        divisor = 2 ** self.n_levels
        pad = (divisor - T % divisor) % divisor
        if pad:
            h = F.pad(h, (0, pad), mode='reflect')

        h = self.input_proj(h)

        skips = []
        for enc, down in zip(self.encoders, self.downsamples):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:   # handle rounding
                h = h[..., :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        delta = self.output_proj(h)[..., :T].squeeze(1)   # (B, 192)
        return x + self.alpha * delta

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# ECAPA extraction
# ─────────────────────────────────────────────────────────────

def load_ecapa():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"})


def extract_embeddings_multicrop(ecapa, wav_path, n_crops, crop_sec):
    sig, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        sig = torchaudio.functional.resample(sig, sr, SAMPLE_RATE)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)

    total       = sig.shape[1]
    crop_frames = int(crop_sec * SAMPLE_RATE)

    if n_crops == 1 or total <= crop_frames:
        with torch.no_grad():
            emb = ecapa.encode_batch(sig)
        return [emb.squeeze().cpu()]

    starts = np.linspace(0, total - crop_frames, n_crops).astype(int)
    embs = []
    for s in starts:
        with torch.no_grad():
            emb = ecapa.encode_batch(sig[:, s : s + crop_frames])
        embs.append(emb.squeeze().cpu())
    return embs


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def mixup(X, Y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0))
    return lam * X + (1 - lam) * X[idx], lam * Y + (1 - lam) * Y[idx]


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

def train_one_fold(X_tr, Y_tr, X_val, Y_val, fold_i, ckpt_path):
    model     = UNet1D()
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
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            idx = torch.randperm(X_tr.size(0))
            X_parts.append(lam * X_tr + (1 - lam) * X_tr[idx])
            Y_parts.append(lam * Y_tr + (1 - lam) * Y_tr[idx])

        pred = model(torch.cat(X_parts))
        loss = (1.0 - F.cosine_similarity(pred, torch.cat(Y_parts))).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def train_final(X_cv, Y_cv, n_epochs, ckpt_path):
    model     = UNet1D()
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
            lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            idx = torch.randperm(X_cv.size(0))
            X_parts.append(lam * X_cv + (1 - lam) * X_cv[idx])
            Y_parts.append(lam * Y_cv + (1 - lam) * Y_cv[idx])

        pred = model(torch.cat(X_parts))
        loss = (1.0 - F.cosine_similarity(pred, torch.cat(Y_parts))).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), ckpt_path)
    return model


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery',  type=str, default='Tonsill')
    parser.add_argument('--n_test',   type=int, default=5)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--output',   type=str, default=None)
    parser.add_argument('--k_folds',  type=int, default=K_FOLDS)
    parser.add_argument('--n_crops',  type=int, default=N_CROPS)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), f'results_{args.surgery.lower()}_unet')
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  UNet ECAPA Mapper: pre (ses1) -> post (ses2)")
    print(f"  Surgery: {args.surgery}  n_crops={args.n_crops}  crop_sec={CROP_SEC}")
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

    print(f"\n{n} patients total")
    print(f"  CV pool ({len(cv_idx)}): {[names[i] for i in cv_idx]}")
    print(f"  Test   ({len(test_idx)}): {[names[i] for i in test_idx]}")

    # ── Extract ECAPA embeddings ──
    print(f"\nLoading ECAPA-TDNN...")
    ecapa = load_ecapa()
    print(f"\nExtracting embeddings ({args.n_crops} crops × {CROP_SEC}s)...")

    pre_crops, post_crops = {}, {}
    for i in range(n):
        pre_crops[i]  = extract_embeddings_multicrop(
            ecapa, pre_files[i],  args.n_crops, CROP_SEC)
        post_crops[i] = extract_embeddings_multicrop(
            ecapa, post_files[i], args.n_crops, CROP_SEC)
        print(f"  {names[i]}: {len(pre_crops[i])} pre / {len(post_crops[i])} post crops")

    mean_pre  = {i: torch.stack(pre_crops[i]).mean(0)  for i in range(n)}
    mean_post = {i: torch.stack(post_crops[i]).mean(0) for i in range(n)}

    # ── Baseline ──
    baseline_cv   = [cosine_sim(mean_pre[i], mean_post[i]) for i in cv_idx]
    baseline_test = [cosine_sim(mean_pre[i], mean_post[i]) for i in test_idx]
    print(f"\nBaseline cosine sim (no transform, mean embeddings):")
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

    model_info = UNet1D()
    print(f"\nUNet1D  base_ch={BASE_CH}  n_levels={N_LEVELS}  "
          f"dropout={DROPOUT}  LR={LR}  wd={WEIGHT_DECAY}")
    print(f"Parameters: {model_info.count_parameters():,}")
    del model_info

    print(f"\nK-fold CV ({args.k_folds} folds)  "
          f"n_aug={N_AUG}  mixup={MIXUP_ALPHA}  noise={NOISE_SCALE}\n")
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

        ckpt  = os.path.join(args.output, f'fold_{fold_i}_unet.pt')
        model, best_ep = train_one_fold(X_tr, Y_tr, X_val, Y_val, fold_i, ckpt)
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
          f"median={median_ep}  final_epochs={final_ep}")

    print(f"\nTraining final model on {len(X_cv_all)} pairs "
          f"for {final_ep} epochs...")
    ckpt_final  = os.path.join(args.output, 'final_unet.pt')
    final_model = train_final(X_cv_all, Y_cv_all, final_ep, ckpt_final)
    final_model.eval()

    all_models = fold_models + [final_model]
    with torch.no_grad():
        ensemble_pred = torch.stack([m(X_test) for m in all_models]).mean(0)

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
              f"unet={sim_ens:.4f}  "
              f"delta={sim_ens - sim_base:+.4f}")

    mean_ens  = np.mean(test_sims)
    mean_base = np.mean(baseline_test)
    mean_ms   = np.mean(ms_test)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:   {mean_base:.4f}")
    print(f"  Mean shift: {mean_ms:.4f}  delta={mean_ms  - mean_base:+.4f}")
    print(f"  UNet:       {mean_ens:.4f}  delta={mean_ens - mean_base:+.4f}")
    print(f"  (CV val sim: {np.mean(fold_val_sims):.4f} "
          f"± {np.std(fold_val_sims):.4f})")
    print(f"{'='*60}")

    results = {
        'surgery': args.surgery, 'k_folds': args.k_folds, 'n_crops': args.n_crops,
        'architecture': {
            'base_ch': BASE_CH, 'n_levels': N_LEVELS, 'dropout': DROPOUT,
            'lr': LR, 'weight_decay': WEIGHT_DECAY,
            'n_aug': N_AUG, 'mixup_alpha': MIXUP_ALPHA, 'noise_scale': NOISE_SCALE,
            'parameters': UNet1D().count_parameters(),
        },
        'split': {
            'cv_pool': [names[i] for i in cv_idx],
            'test':    [names[i] for i in test_idx],
        },
        'cv_val_sim': {'mean': float(np.mean(fold_val_sims)),
                       'std':  float(np.std(fold_val_sims))},
        'fold_best_epochs':   fold_epochs,
        'final_model_epochs': final_ep,
        'summary': {
            'baseline':   float(mean_base),
            'mean_shift': float(mean_ms),
            'unet':       float(mean_ens),
            'delta':      float(mean_ens - mean_base),
        },
        'per_patient_test': [
            {'name':       names[test_idx[j]],
             'baseline':   float(baseline_test[j]),
             'mean_shift': float(ms_test[j]),
             'unet':       float(cosine_sim(ensemble_pred[j], Y_test[j]))}
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
