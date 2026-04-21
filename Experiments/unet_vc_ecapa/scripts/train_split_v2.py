"""
UNet-VC Content + ECAPA Speaker Loss — v2

Root cause of v1 failure:
  MSE(output, input) preserves BOTH content AND style (speaker characteristics).
  It therefore directly fights the ECAPA style loss, so the model barely transforms
  anything (alpha stays small, test ECAPA similarity gets worse than baseline).

Fix — decouple the two losses:
  Content loss:  1 - cosine_similarity(output, input)  per time-frame
    -> Cosine removes magnitude entirely (which carries speaker style in WavLM features).
    -> Only compares feature DIRECTIONS (which encode phoneme/linguistic content).
    -> The style loss can now freely move output magnitude/distribution without fighting
       content loss.

  Style loss:    1 - cosine_similarity(ECAPA(vocode(output)), ECAPA(post))
    -> Unchanged from v1 but applied EVERY step (was every 3) and weight raised to 3.0.

Other changes vs v1:
  - ECAPA loss every step, sampling 4 utterances for stable gradient
  - ECAPA weight 1.0 (content weight also 1.0, now orthogonal to style)
  - alpha initialised at 0.1 and clamped to [0, 0.5] to prevent content destruction
  - Smaller model (hidden_dim=64, dropout=0.4) to reduce overfitting on 28 patients

Usage:
    python scripts/train_split_v2.py
    python scripts/train_split_v2.py --surgery Tonsill --n_test 5
"""

import argparse
import os
import sys
import glob
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D

SAMPLE_RATE = 16000
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"

# Model config (same as v1 so checkpoints are comparable)
HIDDEN_DIM = 64
N_LEVELS = 2
DROPOUT = 0.4
BATCH_SIZE = 32
SEGMENT_LEN = 64
SEGMENT_HOP = 16
LR = 5e-4
WEIGHT_DECAY = 1e-3
EPOCHS = 300
PATIENCE = 20
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1

# Loss weights — the main change
CONTENT_WEIGHT = 1.0   # cosine content loss, range [0, 2]; 1.0 is fine
ECAPA_WEIGHT = 1.0     # lowered from 3.0 — high weight caused noisy, unstable gradients
ECAPA_UTTS_PER_STEP = 4  # raised from 2 for more stable gradient estimates


# ──────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────

def extract_features_for_files(knn_vc, wav_files):
    results = []
    for wf in tqdm(wav_files, desc="  WavLM features"):
        features = knn_vc.get_features(wf)
        results.append(features.cpu())
    return results


def build_segments_from_pre(pre_feats, indices,
                            segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    segments = []
    for idx in indices:
        feats = pre_feats[idx]
        n = feats.shape[0]
        if n < segment_len:
            continue
        for s in range(0, n - segment_len + 1, segment_hop):
            segments.append((feats[s:s + segment_len].t(), idx))  # (1024, T)
    print(f"    {len(segments)} segments")
    return segments


class PreSegmentDataset(Dataset):
    def __init__(self, segments, augment=False):
        self.segments = segments
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, utt_idx = self.segments[idx]
        if self.augment:
            x = x + torch.randn_like(x) * AUGMENT_NOISE_STD
            mask = torch.rand(x.shape[-1]) > AUGMENT_MASK_PROB
            x = x * mask.unsqueeze(0)
        return x, utt_idx


# ──────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────

def cosine_content_loss(pred, target):
    """
    Per-frame cosine similarity loss between pred and target.

    pred, target: (B, 1024, T)

    Cosine similarity over the 1024 feature dimension at each time step
    removes magnitude (which carries speaker identity in WavLM features)
    and only compares feature directions (which carry linguistic content).

    This lets the style loss move the output's global spectral statistics
    freely without conflicting with the content loss.

    Returns a scalar in [0, 2]; 0 = perfect content match.
    """
    # Flatten time into batch: (B*T, 1024)
    B, C, T = pred.shape
    p = pred.permute(0, 2, 1).reshape(B * T, C)
    t = target.permute(0, 2, 1).reshape(B * T, C)
    # Cosine similarity: bounded in [-1, 1]
    cos = F.cosine_similarity(p, t, dim=1)  # (B*T,)
    return (1.0 - cos).mean()              # 0 = perfect, 2 = opposite


def ecapa_embed_differentiable(ecapa, wav):
    """wav: (1, T_samples) -> emb: (1, 1, 192)  [fully differentiable]."""
    wav_lens = torch.ones(1, device=wav.device)
    feats = ecapa.mods.compute_features(wav)
    feats = ecapa.mods.mean_var_norm(feats, wav_lens)
    emb = ecapa.mods.embedding_model(feats, wav_lens)
    return emb


def vocode(hifigan, feats):
    """feats: (1, T, 1024) -> audio: (1, T_audio)."""
    return hifigan(feats).squeeze(1)


STYLE_CROP_LEN = 128   # WavLM frames fed to HiFiGAN for style loss (~0.8 s of audio)
                       # Full utterances (~3000 frames) OOM on 20GB GPU because
                       # all HiFiGAN conv activations are retained for backprop.
                       # 128 frames → 128×320 = 40,960 audio samples — fits fine.

def ecapa_style_loss(model, hifigan, ecapa, pre_feats_list, pre_ecapa_embs,
                     train_idx, post_ecapa_embs, device, n_utts=ECAPA_UTTS_PER_STEP):
    """
    Sample n_utts random training utterances, convert a short crop with `model`,
    vocode, compute ECAPA embedding, and compare to paired post-surgery target.

    Returns scalar cosine-distance style loss (differentiable through model + vocoder).
    """
    losses = []
    chosen = random.sample(train_idx, min(n_utts, len(train_idx)))
    for utt_idx in chosen:
        full_feats = pre_feats_list[utt_idx].to(device)          # (T, 1024)
        T = full_feats.shape[0]
        # Random crop to keep HiFiGAN activation memory bounded
        crop_len = min(T, STYLE_CROP_LEN)
        start = random.randint(0, max(0, T - crop_len))
        feats = full_feats[start:start + crop_len]               # (crop_len, 1024)
        spk_emb = pre_ecapa_embs[utt_idx].unsqueeze(0).to(device)  # (1, 192)
        out = model(feats.t().unsqueeze(0), spk_emb)             # (1, 1024, crop_len)
        audio = vocode(hifigan, out.transpose(1, 2))             # (1, T_audio)
        emb = ecapa_embed_differentiable(ecapa, audio).squeeze() # (192,)
        target = post_ecapa_embs[utt_idx].to(device)             # (192,)
        sim = F.cosine_similarity(emb.unsqueeze(0), target.unsqueeze(0))
        losses.append(1.0 - sim.squeeze())
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────
# Validation: ECAPA similarity on held-out patients
# ──────────────────────────────────────────────

def compute_val_ecapa(model, hifigan, ecapa, pre_feats, pre_ecapa_embs,
                      post_ecapa_embs, val_idx, device):
    model.eval()
    sims = []
    with torch.no_grad():
        for i in val_idx:
            feats = pre_feats[i].to(device)
            spk_emb = pre_ecapa_embs[i].unsqueeze(0).to(device)  # (1, 192)
            out = model(feats.t().unsqueeze(0), spk_emb)
            audio = vocode(hifigan, out.transpose(1, 2))
            emb = ecapa_embed_differentiable(ecapa, audio).squeeze()
            sim = F.cosine_similarity(
                emb.unsqueeze(0), post_ecapa_embs[i].to(device).unsqueeze(0)
            ).item()
            sims.append(sim)
    return float(np.mean(sims))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

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
        args.output = os.path.join(
            os.path.dirname(__file__), '..', f'results_{args.surgery.lower()}_v2{suffix}')

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
    n_extra = 0
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

    # ── Load external models ──
    print("\nLoading kNN-VC (WavLM + HiFi-GAN)...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("Loading ECAPA-TDNN (frozen speaker encoder)...")
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})

    hifigan = knn_vc.hifigan
    hifigan.eval()
    for p in hifigan.parameters():
        p.requires_grad = False
    for p in ecapa.mods.parameters():
        p.requires_grad = False

    # ── Extract pre-surgery WavLM features ──
    print("\nExtracting pre-surgery WavLM features...")
    pre_feats = extract_features_for_files(knn_vc, pre_files)

    # ── Pre-compute pre-surgery ECAPA embeddings (patient identity — available at inference) ──
    print("\nPre-computing pre-surgery ECAPA embeddings...")
    pre_ecapa_embs = {}
    for i in tqdm(range(len(pre_files)), desc="  Pre ECAPA embs"):
        sig, sr = torchaudio.load(pre_files[i])
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = ecapa.encode_batch(sig.to(device))
        pre_ecapa_embs[i] = emb.squeeze().detach().cpu()

    # ── Pre-compute post-surgery ECAPA embeddings (style targets) ──
    print("\nPre-computing post-surgery ECAPA embeddings...")
    post_ecapa_embs = {}
    for i in tqdm(train_idx + val_idx, desc="  Post ECAPA embs"):
        sig, sr = torchaudio.load(post_files[i])
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = ecapa.encode_batch(sig.to(device))
        post_ecapa_embs[i] = emb.squeeze().detach().cpu()

    # ── Build segments ──
    print("\nBuilding training segments...")
    train_segs = build_segments_from_pre(pre_feats, train_idx)
    print("Building validation segments...")
    val_segs = build_segments_from_pre(pre_feats, val_idx)

    train_ds = PreSegmentDataset(train_segs, augment=True)
    val_ds = PreSegmentDataset(val_segs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"\nTrain: {len(train_ds)} segs, Val: {len(val_ds)} segs")

    # ── Model ──
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(args.output, 'best_model.pt')
    best_val_ecapa = -1.0
    patience_counter = 0

    # ── Training loop ──
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        content_losses, style_losses = [], []

        for xb, utt_indices in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
            xb = xb.to(device)  # (B, 1024, T)

            # Build per-sample speaker conditioning from pre-surgery ECAPA
            spk_emb_batch = torch.stack(
                [pre_ecapa_embs[i.item()] for i in utt_indices]
            ).to(device)  # (B, 192)

            pred = model(xb, spk_emb_batch)    # (B, 1024, T)

            # Content loss: cosine per frame — orthogonal to style
            c_loss = CONTENT_WEIGHT * cosine_content_loss(pred, xb)

            # Style loss: ECAPA similarity on a short crop (STYLE_CROP_LEN frames)
            # to avoid OOM from storing full-utterance HiFiGAN activations.
            s_loss = ECAPA_WEIGHT * ecapa_style_loss(
                model, hifigan, ecapa, pre_feats, pre_ecapa_embs,
                train_idx, post_ecapa_embs, device)

            loss = c_loss + s_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            content_losses.append(c_loss.item())
            style_losses.append(s_loss.item())

        scheduler.step()

        # ── Validation ──
        val_ecapa = compute_val_ecapa(
            model, hifigan, ecapa, pre_feats, pre_ecapa_embs,
            post_ecapa_embs, val_idx, device)

        model.eval()
        val_content = []
        with torch.no_grad():
            for xb, utt_indices in val_loader:
                xb = xb.to(device)
                spk_emb_batch = torch.stack(
                    [pre_ecapa_embs[i.item()] for i in utt_indices]
                ).to(device)
                val_content.append(cosine_content_loss(model(xb, spk_emb_batch), xb).item())

        avg_c = np.mean(content_losses)
        avg_s = np.mean(style_losses)
        avg_vc = np.mean(val_content)
        alpha = model.alpha.item()

        epoch_bar.set_postfix(content=f"{avg_c:.4f}", style=f"{avg_s:.4f}",
                              val_ecapa=f"{val_ecapa:.4f}", alpha=f"{alpha:.4f}")
        tqdm.write(f"Epoch {epoch:3d}/{EPOCHS}  "
                   f"cos_content={avg_c:.4f}  ecapa_style={avg_s:.4f}  "
                   f"val_content={avg_vc:.4f}  val_ecapa={val_ecapa:.4f}  "
                   f"alpha={alpha:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_ecapa > best_val_ecapa:
            best_val_ecapa = val_ecapa
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_ecapa': val_ecapa,
                'alpha': alpha,
                'config': {'feat_dim': 1024, 'hidden_dim': HIDDEN_DIM,
                           'n_levels': N_LEVELS, 'dropout': DROPOUT},
            }, ckpt_path)
            tqdm.write(f"  -> Saved best (val_ecapa={val_ecapa:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f"  Early stop at epoch {epoch}")
                break

    print(f"\nBest val ECAPA similarity: {best_val_ecapa:.4f}")

    # ── Evaluate on test + train sets ──
    print(f"\n{'='*70}")
    print(f"  Evaluating on {len(test_idx)} test patients")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=0.0).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    def get_emb(path):
        sig, sr = torchaudio.load(path)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        with torch.no_grad():
            return ecapa.encode_batch(sig.to(device)).squeeze()

    conv_dir = os.path.join(args.output, 'converted')
    os.makedirs(conv_dir, exist_ok=True)

    def evaluate_patient(i, tag):
        feats = knn_vc.get_features(pre_files[i])
        spk_emb = pre_ecapa_embs[i].unsqueeze(0).to(device)  # (1, 192)
        with torch.no_grad():
            out = model(feats.t().unsqueeze(0).to(device), spk_emb).squeeze(0).t()
        wav = knn_vc.vocode(out[None]).cpu().squeeze()
        out_path = os.path.join(conv_dir, f"{names[i]}{tag}.wav")
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[i])
        emb_pre = get_emb(pre_files[i])
        sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        sim_pre = F.cosine_similarity(emb_conv.unsqueeze(0), emb_pre.unsqueeze(0)).item()
        baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()
        return {'name': names[i], 'sim_post': sim_post, 'sim_pre': sim_pre,
                'baseline': baseline}

    results_test, results_train = [], []

    print("\n--- TEST set ---")
    for i in tqdm(test_idx, desc="  Eval test"):
        r = evaluate_patient(i, '')
        results_test.append(r)
        tqdm.write(f"  [TEST]  {r['name']}: conv->post={r['sim_post']:.3f}  "
                   f"conv->pre={r['sim_pre']:.3f}  baseline={r['baseline']:.3f}  "
                   f"delta={r['sim_post'] - r['baseline']:+.3f}")

    print("\n--- TRAIN set (overfitting check) ---")
    for i in tqdm(train_idx, desc="  Eval train"):
        r = evaluate_patient(i, '_train')
        results_train.append(r)
        tqdm.write(f"  [TRAIN] {r['name']}: conv->post={r['sim_post']:.3f}  "
                   f"conv->pre={r['sim_pre']:.3f}  baseline={r['baseline']:.3f}  "
                   f"delta={r['sim_post'] - r['baseline']:+.3f}")

    test_post = [r['sim_post'] for r in results_test]
    test_base = [r['baseline'] for r in results_test]
    test_pre  = [r['sim_pre']  for r in results_test]
    train_post = [r['sim_post'] for r in results_train]
    train_base = [r['baseline'] for r in results_train]

    print(f"\n{'='*70}")
    print(f"  UNet-VC v2 (cosine content) — {args.surgery} — SUMMARY")
    print(f"{'='*70}")
    print(f"  TEST ({len(test_idx)} patients):")
    print(f"    Baseline (pre->post): {np.mean(test_base):.3f} ± {np.std(test_base):.3f}")
    print(f"    Conv -> post:         {np.mean(test_post):.3f} ± {np.std(test_post):.3f}")
    print(f"    Conv -> pre:          {np.mean(test_pre):.3f} ± {np.std(test_pre):.3f}")
    print(f"    Improvement:          {np.mean(test_post) - np.mean(test_base):+.3f}")
    print(f"  TRAIN ({len(train_idx)} patients):")
    print(f"    Baseline (pre->post): {np.mean(train_base):.3f} ± {np.std(train_base):.3f}")
    print(f"    Conv -> post:         {np.mean(train_post):.3f} ± {np.std(train_post):.3f}")
    print(f"    Improvement:          {np.mean(train_post) - np.mean(train_base):+.3f}")
    print(f"{'='*70}")

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump({
            'method': 'UNet-VC v2 (cosine content loss)',
            'surgery': args.surgery,
            'test': results_test, 'train': results_train,
            'test_summary': {
                'baseline': float(np.mean(test_base)),
                'conv_post': float(np.mean(test_post)),
                'conv_pre':  float(np.mean(test_pre)),
            },
            'train_summary': {
                'baseline': float(np.mean(train_base)),
                'conv_post': float(np.mean(train_post)),
            },
        }, f, indent=2)


if __name__ == '__main__':
    main()
