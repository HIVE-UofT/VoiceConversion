"""
UNet-VC with Content + ECAPA Speaker Loss — Train/Test Split

Loss decomposition that separates content from voice quality:
  - Content loss (every step):  MSE(output, input_pre) — preserve what is said
  - Speaker loss (every N steps): 1 - cos(ECAPA(output), ECAPA(post)) — sound like post speaker

No frame-level comparison with post-surgery WavLM features at all.
The model only sees post-surgery through the ECAPA speaker embedding.

Validation / model selection uses ECAPA similarity on val patients.

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
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D

SAMPLE_RATE = 16000
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"

# Model (v1 config)
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
AUGMENT_NOISE_STD = 0.02
AUGMENT_MASK_PROB = 0.1

# Loss weights
CONTENT_WEIGHT = 1.0       # MSE(output, input) — full content preservation
ECAPA_EVERY = 3            # speaker loss every 3 steps
ECAPA_LOSS_WEIGHT = 1.0    # equal footing with content


def extract_features_for_files(knn_vc, wav_files):
    results = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)
        results.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")
    return results


def build_segments_from_pre(pre_feats, indices,
                            segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    """Build segments from pre-surgery features only (no post pairing needed).
    Each segment records which utterance it came from."""
    segments = []
    for idx in indices:
        feats = pre_feats[idx]
        n = feats.shape[0]
        if n < segment_len:
            continue
        for s in range(0, n - segment_len + 1, segment_hop):
            segments.append((
                feats[s:s+segment_len].t(),  # (1024, seg_len)
                idx,                          # utterance index
            ))
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


def content_loss(y_pred, x_input):
    """Penalise deviation from input — preserve content."""
    return F.mse_loss(y_pred, x_input)


def ecapa_embed_differentiable(ecapa, wav):
    """Differentiable ECAPA embedding: wav (1, T) -> emb (1, 1, 192)."""
    wav_lens = torch.ones(1, device=wav.device)
    feats = ecapa.mods.compute_features(wav)
    feats = ecapa.mods.mean_var_norm(feats, wav_lens)
    emb = ecapa.mods.embedding_model(feats, wav_lens)
    return emb


def vocode_differentiable(hifigan, c):
    """Differentiable vocoding: c (1, T, 1024) -> audio (1, audio_len)."""
    y = hifigan(c)
    return y.squeeze(1)


def compute_val_ecapa(model, vocoder, ecapa, pre_feats, post_ecapa_embs,
                      val_idx, device):
    """Compute mean ECAPA similarity on val patients (vocode full utterances)."""
    model.eval()
    sims = []
    with torch.no_grad():
        for i in val_idx:
            feats = pre_feats[i].to(device)  # (T, 1024)
            out = model(feats.t().unsqueeze(0))  # (1, 1024, T)
            audio = vocode_differentiable(vocoder, out.transpose(1, 2))
            emb = ecapa_embed_differentiable(ecapa, audio).squeeze()
            target = post_ecapa_embs[i]
            sim = F.cosine_similarity(emb.unsqueeze(0), target.unsqueeze(0)).item()
            sims.append(sim)
    return np.mean(sims)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..',
                                    f'results_{args.surgery.lower()}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n = len(pre_files)
    names = [Path(f).stem for f in pre_files]

    # Patient-level split
    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    test_idx = sorted(indices[:args.n_test])
    cv_idx = sorted(indices[args.n_test:])
    random.shuffle(cv_idx)
    n_val = max(1, int(0.15 * len(cv_idx)))
    val_idx = sorted(cv_idx[:n_val])
    train_idx = sorted(cv_idx[n_val:])

    print(f"\n{args.surgery}: {n} patients")
    print(f"  Test  ({len(test_idx)}):  {[names[i] for i in test_idx]}")
    print(f"  Train ({len(train_idx)}): {len(train_idx)} patients")
    print(f"  Val   ({len(val_idx)}):   {[names[i] for i in val_idx]}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': [names[i] for i in test_idx],
                   'train': [names[i] for i in train_idx],
                   'val': [names[i] for i in val_idx],
                   'seed': args.seed}, f, indent=2)

    # ═══ Load models ═══
    print("\nLoading kNN-VC...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("Loading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})

    # Freeze vocoder and ECAPA
    vocoder = knn_vc.hifigan
    vocoder.eval()
    for p in vocoder.parameters():
        p.requires_grad = False
    for p in ecapa.mods.parameters():
        p.requires_grad = False

    # ═══ Extract features ═══
    print("\nExtracting pre-surgery features...")
    pre_feats = extract_features_for_files(knn_vc, pre_files)

    # ═══ Pre-compute post-surgery ECAPA embeddings (targets for speaker loss) ═══
    print("\nPre-computing post-surgery ECAPA embeddings...")
    post_ecapa_embs = {}
    for i in train_idx + val_idx:
        sig, sr = torchaudio.load(post_files[i])
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        with torch.no_grad():
            emb = ecapa.encode_batch(sig.to(device))
        post_ecapa_embs[i] = emb.squeeze().detach()
        print(f"  {names[i]}: done")

    # ═══ Build segments (pre-surgery only — no post pairing) ═══
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

    # ═══ Model ═══
    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(args.output, 'best_model.pt')
    best_val_ecapa = -1.0  # higher is better (cosine similarity)
    patience_counter = 0
    global_step = 0

    # ═══ Training loop ═══
    for epoch in range(1, EPOCHS + 1):
        model.train()
        content_losses = []
        ecapa_losses = []

        for xb, utt_indices in train_loader:
            xb = xb.to(device)  # (B, 1024, T)

            # Content preservation loss: output should stay close to input
            pred = model(xb)
            c_loss = CONTENT_WEIGHT * content_loss(pred, xb)
            loss = c_loss

            # Speaker loss (every ECAPA_EVERY steps) on a random full utterance
            global_step += 1
            if global_step % ECAPA_EVERY == 0:
                utt_idx = random.choice(train_idx)
                full_feats = pre_feats[utt_idx].to(device)  # (T, 1024)

                out_feats = model(full_feats.t().unsqueeze(0))       # (1, 1024, T)
                audio = vocode_differentiable(vocoder,
                            out_feats.transpose(1, 2))               # (1, audio_len)
                emb = ecapa_embed_differentiable(ecapa, audio).squeeze()  # (192,)

                target_emb = post_ecapa_embs[utt_idx]
                e_loss = 1.0 - F.cosine_similarity(
                    emb.unsqueeze(0), target_emb.unsqueeze(0)).squeeze()
                loss = loss + ECAPA_LOSS_WEIGHT * e_loss
                ecapa_losses.append(e_loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            content_losses.append(c_loss.item())

        scheduler.step()

        # ═══ Validation: ECAPA similarity on val patients (model selection) ═══
        val_ecapa = compute_val_ecapa(model, vocoder, ecapa, pre_feats,
                                       post_ecapa_embs, val_idx, device)

        # Also compute content loss on val for monitoring
        model.eval()
        val_content = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                val_content.append(content_loss(model(xb), xb).item())

        avg_content = np.mean(content_losses)
        avg_ecapa = np.mean(ecapa_losses) if ecapa_losses else 0.0
        avg_val_content = np.mean(val_content)
        print(f"Epoch {epoch:3d}/{EPOCHS}  content={avg_content:.6f}  ecapa_l={avg_ecapa:.4f}"
              f"  val_content={avg_val_content:.6f}  val_ecapa={val_ecapa:.4f}"
              f"  alpha={model.alpha.item():.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
        ecapa_losses = []

        # Save best by ECAPA similarity (higher = better)
        if val_ecapa > best_val_ecapa:
            best_val_ecapa = val_ecapa
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val_ecapa': val_ecapa, 'alpha': model.alpha.item(),
                'config': {'feat_dim': 1024, 'hidden_dim': HIDDEN_DIM,
                           'n_levels': N_LEVELS, 'dropout': DROPOUT},
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"Best val ECAPA similarity: {best_val_ecapa:.4f}")

    # ═══ Evaluate on test set ═══
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
        return ecapa.encode_batch(sig.to(device)).squeeze()

    conv_dir = os.path.join(args.output, 'converted')
    os.makedirs(conv_dir, exist_ok=True)

    results_test = []
    results_train = []

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
        results_test.append({'name': names[i], 'sim_post': sim_post,
                             'sim_pre': sim_pre, 'baseline': baseline})
        print(f"  [TEST]  {names[i]}: conv->post={sim_post:.3f}  conv->pre={sim_pre:.3f}"
              f"  baseline={baseline:.3f}  delta={sim_post-baseline:+.3f}")

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
        results_train.append({'name': names[i], 'sim_post': sim_post,
                              'sim_pre': sim_pre, 'baseline': baseline})
        print(f"  [TRAIN] {names[i]}: conv->post={sim_post:.3f}  conv->pre={sim_pre:.3f}"
              f"  baseline={baseline:.3f}  delta={sim_post-baseline:+.3f}")

    test_post = [r['sim_post'] for r in results_test]
    test_base = [r['baseline'] for r in results_test]
    test_pre = [r['sim_pre'] for r in results_test]
    train_post = [r['sim_post'] for r in results_train]
    train_base = [r['baseline'] for r in results_train]

    print(f"\n{'='*70}")
    print(f"  UNet-VC Content+ECAPA — {args.surgery} — SUMMARY")
    print(f"{'='*70}")
    print(f"  TEST ({len(test_idx)} patients):")
    print(f"    Baseline (pre->post): {np.mean(test_base):.3f} +/- {np.std(test_base):.3f}")
    print(f"    Conv -> post:         {np.mean(test_post):.3f} +/- {np.std(test_post):.3f}")
    print(f"    Conv -> pre:          {np.mean(test_pre):.3f} +/- {np.std(test_pre):.3f}")
    print(f"    Improvement:          {np.mean(test_post) - np.mean(test_base):+.3f}")
    print(f"  TRAIN ({len(train_idx)} patients):")
    print(f"    Baseline (pre->post): {np.mean(train_base):.3f} +/- {np.std(train_base):.3f}")
    print(f"    Conv -> post:         {np.mean(train_post):.3f} +/- {np.std(train_post):.3f}")
    print(f"    Improvement:          {np.mean(train_post) - np.mean(train_base):+.3f}")
    print(f"{'='*70}")

    all_results = {
        'method': 'UNet-VC Content+ECAPA',
        'surgery': args.surgery,
        'test': results_test, 'train': results_train,
        'test_summary': {'baseline': float(np.mean(test_base)),
                         'conv_post': float(np.mean(test_post)),
                         'conv_pre': float(np.mean(test_pre))},
        'train_summary': {'baseline': float(np.mean(train_base)),
                          'conv_post': float(np.mean(train_post))},
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
