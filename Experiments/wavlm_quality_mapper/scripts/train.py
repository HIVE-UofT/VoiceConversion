"""
Train Multi-Layer Quality Mapper: Pre → Post Surgery

Learns to transform WavLM layers 12-16 from pre-surgery to post-surgery.
Uses cached multi-layer features from analyze_layers.py (or extracts fresh).

Training pipeline:
  1. Load/extract WavLM layers 12-16 for all patients
  2. Same-patient frame pairing via cosine similarity (on layer-averaged features)
  3. Segment into temporal windows
  4. Train MultiLayerMapper with MSE + cosine loss per layer
  5. Evaluate on held-out test patients

Usage:
    python scripts/train.py
    python scripts/train.py --surgery Tonsill --n_test 5
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.mapper import MultiLayerMapper

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"
SAMPLE_RATE = 16000
LAYERS = [12, 13, 14, 15, 16]
LAYER_INDICES = [l - 1 for l in LAYERS]  # 0-indexed

# Model
HIDDEN_DIM = 128
N_LEVELS = 2
DROPOUT = 0.3

# Training
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

# Vocoder layer (for audio reconstruction evaluation)
VOCODER_LAYER = 6  # knn-vc HiFi-GAN was trained on layer 6


# ──────────────────────────────────────────────
# WavLM Multi-Layer Extractor
# ──────────────────────────────────────────────
class WavLMMultiLayerExtractor:
    def __init__(self, device, layers=LAYERS):
        from transformers import WavLMModel
        print("Loading WavLM-Large...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.layer_indices = [l - 1 for l in layers]
        print(f"  Extracting layers: {layers}")

    @torch.no_grad()
    def extract(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).to(self.device)

        outputs = self.model(wav.unsqueeze(0), output_hidden_states=True)
        # Stack selected layers: (n_layers, T, 1024)
        selected = torch.stack(
            [outputs.hidden_states[li + 1].squeeze(0) for li in self.layer_indices],
            dim=0
        ).cpu()
        return selected  # (n_layers, T, 1024)

    @torch.no_grad()
    def extract_layer6(self, wav_path):
        """Extract layer 6 for vocoder compatibility."""
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).to(self.device)

        outputs = self.model(wav.unsqueeze(0), output_hidden_states=True)
        return outputs.hidden_states[VOCODER_LAYER].squeeze(0).cpu()  # (T, 1024)


def extract_all(extractor, wav_files, cache_path):
    if os.path.exists(cache_path):
        print(f"  Loading cached: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    results = []
    for wf in wav_files:
        feats = extractor.extract(wf)  # (n_layers, T, 1024)
        name = Path(wf).stem
        results.append((name, feats))
        print(f"    {name}: {feats.shape[1]} frames")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(results, cache_path)
    return results


# ──────────────────────────────────────────────
# Frame Pairing & Segmentation
# ──────────────────────────────────────────────

def pair_frames(pre_feats, post_feats):
    """
    pre_feats: (n_layers, T1, 1024)
    post_feats: (n_layers, T2, 1024)

    Pairs frames using cosine similarity on the average across layers.
    Returns: (pre_paired, post_paired) both (n_layers, T1, 1024)
    """
    # Average across layers for matching
    pre_avg = pre_feats.mean(dim=0)   # (T1, 1024)
    post_avg = post_feats.mean(dim=0)  # (T2, 1024)

    pre_norm = pre_avg / (pre_avg.norm(dim=1, keepdim=True) + 1e-8)
    post_norm = post_avg / (post_avg.norm(dim=1, keepdim=True) + 1e-8)

    chunk_size = 5000
    all_indices = []
    for i in range(0, pre_norm.shape[0], chunk_size):
        sim = pre_norm[i:i + chunk_size] @ post_norm.t()
        all_indices.append(sim.argmax(dim=1))
    indices = torch.cat(all_indices)

    return pre_feats, post_feats[:, indices]  # (n_layers, T1, 1024) each


def build_segments(pre_data, post_data, patient_indices,
                   segment_len=SEGMENT_LEN, segment_hop=SEGMENT_HOP):
    """Build (source, target) segment pairs for training."""
    segments = []
    for idx in patient_indices:
        name_pre, feats_pre = pre_data[idx]
        name_post, feats_post = post_data[idx]

        pre_paired, post_paired = pair_frames(feats_pre, feats_post)
        # pre_paired: (n_layers, T, 1024), post_paired: (n_layers, T, 1024)

        T = pre_paired.shape[1]
        if T < segment_len:
            continue

        for s in range(0, T - segment_len + 1, segment_hop):
            # (n_layers, 1024, seg_len)
            x = pre_paired[:, s:s+segment_len].permute(0, 2, 1)
            y = post_paired[:, s:s+segment_len].permute(0, 2, 1)
            segments.append((x, y))

    print(f"    {len(segments)} segments from {len(patient_indices)} patients")
    return segments


class MultiLayerDataset(Dataset):
    def __init__(self, segments, augment=False):
        self.segments = segments
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y = self.segments[idx]  # each (n_layers, 1024, seg_len)
        if self.augment:
            x = x + torch.randn_like(x) * AUGMENT_NOISE_STD
            mask = torch.rand(x.shape[-1]) > AUGMENT_MASK_PROB
            x = x * mask.unsqueeze(0).unsqueeze(0)
        return x, y


# ──────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────

def multi_layer_loss(pred, target):
    """
    pred, target: (B, n_layers, 1024, T)
    Returns weighted sum of per-layer MSE + cosine loss.
    """
    total = 0.0
    B, L, C, T = pred.shape
    for i in range(L):
        p = pred[:, i]  # (B, 1024, T)
        t = target[:, i]
        mse = F.mse_loss(p, t)
        cos = 1.0 - F.cosine_similarity(p, t, dim=1).mean()
        total = total + mse + COSINE_LOSS_WEIGHT * cos
    return total / L


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

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

    # ── Data ──
    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n = len(pre_files)
    names = [Path(f).stem for f in pre_files]
    print(f"\n{args.surgery}: {n} patients")

    # ── Split ──
    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    test_idx = sorted(indices[:args.n_test])
    cv_idx = sorted(indices[args.n_test:])
    random.shuffle(cv_idx)
    n_val = max(1, int(0.15 * len(cv_idx)))
    val_idx = sorted(cv_idx[:n_val])
    train_idx = sorted(cv_idx[n_val:])

    print(f"  Test ({len(test_idx)}):  {[names[i] for i in test_idx]}")
    print(f"  Train ({len(train_idx)}), Val ({len(val_idx)})")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({
            'test': [names[i] for i in test_idx],
            'train': [names[i] for i in train_idx],
            'val': [names[i] for i in val_idx],
            'seed': args.seed,
            'layers': LAYERS,
        }, f, indent=2)

    # ── Extract Features ──
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
    extractor = WavLMMultiLayerExtractor(device, LAYERS)

    print("\nExtracting pre-surgery features (layers 12-16)...")
    pre_data = extract_all(extractor, pre_files,
                           os.path.join(cache_dir, f'pre_multilayer_{args.surgery.lower()}.pt'))
    print("\nExtracting post-surgery features (layers 12-16)...")
    post_data = extract_all(extractor, post_files,
                            os.path.join(cache_dir, f'post_multilayer_{args.surgery.lower()}.pt'))

    # ── Build Segments ──
    print("\nBuilding training segments...")
    train_segs = build_segments(pre_data, post_data, train_idx)
    print("Building validation segments...")
    val_segs = build_segments(pre_data, post_data, val_idx)

    train_ds = MultiLayerDataset(train_segs, augment=True)
    val_ds = MultiLayerDataset(val_segs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} segs, Val: {len(val_ds)} segs")

    # ── Model ──
    model = MultiLayerMapper(
        n_layers=len(LAYERS), feat_dim=1024,
        hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS, dropout=DROPOUT
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Train ──
    ckpt_path = os.path.join(args.output, 'best_model.pt')
    best_val = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = multi_layer_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
        scheduler.step()
        train_losses.append(np.mean(epoch_loss))

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl.append(multi_layer_loss(model(xb), yb).item())
        val_loss = np.mean(vl)
        val_losses.append(val_loss)

        alphas_str = " ".join([f"{a.item():.3f}" for a in model.alphas])
        print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_losses[-1]:.6f}  val={val_loss:.6f}  "
              f"alphas=[{alphas_str}]  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'alphas': [a.item() for a in model.alphas],
                'layer_attn_weights': F.softmax(model.layer_attn.weights, dim=0).tolist(),
                'config': {
                    'n_layers': len(LAYERS), 'layers': LAYERS,
                    'feat_dim': 1024, 'hidden_dim': HIDDEN_DIM,
                    'n_levels': N_LEVELS, 'dropout': DROPOUT,
                },
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── Plot Training Curves ──
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='train')
    ax.plot(val_losses, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_title('Multi-Layer Quality Mapper Training')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_curve.png'), dpi=150)
    plt.close()

    print(f"\nBest val loss: {best_val:.6f}")

    # ═══ Evaluate ═══
    print(f"\n{'='*70}")
    print(f"  Evaluating on {len(test_idx)} test patients")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiLayerMapper(
        n_layers=len(LAYERS), feat_dim=1024,
        hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS, dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Learned layer attention weights: {ckpt['layer_attn_weights']}")
    print(f"  Learned alphas: {ckpt['alphas']}")

    # For audio reconstruction, we also need layer 6 features + vocoder
    print("\nLoading kNN-VC for vocoding...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

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

    # Evaluate: convert pre→post using the mapper, then vocode layer 6 features
    # Since our mapper predicts layers 12-16 (not layer 6), we use the original
    # layer 6 features for vocoding. The evaluation measures whether the mapped
    # layers 12-16 capture the voice quality shift via feature-space metrics.
    #
    # For audio output, we use knn-vc layer 6 features directly (same as baseline).
    # The mapper's value is in the quality-space representation, not audio reconstruction.

    results_test = []
    results_train = []

    def evaluate_patient(idx, tag="TEST"):
        name = names[idx]
        pre_feats_ml = pre_data[idx][1]   # (n_layers, T, 1024)
        post_feats_ml = post_data[idx][1]  # (n_layers, T, 1024)

        # Map pre → post in layers 12-16
        with torch.no_grad():
            inp = pre_feats_ml.permute(0, 2, 1).unsqueeze(0).to(device)  # (1, 5, 1024, T)
            pred = model(inp).squeeze(0)  # (5, 1024, T)
            pred = pred.permute(0, 2, 1).cpu()  # (5, T, 1024)

        # Feature-space metrics: cosine similarity between predicted and actual post
        sims = []
        for li in range(len(LAYERS)):
            # Average over frames
            pred_avg = pred[li].mean(dim=0)
            post_avg = post_feats_ml[li].mean(dim=0)
            pre_avg = pre_feats_ml[li].mean(dim=0)
            sim_mapped = F.cosine_similarity(pred_avg.unsqueeze(0), post_avg.unsqueeze(0)).item()
            sim_baseline = F.cosine_similarity(pre_avg.unsqueeze(0), post_avg.unsqueeze(0)).item()
            sims.append({'layer': LAYERS[li], 'mapped': sim_mapped, 'baseline': sim_baseline})

        # Audio evaluation: vocode using knn-vc layer 6 (standard pipeline)
        pre_l6 = knn_vc.get_features(pre_files[idx]).cpu()  # (T, 1024)
        with torch.no_grad():
            wav = knn_vc.vocode(pre_l6[None].to(device)).cpu().squeeze()
        out_path = os.path.join(conv_dir, f'{name}_{tag.lower()}.wav')
        torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

        # ECAPA similarity
        emb_conv = get_emb(out_path)
        emb_post = get_emb(post_files[idx])
        emb_pre = get_emb(pre_files[idx])
        spk_sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
        spk_sim_baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()

        result = {
            'name': name,
            'layer_sims': sims,
            'spk_sim_post': spk_sim_post,
            'spk_sim_baseline': spk_sim_baseline,
        }
        avg_mapped = np.mean([s['mapped'] for s in sims])
        avg_baseline = np.mean([s['baseline'] for s in sims])
        print(f"  [{tag}] {name}: "
              f"L12-16 mapped→post={avg_mapped:.3f} (baseline={avg_baseline:.3f}, "
              f"delta={avg_mapped - avg_baseline:+.3f}) | "
              f"SpkSim={spk_sim_post:.3f} (baseline={spk_sim_baseline:.3f})")
        return result

    print("\n--- TEST set ---")
    for i in test_idx:
        results_test.append(evaluate_patient(i, "TEST"))

    print("\n--- TRAIN set (overfit check) ---")
    for i in train_idx:
        results_train.append(evaluate_patient(i, "TRAIN"))

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  Multi-Layer Quality Mapper — {args.surgery} — SUMMARY")
    print(f"{'='*70}")

    for tag, results in [("TEST", results_test), ("TRAIN", results_train)]:
        if not results:
            continue
        avg_mapped = np.mean([[s['mapped'] for s in r['layer_sims']] for r in results])
        avg_baseline = np.mean([[s['baseline'] for s in r['layer_sims']] for r in results])
        avg_spk = np.mean([r['spk_sim_post'] for r in results])
        avg_spk_base = np.mean([r['spk_sim_baseline'] for r in results])
        print(f"  {tag} ({len(results)} patients):")
        print(f"    L12-16 feature sim:  mapped={avg_mapped:.3f}  baseline={avg_baseline:.3f}  delta={avg_mapped-avg_baseline:+.3f}")
        print(f"    ECAPA SpkSim:        conv={avg_spk:.3f}  baseline={avg_spk_base:.3f}")

        # Per-layer breakdown
        for li_idx, layer in enumerate(LAYERS):
            mapped = np.mean([r['layer_sims'][li_idx]['mapped'] for r in results])
            base = np.mean([r['layer_sims'][li_idx]['baseline'] for r in results])
            print(f"      Layer {layer}: mapped={mapped:.3f}  baseline={base:.3f}  delta={mapped-base:+.3f}")

    print(f"{'='*70}")

    all_results = {
        'method': 'Multi-Layer Quality Mapper (L12-16)',
        'surgery': args.surgery,
        'layers': LAYERS,
        'test': results_test,
        'train': results_train,
        'config': ckpt['config'],
        'layer_attn_weights': ckpt['layer_attn_weights'],
        'alphas': ckpt['alphas'],
    }
    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}/results.json")


if __name__ == '__main__':
    main()
