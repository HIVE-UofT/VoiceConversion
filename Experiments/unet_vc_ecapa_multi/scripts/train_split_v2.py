"""
UNet-VC Content + ECAPA Speaker Loss — Multi-Surgery v2

Same architecture and losses as the single-surgery unet_vc_ecapa/v2, but trained
on the combined training pools of Tonsill + Fess + Sept. Per-surgery held-out
test patients (5 each, 15 total) are excluded from train and val; the
remaining patients are split per surgery (~15% val, rest train) so each
surgery contributes to both splits.

Usage:
    python scripts/train_split_v2.py
    python scripts/train_split_v2.py --surgeries Tonsill,Fess,Sept --output results_multi_v2
"""

import argparse
import os
import sys
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

# Hardcoded per-surgery test patient IDs (held out from train + val).
TEST_PATIENTS_BY_SURGERY = {
    "Tonsill": ["0045", "0085", "0110", "0122", "0132"],
    "Sept":    ["0023", "0033", "0044", "0076", "0077"],
    "Fess":    ["0030", "0046", "0086", "0117", "0123"],
}

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

# Loss weights
CONTENT_WEIGHT = 1.0
ECAPA_WEIGHT = 1.0
ECAPA_UTTS_PER_STEP = 4


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
    B, C, T = pred.shape
    p = pred.permute(0, 2, 1).reshape(B * T, C)
    t = target.permute(0, 2, 1).reshape(B * T, C)
    cos = F.cosine_similarity(p, t, dim=1)
    return (1.0 - cos).mean()


def ecapa_embed_differentiable(ecapa, wav):
    wav_lens = torch.ones(1, device=wav.device)
    feats = ecapa.mods.compute_features(wav)
    feats = ecapa.mods.mean_var_norm(feats, wav_lens)
    emb = ecapa.mods.embedding_model(feats, wav_lens)
    return emb


def vocode(hifigan, feats):
    return hifigan(feats).squeeze(1)


STYLE_CROP_LEN = 128

def ecapa_style_loss(model, hifigan, ecapa, pre_feats_list, pre_ecapa_embs,
                     train_idx, post_ecapa_embs, device, n_utts=ECAPA_UTTS_PER_STEP):
    losses = []
    chosen = random.sample(train_idx, min(n_utts, len(train_idx)))
    for utt_idx in chosen:
        full_feats = pre_feats_list[utt_idx].to(device)
        T = full_feats.shape[0]
        crop_len = min(T, STYLE_CROP_LEN)
        start = random.randint(0, max(0, T - crop_len))
        feats = full_feats[start:start + crop_len]
        spk_emb = pre_ecapa_embs[utt_idx].unsqueeze(0).to(device)
        out = model(feats.t().unsqueeze(0), spk_emb)
        audio = vocode(hifigan, out.transpose(1, 2))
        emb = ecapa_embed_differentiable(ecapa, audio).squeeze()
        target = post_ecapa_embs[utt_idx].to(device)
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
            spk_emb = pre_ecapa_embs[i].unsqueeze(0).to(device)
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
    parser.add_argument('--surgeries', type=str, default='Tonsill,Fess,Sept',
                        help='Comma-separated list of surgeries to train on')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    surgeries = [s.strip() for s in args.surgeries.split(',') if s.strip()]
    for s in surgeries:
        if s not in TEST_PATIENTS_BY_SURGERY:
            raise ValueError(f"No test-patient list defined for surgery '{s}'. "
                             f"Known: {list(TEST_PATIENTS_BY_SURGERY)}")

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), '..', 'results_multi_v2')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from utils import get_all_audio_pairs

    random.seed(args.seed)

    # ── Collect train/val data per surgery, stratified split ──
    pre_files, post_files = [], []
    pid_of_file, surgery_of_file = [], []
    train_idx, val_idx = [], []
    per_surgery_summary = {}

    for surg in surgeries:
        test_ids = set(TEST_PATIENTS_BY_SURGERY[surg])
        patient_pairs = get_all_audio_pairs(surg, exclude=test_ids)
        all_pids = sorted(patient_pairs.keys())
        # Sanity: ensure no test patient leaked through.
        assert not (set(all_pids) & test_ids), \
            f"Test patient leaked into {surg} pool: {set(all_pids) & test_ids}"

        shuffled = all_pids.copy()
        random.shuffle(shuffled)
        n_val_pids = max(1, int(0.15 * len(shuffled)))
        val_pids = set(shuffled[:n_val_pids])
        train_pids = set(shuffled[n_val_pids:])

        n_files_surg = 0
        for pid in all_pids:
            for pre, post in patient_pairs[pid]:
                idx = len(pre_files)
                pre_files.append(pre)
                post_files.append(post)
                pid_of_file.append(pid)
                surgery_of_file.append(surg)
                if pid in train_pids:
                    train_idx.append(idx)
                elif pid in val_pids:
                    val_idx.append(idx)
                n_files_surg += 1

        per_surgery_summary[surg] = {
            'patients_total': len(all_pids),
            'train_patients': sorted(train_pids),
            'val_patients': sorted(val_pids),
            'test_patients': sorted(test_ids),
            'n_files': n_files_surg,
        }
        print(f"\n[{surg}] {len(all_pids)} train/val patients, {n_files_surg} file pairs "
              f"(+ {len(test_ids)} test patients held out)")
        print(f"  train_pids ({len(train_pids)}): {sorted(train_pids)}")
        print(f"  val_pids   ({len(val_pids)}): {sorted(val_pids)}")
        print(f"  test_pids  ({len(test_ids)}): {sorted(test_ids)}")

    n = len(pre_files)
    # Correctness checks
    all_test_ids = set().union(*[set(v) for v in TEST_PATIENTS_BY_SURGERY.values()])
    assert not (set(pid_of_file) & all_test_ids), \
        f"Test patient leaked into train/val: {set(pid_of_file) & all_test_ids}"

    total_train_val_patients = sum(s['patients_total'] for s in per_surgery_summary.values())
    assert len(set(pid_of_file)) == total_train_val_patients, \
        f"Patient-count mismatch: {len(set(pid_of_file))} unique pids but " \
        f"{total_train_val_patients} expected"

    print(f"\n{'='*70}")
    print(f"  Combined: {n} file pairs across {len(set(pid_of_file))} patients "
          f"(from {len(surgeries)} surgeries)")
    print(f"  Train: {len(train_idx)} files, "
          f"{len(set(pid_of_file[i] for i in train_idx))} patients")
    print(f"  Val:   {len(val_idx)} files, "
          f"{len(set(pid_of_file[i] for i in val_idx))} patients")
    print(f"{'='*70}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({
            'surgeries': surgeries,
            'test_patients_by_surgery': TEST_PATIENTS_BY_SURGERY,
            'per_surgery': per_surgery_summary,
            'n_files': n,
            'seed': args.seed,
        }, f, indent=2)

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

    # ── Pre-compute pre-surgery ECAPA embeddings ──
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

    # ── Pre-compute post-surgery ECAPA embeddings ──
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
            xb = xb.to(device)

            spk_emb_batch = torch.stack(
                [pre_ecapa_embs[i.item()] for i in utt_indices]
            ).to(device)

            pred = model(xb, spk_emb_batch)

            c_loss = CONTENT_WEIGHT * cosine_content_loss(pred, xb)

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
                'surgeries': surgeries,
            }, ckpt_path)
            tqdm.write(f"  -> Saved best (val_ecapa={val_ecapa:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                tqdm.write(f"  Early stop at epoch {epoch}")
                break

    print(f"\nBest val ECAPA similarity: {best_val_ecapa:.4f}")
    print("Training complete. Run scripts/run_eval.py for test-set evaluation.")


if __name__ == '__main__':
    main()
