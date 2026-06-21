"""
DLA-VC — Leave-One-Out Cross-Validation

Same model as train.py but with proper patient-level LOO evaluation.
For each patient: train on N-1, evaluate on the held-out one.

Usage:
    python scripts/train_loo.py
    python scripts/train_loo.py --surgery Tonsill
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import json
import random
import numpy as np
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from model.dla_vc import DLAVCModel, DomainClassifier1D, gradient_reversal


# ──────────────────────────────────────────────
# Config (same as train.py)
# ──────────────────────────────────────────────
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"
WAVLM_LAYER_FOR_VOCODER = 6
SAMPLE_RATE = 16000

FEAT_DIM = 1024
CODE_DIM = 64
NUM_CODES = 32
NUM_HEADS = 4
QUALITY_DIM = 64
COMMITMENT_WEIGHT = 0.25
EMA_DECAY = 0.99
ENTROPY_WEIGHT = 0.5
DROPOUT = 0.15
CONTENT_NOISE = 0.1

BATCH_SIZE = 8
EPOCHS = 200  # reduced from 400 since we run N times
LR = 1e-4
LR_ADV = 1e-4
SEGMENT_SAMPLES = 40000
SEGMENT_HOP_SAMPLES = 20000
WARMUP_EPOCHS = 20  # reduced from 30
PATIENCE = 30

LAMBDA_RECON = 5.0
LAMBDA_VQ = 1.0
LAMBDA_ADV = 1.5
LAMBDA_QUAL_CLS = 5.0
LAMBDA_CYCLE = 2.0
LAMBDA_CROSS_RECON = 2.0


# ──────────────────────────────────────────────
# WavLM
# ──────────────────────────────────────────────

class WavLMFeatureExtractor:
    def __init__(self, device):
        from transformers import WavLMModel
        print("Loading WavLM-Large...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract(self, audio_batch):
        outputs = self.model(audio_batch.to(self.device), output_hidden_states=True)
        all_layers = torch.stack(outputs.hidden_states[1:], dim=1)
        all_layers = all_layers.permute(0, 1, 3, 2)
        layer6 = all_layers[:, WAVLM_LAYER_FOR_VOCODER - 1]
        return all_layers, layer6


# ──────────────────────────────────────────────
# Dataset (file-list based, not directory based)
# ──────────────────────────────────────────────

class AudioSegmentDatasetFromFiles(Dataset):
    """Loads audio segments from a specific list of wav files."""
    def __init__(self, wav_files, label, segment_samples=40000,
                 hop_samples=20000, augment=False):
        self.segments = []
        self.label = label
        self.augment = augment

        for wf in wav_files:
            audio, sr = torchaudio.load(wf)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = audio[0]
            T = audio.shape[0]
            if T < segment_samples:
                audio = F.pad(audio, (0, segment_samples - T))
                self.segments.append(audio[:segment_samples])
            else:
                for start in range(0, T - segment_samples + 1, hop_samples):
                    self.segments.append(audio[start:start + segment_samples])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        audio = self.segments[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            audio = audio + torch.randn_like(audio) * 0.002
        return audio, torch.tensor(self.label, dtype=torch.float32)


class PairedDomainLoader:
    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        for _ in range(max(len(self.loader_a), len(self.loader_b))):
            try:
                a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a = next(iter_a)
            try:
                b = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b = next(iter_b)
            min_bs = min(a[0].shape[0], b[0].shape[0])
            yield a[0][:min_bs], a[1][:min_bs], b[0][:min_bs], b[1][:min_bs]

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ──────────────────────────────────────────────
# Training one fold
# ──────────────────────────────────────────────

def train_one_fold(train_pre_files, train_post_files,
                   val_pre_files, val_post_files,
                   wavlm, device, ckpt_path, tag=""):
    """Train DLA-VC on given files. Returns best val loss."""

    # Datasets
    ds_pre = AudioSegmentDatasetFromFiles(
        train_pre_files, label=0, segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_HOP_SAMPLES, augment=True)
    ds_post = AudioSegmentDatasetFromFiles(
        train_post_files, label=1, segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_HOP_SAMPLES, augment=True)

    loader_pre = DataLoader(ds_pre, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=2, pin_memory=True)
    loader_post = DataLoader(ds_post, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    paired_loader = PairedDomainLoader(loader_pre, loader_post)

    # Val
    ds_pre_val = AudioSegmentDatasetFromFiles(
        val_pre_files, label=0, segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_SAMPLES, augment=False)
    ds_post_val = AudioSegmentDatasetFromFiles(
        val_post_files, label=1, segment_samples=SEGMENT_SAMPLES,
        hop_samples=SEGMENT_SAMPLES, augment=False)
    val_loader = DataLoader(ConcatDataset([ds_pre_val, ds_post_val]),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                            pin_memory=True)

    print(f"  {tag} Train: {len(ds_pre)}+{len(ds_post)} segs, Val: {len(ds_pre_val)}+{len(ds_post_val)} segs")

    # Models
    model = DLAVCModel(
        feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
        num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
        num_wavlm_layers=wavlm.num_layers,
        commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
        entropy_weight=ENTROPY_WEIGHT, dropout=DROPOUT,
        content_noise_std=CONTENT_NOISE,
    ).to(device)
    domain_cls = DomainClassifier1D(code_dim=CODE_DIM).to(device)
    quality_cls = nn.Sequential(
        nn.Linear(QUALITY_DIM, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
    ).to(device)

    opt_model = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    opt_adv = torch.optim.Adam(domain_cls.parameters(), lr=LR_ADV)
    opt_qual = torch.optim.Adam(quality_cls.parameters(), lr=LR_ADV)
    sched_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=1e-6)
    sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train(); domain_cls.train(); quality_cls.train()

        for audio_pre, lab_pre, audio_post, lab_post in paired_loader:
            audio_all = torch.cat([audio_pre, audio_post], dim=0).to(device)
            labels = torch.cat([lab_pre, lab_post], dim=0).to(device)
            B_half = audio_pre.shape[0]
            hidden_all, target_all = wavlm.extract(audio_all)

            loss_adv_g = torch.tensor(0.0, device=device)
            loss_qual = torch.tensor(0.0, device=device)
            loss_cycle = torch.tensor(0.0, device=device)
            loss_cross_recon = torch.tensor(0.0, device=device)

            if epoch >= WARMUP_EPOCHS:
                with torch.no_grad():
                    content_q_det, _ = model.encode_content(hidden_all)
                opt_adv.zero_grad()
                adv_pred = domain_cls(content_q_det.detach())
                loss_adv_cls = F.binary_cross_entropy_with_logits(adv_pred.squeeze(1), labels)
                loss_adv_cls.backward()
                opt_adv.step()

            opt_model.zero_grad(); opt_qual.zero_grad()
            recon, vq_loss, perplexity, content_z, quality = model(hidden_all, target_all)
            loss_recon = F.mse_loss(recon, target_all)

            if epoch >= WARMUP_EPOCHS:
                content_reversed = gradient_reversal(content_z, alpha=LAMBDA_ADV)
                adv_pred_gr = domain_cls(content_reversed)
                loss_adv_g = F.binary_cross_entropy_with_logits(adv_pred_gr.squeeze(1), labels)

                qual_pred = quality_cls(quality)
                loss_qual = F.binary_cross_entropy_with_logits(qual_pred.squeeze(1), labels)

                # Cycle + cross-recon
                hidden_pre, hidden_post = hidden_all[:B_half], hidden_all[B_half:]
                target_pre, target_post = target_all[:B_half], target_all[B_half:]

                content_q_pre, skips_pre = model.encode_content(hidden_pre)
                quality_pre = model.encode_quality(hidden_pre)
                content_q_post, skips_post = model.encode_content(hidden_post)
                quality_post = model.encode_quality(hidden_post)

                cross_a2b = model.unet_decoder(content_q_pre, quality_post, skips_pre)
                cross_a2b = model._match_time(cross_a2b, target_pre)
                cross_b2a = model.unet_decoder(content_q_post, quality_pre, skips_post)
                cross_b2a = model._match_time(cross_b2a, target_post)

                re_enc_a2b, _ = model.unet_encoder(cross_a2b)
                re_cq_a2b, _, _ = model.vq(model.content_proj(re_enc_a2b))
                re_q_a2b = model.quality_encoder(cross_a2b)
                re_enc_b2a, _ = model.unet_encoder(cross_b2a)
                re_cq_b2a, _, _ = model.vq(model.content_proj(re_enc_b2a))
                re_q_b2a = model.quality_encoder(cross_b2a)

                loss_cycle = (F.l1_loss(re_cq_a2b, content_q_pre.detach())
                            + F.l1_loss(re_cq_b2a, content_q_post.detach())
                            + F.l1_loss(re_q_a2b, quality_post.detach())
                            + F.l1_loss(re_q_b2a, quality_pre.detach()))

                cross_qp_a2b = quality_cls(re_q_a2b)
                cross_qp_b2a = quality_cls(re_q_b2a)
                loss_cross_recon = (
                    F.binary_cross_entropy_with_logits(cross_qp_a2b.squeeze(1), torch.ones(B_half, device=device))
                    + F.binary_cross_entropy_with_logits(cross_qp_b2a.squeeze(1), torch.zeros(B_half, device=device)))

            if epoch < WARMUP_EPOCHS:
                loss_total = LAMBDA_RECON * loss_recon + LAMBDA_VQ * vq_loss
            else:
                loss_total = (LAMBDA_RECON * loss_recon + LAMBDA_VQ * vq_loss
                             + loss_adv_g + LAMBDA_QUAL_CLS * loss_qual
                             + LAMBDA_CYCLE * loss_cycle + LAMBDA_CROSS_RECON * loss_cross_recon)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step(); opt_qual.step()

        sched_model.step(); sched_adv.step()

        # Validate
        model.eval()
        val_recon = 0; n_val = 0
        with torch.no_grad():
            for audio_v, _ in val_loader:
                audio_v = audio_v.to(device)
                hidden_v, target_v = wavlm.extract(audio_v)
                recon_v, _, _, _, _ = model(hidden_v, target_v)
                val_recon += F.mse_loss(recon_v, target_v).item()
                n_val += 1
        avg_val = val_recon / max(n_val, 1)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    {tag} Epoch {epoch+1:3d}  val_recon={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0

            # Compute avg quality vectors from training files
            avg_q_pre = compute_avg_quality_from_files(model, wavlm, train_pre_files, device)
            avg_q_post = compute_avg_quality_from_files(model, wavlm, train_post_files, device)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'domain_cls': domain_cls.state_dict(),
                'quality_cls': quality_cls.state_dict(),
                'avg_quality_pre': avg_q_pre,
                'avg_quality_post': avg_q_post,
                'adapter_weights': model.get_adapter_weights(),
                'val_loss': avg_val,
                'config': {
                    'feat_dim': FEAT_DIM, 'code_dim': CODE_DIM,
                    'num_codes': NUM_CODES, 'num_heads': NUM_HEADS,
                    'quality_dim': QUALITY_DIM,
                    'num_wavlm_layers': wavlm.num_layers,
                },
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    {tag} Early stop at epoch {epoch+1}")
                break

    print(f"    {tag} Best val: {best_val_loss:.4f}")
    return best_val_loss, model


def compute_avg_quality_from_files(model, wavlm, wav_files, device):
    model.eval()
    qualities = []
    for wf in wav_files:
        audio, sr = torchaudio.load(wf)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio[0].unsqueeze(0).to(device)
        with torch.no_grad():
            hidden, _ = wavlm.extract(audio)
            q = model.encode_quality(hidden)
        qualities.append(q.cpu())
    return torch.cat(qualities, dim=0).mean(dim=0)


def convert_and_evaluate(knn_vc, model, wavlm, pre_wav, post_wav,
                          avg_q_post, output_path, device, ecapa):
    """Convert one patient and compute SpkSim."""
    # Load and extract
    audio, sr = torchaudio.load(pre_wav)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio[0].unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, _ = wavlm.extract(audio)
        converted = model.convert(hidden, avg_q_post.unsqueeze(0).to(device))

    # Vocode
    converted_t = converted.squeeze(0).t()  # (T, 1024)
    out_wav = knn_vc.vocode(converted_t[None]).cpu().squeeze()
    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)

    # SpkSim
    def get_emb(wav_path):
        sig, sr2 = torchaudio.load(wav_path)
        if sr2 != 16000:
            sig = torchaudio.functional.resample(sig, sr2, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return ecapa.encode_batch(sig).squeeze()

    emb_conv = get_emb(output_path)
    emb_post = get_emb(post_wav)
    emb_pre = get_emb(pre_wav)

    sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
    sim_pre = F.cosine_similarity(emb_conv.unsqueeze(0), emb_pre.unsqueeze(0)).item()
    baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()
    return sim_post, sim_pre, baseline


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DLA-VC — LOO evaluation")
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'results_loo'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n_patients = len(pre_files)
    names = [Path(f).stem for f in pre_files]

    print(f"\n{args.surgery}: {n_patients} patients, LOO evaluation")

    # Load WavLM
    wavlm = WavLMFeatureExtractor(device)

    # Load kNN-VC for vocoding
    print("Loading kNN-VC for vocoding...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load ECAPA-TDNN
    print("Loading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )

    conv_dir = os.path.join(args.output, 'converted')
    os.makedirs(conv_dir, exist_ok=True)

    all_sim_post = []
    all_sim_pre = []
    all_baseline = []

    for test_idx in range(n_patients):
        test_name = names[test_idx]
        train_idx = [i for i in range(n_patients) if i != test_idx]

        # Use 3 random patients as val for early stopping
        np.random.seed(args.seed + test_idx)
        val_idx = list(np.random.choice(train_idx, size=min(3, len(train_idx) - 1), replace=False))
        pure_train = [i for i in train_idx if i not in val_idx]

        train_pre = [pre_files[i] for i in pure_train]
        train_post = [post_files[i] for i in pure_train]
        val_pre = [pre_files[i] for i in val_idx]
        val_post = [post_files[i] for i in val_idx]

        print(f"\n--- LOO {test_idx+1}/{n_patients}: test={test_name} ---")

        ckpt_path = os.path.join(args.output, f'loo_{test_idx}_model.pth')
        best_val, model = train_one_fold(
            train_pre, train_post, val_pre, val_post,
            wavlm, device, ckpt_path, tag=f"LOO-{test_idx+1}")

        # Load best model
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_eval = DLAVCModel(
            feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
            num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
            num_wavlm_layers=wavlm.num_layers,
            commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
            entropy_weight=ENTROPY_WEIGHT, dropout=0.0,
            content_noise_std=0.0,
        ).to(device)
        model_eval.load_state_dict(ckpt['model'])
        model_eval.eval()
        avg_q_post = ckpt['avg_quality_post']

        out_path = os.path.join(conv_dir, names[test_idx] + '.wav')
        sim_post, sim_pre, baseline = convert_and_evaluate(
            knn_vc, model_eval, wavlm, pre_files[test_idx], post_files[test_idx],
            avg_q_post, out_path, device, ecapa)

        all_sim_post.append(sim_post)
        all_sim_pre.append(sim_pre)
        all_baseline.append(baseline)

        print(f"    {test_name}: conv->post={sim_post:.3f}  conv->pre={sim_pre:.3f}  "
              f"baseline={baseline:.3f}  delta={sim_post-baseline:+.3f}")

        # Cleanup checkpoint
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # Summary
    print(f"\n{'='*70}")
    print(f"  DLA-VC LOO Results — {args.surgery} ({n_patients} patients)")
    print(f"{'='*70}")
    print(f"  Baseline (no conversion):   {np.mean(all_baseline):.3f} +/- {np.std(all_baseline):.3f}")
    print(f"  DLA-VC (conv vs post):      {np.mean(all_sim_post):.3f} +/- {np.std(all_sim_post):.3f}")
    print(f"  DLA-VC (conv vs pre):       {np.mean(all_sim_pre):.3f} +/- {np.std(all_sim_pre):.3f}")
    print(f"  Improvement over baseline:  {np.mean(all_sim_post) - np.mean(all_baseline):+.3f}")
    print(f"{'='*70}")

    print(f"\nPer-patient:")
    print(f"{'Patient':<35} {'Baseline':>8} {'Conv->Post':>10} {'Conv->Pre':>9} {'Delta':>7}")
    print("-" * 75)
    for i in range(n_patients):
        delta = all_sim_post[i] - all_baseline[i]
        print(f"  {names[i]:<33} {all_baseline[i]:>8.3f} {all_sim_post[i]:>10.3f} "
              f"{all_sim_pre[i]:>9.3f} {delta:>+7.3f}")
    n_improved = sum(1 for i in range(n_patients) if all_sim_post[i] > all_baseline[i])
    print(f"\nImproved: {n_improved}/{n_patients} ({100*n_improved/n_patients:.0f}%)")

    # Save
    results = {
        'method': 'DLA-VC', 'surgery': args.surgery, 'n_patients': n_patients,
        'baseline_mean': float(np.mean(all_baseline)),
        'conv_vs_post_mean': float(np.mean(all_sim_post)),
        'conv_vs_post_std': float(np.std(all_sim_post)),
        'per_patient': [
            {'name': names[i], 'baseline': all_baseline[i],
             'conv_vs_post': all_sim_post[i], 'conv_vs_pre': all_sim_pre[i]}
            for i in range(n_patients)
        ]
    }
    with open(os.path.join(args.output, 'loo_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}/loo_results.json")


if __name__ == '__main__':
    main()
