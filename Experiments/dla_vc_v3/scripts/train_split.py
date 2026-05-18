"""
DLA-VC — Train/Test Split with ECAPA Speaker Loss

Hold out N_TEST patients, train on the rest, evaluate on test set.
Replaces binary quality classification with ECAPA speaker similarity:
  - Content: reconstruction loss (MSE with input WavLM layer 6)
  - Speaker: ECAPA similarity on cross-reconstructed output vs post

Usage:
    python scripts/train_split.py --surgery Tonsill --n_test 5
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

from model.dla_vc import DLAVCModel

CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"
WAVLM_LAYER_FOR_VOCODER = 6
SAMPLE_RATE = 16000

FEAT_DIM = 1024
CODE_DIM = 128             # bumped 64→128 (16-d per head × 8 heads)
NUM_CODES = 32
NUM_HEADS = 8              # bumped 4→8 (effective codebook = 32^8)
QUALITY_DIM = 192          # bumped 64→192 (match ECAPA speaker embedding dim)
COMMITMENT_WEIGHT = 0.25
EMA_DECAY = 0.99
ENTROPY_WEIGHT = 0.5
DROPOUT = 0.15
CONTENT_NOISE = 0.1

BATCH_SIZE = 4             # dropped 8→4 to reduce memory pressure (avoids OOM at later epochs)
EPOCHS = 400
LR = 3e-4                  # bumped 1e-4→3e-4 (matches working UNet-VC baselines)
SEGMENT_SAMPLES = 40000
SEGMENT_HOP_SAMPLES = 20000
WARMUP_EPOCHS = 0          # v2: residual-output is active; conv_loss must
                           # drive alpha away from 0 from epoch 1 (otherwise
                           # alpha collapses; conv_loss is the only non-degenerate
                           # gradient signal on alpha).
PATIENCE = 40

LAMBDA_RECON = 2.0         # v3b: cut 5.0 -> 2.0. v3 logs showed model regressed
                           # to identity (val ECAPA bottomed at ep16=0.508, then
                           # crept back to 0.61 by ep150) — recon dominated.
LAMBDA_RECON_PHASE2 = 2.0  # Same as LAMBDA_RECON (WARMUP_EPOCHS=0 so symmetric).
LAMBDA_CONTENT_CYCLE = 0.0 # v2: DROPPED. Was 1.0. On 23 patients the content-cycle
                           # regulariser fights the conv signal more than it helps.
LAMBDA_VQ = 1.0
LAMBDA_CYCLE = 0.0         # v2: DROPPED. Was 2.0. Cross-domain cycle wasted
                           # gradient capacity on 23 patients.
LAMBDA_CONV = 10.0         # v2: bumped from 5.0 — conv loss is now the primary
                           # supervision signal, deserves to dominate.
LAMBDA_Q_SHIFT = 1.0       # v2: lowered from 2.0 — kept (enables per-patient
                           # post-quality prediction at inference) but smaller.

# v3 additions: distill from a frozen UNet-VC-ECAPA teacher + light ECAPA loss.
LAMBDA_KD = 5.0            # Knowledge distillation: DLA-VC output must match
                           # the frozen UNet-VC-ECAPA teacher's output on the
                           # same pre features. Strong supervision signal.
TEACHER_CKPT = '/lustre06/project/6086959/sepharfi/VoiceConversion/Experiments/unet_vc_ecapa/results_tonsill_v2/best_model.pt'

ECAPA_EVERY = 10           # v3: re-enabled (was disabled at 999999). Less
                           # frequent than UNet-VC-ECAPA (which uses 3) to
                           # reduce per-step noise.
LAMBDA_ECAPA = 1.0         # v3: dropped 3.0→1.0 — KD is the main extra signal.
ECAPA_STOP_THRESH = 0.25   # Don't early-stop until ECAPA loss drops below this
MIN_EPOCHS = 150           # Minimum epochs before early stopping is allowed
STYLE_CROP_LEN = 128       # WavLM frames fed to HiFiGAN for ECAPA loss (~0.8 s)
                           # Full utterances OOM on 20GB — all HiFiGAN activations
                           # must be kept for backprop when input requires_grad=True.
MAX_WAVLM_SAMPLES = 48000  # Audio samples fed to WavLM for ECAPA loss (3 s).
                           # Longer recordings OOM in WavLM attention layers even
                           # under no_grad, because attention is O(T^2) in memory.


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
        all_layers = torch.stack(outputs.hidden_states[1:], dim=1).permute(0, 1, 3, 2)
        layer6 = all_layers[:, WAVLM_LAYER_FOR_VOCODER - 1]
        return all_layers, layer6


class AudioSegmentDatasetFromFiles(Dataset):
    """Single-domain segment dataset (used for val set only)."""
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


def _build_audio_augmenter_dla():
    """Audio-domain augmentation chain (audiomentations). Same params applied
    to pre and post within a pair preserves the surgery direction in feature
    space, just shifted in pitch/time/gain."""
    from audiomentations import (Compose, PitchShift, TimeStretch,
                                  AddGaussianNoise, Gain)
    return Compose([
        PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.7),
        TimeStretch(min_rate=0.92, max_rate=1.08, leave_length_unchanged=True, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
        Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.4),
    ])


class PatientPairedSegmentDataset(Dataset):
    """Same-patient (pre, post) segment pairs — same file index, same utterance.
    Each item returns (pre_segment, post_segment) from the same file pair, so
    both come from the same patient saying the same thing. Time alignment within
    segments is approximate (not DTW-aligned); kNN frame-pairing in the training
    loop handles fine-grained alignment.

    When augment=True, each __getitem__ call samples a SINGLE random transform
    (pitch shift, time stretch, gain, noise) and applies it to both pre and
    post with the same parameters, so the surgery direction is preserved while
    each epoch sees a different acoustic variant of every pair (more effective
    data per parameter update)."""
    def __init__(self, pre_files, post_files, segment_samples=40000,
                 hop_samples=20000, augment=False):
        assert len(pre_files) == len(post_files)
        self.items = []
        self.augment = augment
        # Build augmenter once — re-seeded per __getitem__ call to ensure the
        # SAME random transform is applied to both pre and post in a pair.
        self.audio_augmenter = _build_audio_augmenter_dla() if augment else None
        for pre_path, post_path in zip(pre_files, post_files):
            pre_audio, sr = torchaudio.load(pre_path)
            if sr != SAMPLE_RATE:
                pre_audio = torchaudio.functional.resample(pre_audio, sr, SAMPLE_RATE)
            pre_audio = pre_audio[0]
            post_audio, sr = torchaudio.load(post_path)
            if sr != SAMPLE_RATE:
                post_audio = torchaudio.functional.resample(post_audio, sr, SAMPLE_RATE)
            post_audio = post_audio[0]

            if pre_audio.shape[0] < segment_samples:
                pre_audio = F.pad(pre_audio, (0, segment_samples - pre_audio.shape[0]))
            if post_audio.shape[0] < segment_samples:
                post_audio = F.pad(post_audio, (0, segment_samples - post_audio.shape[0]))

            pre_segs = [pre_audio[s:s + segment_samples]
                        for s in range(0, pre_audio.shape[0] - segment_samples + 1, hop_samples)]
            post_segs = [post_audio[s:s + segment_samples]
                         for s in range(0, post_audio.shape[0] - segment_samples + 1, hop_samples)]
            if not pre_segs:
                pre_segs = [pre_audio[:segment_samples]]
            if not post_segs:
                post_segs = [post_audio[:segment_samples]]
            # Pair segment i of pre with segment i of post (same file, roughly
            # same time window — kNN frame-pairing handles finer misalignment).
            n_pairs = min(len(pre_segs), len(post_segs))
            for k in range(n_pairs):
                self.items.append((pre_segs[k], post_segs[k]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pre, post = self.items[idx]
        if self.augment and self.audio_augmenter is not None:
            # Apply the SAME random transform to pre and post by reseeding
            # numpy/python RNG between the two calls. audiomentations uses
            # numpy.random under the hood.
            seed = random.randint(0, 2**31 - 1)
            pre_np  = pre.numpy().astype('float32')
            post_np = post.numpy().astype('float32')
            random.seed(seed); np.random.seed(seed)
            pre_np  = self.audio_augmenter(samples=pre_np,  sample_rate=SAMPLE_RATE)
            random.seed(seed); np.random.seed(seed)
            post_np = self.audio_augmenter(samples=post_np, sample_rate=SAMPLE_RATE)
            pre  = torch.from_numpy(pre_np[:pre.shape[0]])     # truncate to original length
            post = torch.from_numpy(post_np[:post.shape[0]])
            # Tiny extra Gaussian noise to keep some independence between pre/post
            if torch.rand(1).item() > 0.5:
                pre  = pre  + torch.randn_like(pre)  * 0.002
            if torch.rand(1).item() > 0.5:
                post = post + torch.randn_like(post) * 0.002
        return pre, post


def knn_pair_frames(X, Y):
    """For each frame of X, find nearest frame of Y by cosine (same batch item).
    X, Y: (B, C, T). Returns Y_paired (B, C, T) where Y_paired[b, :, t] is the
    frame of Y[b] closest to X[b, :, t]."""
    X_norm = F.normalize(X, dim=1)
    Y_norm = F.normalize(Y, dim=1)
    sim = torch.einsum('bct,bcs->bts', X_norm, Y_norm)  # (B, T_X, T_Y)
    idx = sim.argmax(dim=-1)                            # (B, T_X)
    idx_expanded = idx.unsqueeze(1).expand(-1, Y.shape[1], -1)  # (B, C, T_X)
    return torch.gather(Y, 2, idx_expanded)


def compute_avg_quality_from_files(model, wavlm, wav_files, device):
    model.eval()
    qualities = []
    for wf in wav_files:
        audio, sr = torchaudio.load(wf)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio[0].unsqueeze(0)
        if audio.shape[-1] > MAX_WAVLM_SAMPLES:
            audio = audio[:, :MAX_WAVLM_SAMPLES]
        audio = audio.to(device)
        with torch.no_grad():
            hidden, _ = wavlm.extract(audio)
            q = model.encode_quality(hidden)
        qualities.append(q.cpu())
    return torch.cat(qualities, dim=0).mean(dim=0)


def compute_ecapa_val(model, wavlm, vocoder, ecapa, pre_files, val_idx,
                      post_ecapa_embs, avg_q_post, device):
    """Validation by ECAPA similarity: convert val pre files and compare ECAPA
    embedding to corresponding val post audio.
    Uses q_shift(encode_quality(pre)) for per-patient post quality prediction
    (falls back to avg_q_post if q_shift isn't trained yet).
    Returns 1 - mean cosine similarity (lower = better)."""
    model.eval()
    sims = []
    with torch.no_grad():
        for i in val_idx:
            audio, sr = torchaudio.load(pre_files[i])
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = audio[0].unsqueeze(0)
            if audio.shape[-1] > MAX_WAVLM_SAMPLES:
                audio = audio[:, :MAX_WAVLM_SAMPLES]
            audio = audio.to(device)
            hidden, _ = wavlm.extract(audio)
            # Predict this patient's post quality from their pre audio
            q_post_pred = model.predict_post_quality(hidden)   # (1, Q)
            converted = model.convert(hidden, q_post_pred)     # (1, 1024, T)
            audio_conv = vocoder(converted.transpose(1, 2)).squeeze(1)
            wav_lens = torch.ones(1, device=device)
            feats = ecapa.mods.compute_features(audio_conv)
            feats = ecapa.mods.mean_var_norm(feats, wav_lens)
            emb = ecapa.mods.embedding_model(feats, wav_lens).squeeze()
            target_emb = post_ecapa_embs[i]
            sim = F.cosine_similarity(emb.unsqueeze(0),
                                       target_emb.unsqueeze(0)).item()
            sims.append(sim)
    return 1.0 - float(np.mean(sims))


def ecapa_embed_differentiable(ecapa, wav):
    """Differentiable ECAPA embedding: wav (1, T) -> emb (1, 1, 192)."""
    wav_lens = torch.ones(1, device=wav.device)
    feats = ecapa.mods.compute_features(wav)
    feats = ecapa.mods.mean_var_norm(feats, wav_lens)
    return ecapa.mods.embedding_model(feats, wav_lens)


def vocode_differentiable(hifigan, c):
    """Differentiable vocoding: c (1, T, 1024) -> audio (1, audio_len)."""
    return hifigan(c).squeeze(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--n_test', type=int, default=5)
    parser.add_argument('--test_patients', type=str,
                        default="0085,0110,0122,0132,0045",
                        help='Comma-separated fixed test patient IDs (overrides --n_test)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..',
                                    f'results_{args.surgery.lower()}_split')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
    from utils import get_all_audio_pairs

    fixed_ids = set(p.strip() for p in args.test_patients.split(',') if p.strip()) \
                if args.test_patients else set()

    # Collect all audio types (Speech + TDU + Vowels + Sustained vowels), excluding test patients
    patient_pairs = get_all_audio_pairs(args.surgery, exclude=fixed_ids)
    all_pids = sorted(patient_pairs.keys())

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    print(f"\n{args.surgery}: {len(all_pids)} train/val patients, {n} total files")
    print(f"  Train: {len(train_pids)} patients, {len(train_idx)} files")
    print(f"  Val:   {len(val_pids)} patients, {len(val_idx)} files")
    print(f"  Test:  held out: {sorted(fixed_ids)}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': sorted(fixed_ids), 'train': sorted(train_pids),
                   'val': sorted(val_pids), 'n_files': n,
                   'seed': args.seed}, f, indent=2)

    # WavLM
    wavlm = WavLMFeatureExtractor(device)

    # Datasets — train only on train patients
    train_pre  = [pre_files[i] for i in train_idx]
    train_post = [post_files[i] for i in train_idx]
    val_pre    = [pre_files[i] for i in val_idx]
    val_post   = [post_files[i] for i in val_idx]

    # Patient-paired segments: each batch item is (pre_seg, post_seg) from the
    # SAME file (same patient, same utterance). Enables direct conversion loss.
    # v2: audio augmentation OFF (it didn't help on UNet-VC; tiny feature-level
    # noise inside the dataset is still applied).
    paired_ds = PatientPairedSegmentDataset(train_pre, train_post,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_HOP_SAMPLES, augment=False)
    print(f"  Train: {len(paired_ds)} same-patient (pre, post) segment pairs")

    paired_loader = DataLoader(paired_ds, batch_size=BATCH_SIZE, shuffle=True,
                               drop_last=True, num_workers=2, pin_memory=True)

    ds_val_pre = AudioSegmentDatasetFromFiles(val_pre, label=0,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES, augment=False)
    ds_val_post = AudioSegmentDatasetFromFiles(val_post, label=1,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES, augment=False)
    val_loader = DataLoader(ConcatDataset([ds_val_pre, ds_val_post]),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  Val: {len(ds_val_pre)} + {len(ds_val_post)} segments")

    # Model
    model = DLAVCModel(
        feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
        num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
        num_wavlm_layers=wavlm.num_layers,
        commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
        entropy_weight=ENTROPY_WEIGHT, dropout=DROPOUT,
        content_noise_std=CONTENT_NOISE,
    ).to(device)
    print(f"DLA-VC params: {model.count_parameters():,}")

    # v3: load frozen UNet-VC-ECAPA teacher for knowledge distillation.
    # Use importlib to load the teacher's unet.py from an explicit absolute
    # path (avoids name collision with this experiment's `model/dla_vc.py`).
    print(f"\n[v3] Loading frozen UNet-VC-ECAPA teacher from {TEACHER_CKPT}...")
    import importlib.util
    _teacher_module_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'unet_vc_ecapa', 'model', 'unet.py')
    _spec = importlib.util.spec_from_file_location("teacher_unet", _teacher_module_path)
    _teacher_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_teacher_mod)
    _TeacherNet = _teacher_mod.ResUNet1D
    _teacher_ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    _tcfg = _teacher_ckpt.get('config', {}) or {}
    teacher = _TeacherNet(
        feat_dim=_tcfg.get('feat_dim', 1024),
        hidden_dim=_tcfg.get('hidden_dim', 64),
        n_levels=_tcfg.get('n_levels', 2),
        dropout=0.0,
    ).to(device).eval()
    # Checkpoint may store under 'model_state_dict' or 'model'
    _state = _teacher_ckpt.get('model_state_dict') or _teacher_ckpt.get('model') or _teacher_ckpt
    teacher.load_state_dict(_state, strict=False)
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[v3] Teacher loaded "
          f"(val_loss={_teacher_ckpt.get('val_loss','?')}, "
          f"alpha={_teacher_ckpt.get('alpha','?')}).")

    # ECAPA + vocoder for speaker loss
    print("Loading kNN-VC vocoder...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)
    vocoder = knn_vc.hifigan
    vocoder.eval()
    for p in vocoder.parameters():
        p.requires_grad = False

    print("Loading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    for p in ecapa.mods.parameters():
        p.requires_grad = False

    # Pre-compute post-surgery ECAPA embeddings
    print("Pre-computing post-surgery ECAPA embeddings...")
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

    opt_model = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=1e-6)

    ckpt_path = os.path.join(args.output, 'best_model.pth')
    best_val = float('inf')
    patience_counter = 0
    global_step = 0

    # ═══ Training ═══
    for epoch in range(EPOCHS):
        model.train()
        ecapa_losses = []
        torch.cuda.empty_cache()  # defragment before each epoch's training

        for audio_pre, audio_post in paired_loader:
            # audio_pre[b] and audio_post[b] are from the SAME patient/file.
            audio_all = torch.cat([audio_pre, audio_post], dim=0).to(device)
            B_half = audio_pre.shape[0]
            hidden_all, target_all = wavlm.extract(audio_all)

            loss_cycle = torch.tensor(0.0, device=device)
            loss_conv = torch.tensor(0.0, device=device)
            loss_q_shift = torch.tensor(0.0, device=device)
            loss_kd = torch.tensor(0.0, device=device)

            opt_model.zero_grad()
            recon, vq_loss, perp, content_z, quality = model(hidden_all, target_all)

            # Direct feature reconstruction: decoder output must match WavLM layer-6
            # features, otherwise HiFi-GAN (which expects that distribution) produces garbage.
            loss_feat_mse = F.mse_loss(recon, target_all)
            loss_feat_cos = 1.0 - F.cosine_similarity(recon, target_all, dim=1).mean()
            loss_recon = loss_feat_mse + 0.5 * loss_feat_cos

            # Content cycle consistency: re-extract content from output, compare
            recon_enc_out, _ = model.unet_encoder(recon)
            recon_content = model.content_proj(recon_enc_out)
            loss_content_cycle = F.mse_loss(recon_content, content_z.detach())

            if epoch >= WARMUP_EPOCHS:
                h_pre, h_post = hidden_all[:B_half], hidden_all[B_half:]
                t_pre, t_post = target_all[:B_half], target_all[B_half:]
                cq_pre, sk_pre = model.encode_content(h_pre)
                cq_post, sk_post = model.encode_content(h_post)
                q_pre = model.encode_quality(h_pre)
                q_post = model.encode_quality(h_post)

                # v2: residual-output mode. The decoder predicts a delta that
                # is added (scaled by alpha) to the source-domain anchor. This
                # is the UNet-VC residual trick — the decoder only has to learn
                # the shift between domains, not regenerate features from scratch.
                delta_a2b = model._match_time(model.unet_decoder(cq_pre, q_post, sk_pre), t_pre)
                delta_b2a = model._match_time(model.unet_decoder(cq_post, q_pre, sk_post), t_post)
                if getattr(model, 'use_residual_output', False):
                    x_a2b = t_pre  + model.alpha * delta_a2b
                    x_b2a = t_post + model.alpha * delta_b2a
                else:
                    x_a2b, x_b2a = delta_a2b, delta_b2a

                # Direct conversion supervision: pre→post output should match the
                # real post features of the same patient. pre/post aren't temporally
                # aligned, so kNN-pair each pre frame to its content-matched post
                # frame by WavLM cosine (this is the same trick UNet-VC uses).
                # Pairing on the INPUT features (not the model output) gives a
                # stable content-driven target that doesn't depend on current
                # model quality.
                with torch.no_grad():
                    t_post_paired = knn_pair_frames(t_pre,  t_post)
                    t_pre_paired  = knn_pair_frames(t_post, t_pre)
                loss_conv_a2b = (F.mse_loss(x_a2b, t_post_paired) + 0.5 *
                                 (1.0 - F.cosine_similarity(x_a2b, t_post_paired, dim=1).mean()))
                loss_conv_b2a = (F.mse_loss(x_b2a, t_pre_paired) + 0.5 *
                                 (1.0 - F.cosine_similarity(x_b2a, t_pre_paired, dim=1).mean()))
                loss_conv = 0.5 * (loss_conv_a2b + loss_conv_b2a)

                # v3: knowledge distillation from frozen UNet-VC-ECAPA teacher.
                # The teacher converts t_pre → predicted_post directly via its own
                # residual U-Net. DLA-VC's pre→post output (x_a2b) should match
                # the teacher's output. MSE + cosine on the same target space.
                with torch.no_grad():
                    teacher_post_a2b = teacher(t_pre)
                    teacher_pre_b2a = teacher(t_post)  # teacher run "backwards" — not
                    # strictly what it was trained for, but provides a soft target.
                loss_kd_a2b = (F.mse_loss(x_a2b, teacher_post_a2b) + 0.5 *
                               (1.0 - F.cosine_similarity(x_a2b, teacher_post_a2b, dim=1).mean()))
                loss_kd_b2a = (F.mse_loss(x_b2a, teacher_pre_b2a) + 0.5 *
                               (1.0 - F.cosine_similarity(x_b2a, teacher_pre_b2a, dim=1).mean()))
                # b2a is the weaker signal (teacher wasn't trained for post→pre);
                # weight it less.
                loss_kd = 0.8 * loss_kd_a2b + 0.2 * loss_kd_b2a

                re_a2b, _ = model.unet_encoder(x_a2b)
                re_cq_a2b, _, _ = model.vq(model.content_proj(re_a2b))
                re_b2a, _ = model.unet_encoder(x_b2a)
                re_cq_b2a, _, _ = model.vq(model.content_proj(re_b2a))

                loss_cycle = (F.l1_loss(re_cq_a2b, cq_pre.detach()) +
                              F.l1_loss(re_cq_b2a, cq_post.detach()))

                # Pre → post quality mapping: q_shift(q_pre) should predict q_post.
                # Target is detached so only q_shift (and encoder via q_pre) get
                # gradients; post encoder isn't pulled toward the predicted direction.
                q_post_pred = model.q_shift(q_pre)
                loss_q_shift = (F.mse_loss(q_post_pred, q_post.detach()) + 0.5 *
                                (1.0 - F.cosine_similarity(q_post_pred,
                                                            q_post.detach(), dim=-1).mean()))

            # Ramp VQ weight from 0.01 to LAMBDA_VQ over warmup so the encoder
            # can learn unconstrained reconstruction first, then gradually force
            # quantization. Prevents VQ from dominating gradients early on.
            vq_weight = (LAMBDA_VQ * min(1.0, 0.01 + 0.99 * (epoch / WARMUP_EPOCHS))
                         if epoch < WARMUP_EPOCHS else LAMBDA_VQ)

            if epoch < WARMUP_EPOCHS:
                loss = (LAMBDA_RECON * loss_recon
                        + LAMBDA_CONTENT_CYCLE * loss_content_cycle
                        + vq_weight * vq_loss)
            else:
                loss = (LAMBDA_RECON_PHASE2 * loss_recon
                        + LAMBDA_CONTENT_CYCLE * loss_content_cycle
                        + vq_weight * vq_loss
                        + LAMBDA_CYCLE * loss_cycle
                        + LAMBDA_CONV * loss_conv
                        + LAMBDA_KD * loss_kd
                        + LAMBDA_Q_SHIFT * loss_q_shift)

            # ECAPA speaker loss on full utterance (every N steps, after warmup)
            global_step += 1
            if epoch >= WARMUP_EPOCHS and global_step % ECAPA_EVERY == 0:
                utt_idx = random.choice(train_idx)
                utt_audio, utt_sr = torchaudio.load(pre_files[utt_idx])
                if utt_sr != SAMPLE_RATE:
                    utt_audio = torchaudio.functional.resample(utt_audio, utt_sr, SAMPLE_RATE)
                utt_audio = utt_audio[0].unsqueeze(0).to(device)  # (1, T)
                if utt_audio.shape[-1] > MAX_WAVLM_SAMPLES:
                    s = random.randint(0, utt_audio.shape[-1] - MAX_WAVLM_SAMPLES)
                    utt_audio = utt_audio[:, s:s + MAX_WAVLM_SAMPLES]

                utt_post_audio, utt_sr2 = torchaudio.load(post_files[utt_idx])
                if utt_sr2 != SAMPLE_RATE:
                    utt_post_audio = torchaudio.functional.resample(utt_post_audio, utt_sr2, SAMPLE_RATE)
                utt_post_audio = utt_post_audio[0].unsqueeze(0).to(device)
                if utt_post_audio.shape[-1] > MAX_WAVLM_SAMPLES:
                    s2 = random.randint(0, utt_post_audio.shape[-1] - MAX_WAVLM_SAMPLES)
                    utt_post_audio = utt_post_audio[:, s2:s2 + MAX_WAVLM_SAMPLES]

                with torch.no_grad():
                    h_pre_utt, _ = wavlm.extract(utt_audio)
                    h_post_utt, _ = wavlm.extract(utt_post_audio)

                cq_utt, sk_utt = model.encode_content(h_pre_utt)
                q_post_utt = model.encode_quality(h_post_utt)
                conv_feats = model.unet_decoder(cq_utt, q_post_utt, sk_utt)
                conv_feats = conv_feats[:, :, :h_pre_utt.shape[-1]]  # match time

                # Crop to STYLE_CROP_LEN frames before vocoding.
                # Full utterances (~3000 WavLM frames) OOM on 20GB GPU because
                # all HiFiGAN conv activations are stored for backprop.
                # 128 frames → 128×320 = 40,960 audio samples — fits fine.
                T_cf = conv_feats.shape[-1]
                crop_len = min(T_cf, STYLE_CROP_LEN)
                c_start = random.randint(0, max(0, T_cf - crop_len))
                conv_feats_crop = conv_feats[:, :, c_start:c_start + crop_len]

                # Vocode (differentiable) and get ECAPA embedding
                audio_conv = vocode_differentiable(vocoder, conv_feats_crop.transpose(1, 2))
                emb_conv = ecapa_embed_differentiable(ecapa, audio_conv).squeeze()
                target_emb = post_ecapa_embs[utt_idx]
                ecapa_l = 1.0 - F.cosine_similarity(
                    emb_conv.unsqueeze(0), target_emb.unsqueeze(0)).squeeze()
                loss = loss + LAMBDA_ECAPA * ecapa_l
                ecapa_losses.append(ecapa_l.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step()

        sched.step()
        torch.cuda.empty_cache()  # reclaim fragmented allocator blocks before val

        # Validate (content-code consistency — cheap, monitor only)
        model.eval()
        vr = 0; nv = 0
        with torch.no_grad():
            for av, _ in val_loader:
                hv, tv = wavlm.extract(av.to(device))
                rv, _, _, cz_v, _ = model(hv, tv)
                rv_enc_out, _ = model.unet_encoder(rv)
                rv_content = model.content_proj(rv_enc_out)
                vr += F.mse_loss(rv_content, cz_v).item(); nv += 1
        cycle_val = vr / max(nv, 1)

        # ECAPA-based VC quality on val patients (drives save decisions after warmup).
        # Recompute avg quality each epoch since model is changing.
        avg_q_pre = avg_q_post = None
        ecapa_val = float('nan')
        if epoch >= WARMUP_EPOCHS:
            avg_q_pre  = compute_avg_quality_from_files(model, wavlm, train_pre,  device)
            avg_q_post = compute_avg_quality_from_files(model, wavlm, train_post, device)
            avg_q_post_eval = avg_q_post.unsqueeze(0).to(device) if avg_q_post.dim() == 1 else avg_q_post.to(device)
            ecapa_val = compute_ecapa_val(model, wavlm, vocoder, ecapa,
                                          pre_files, val_idx, post_ecapa_embs,
                                          avg_q_post_eval, device)
            save_metric = ecapa_val
        else:
            save_metric = float('inf')  # don't save during warmup

        warmup = " [WARMUP]" if epoch < WARMUP_EPOCHS else ""
        avg_ecapa = np.mean(ecapa_losses) if ecapa_losses else 0.0
        ecapa_losses = []
        if (epoch + 1) % 10 == 0 or epoch == 0:
            conv_val = loss_conv.item() if torch.is_tensor(loss_conv) else float(loss_conv)
            qshift_val = loss_q_shift.item() if torch.is_tensor(loss_q_shift) else float(loss_q_shift)
            kd_val = loss_kd.item() if torch.is_tensor(loss_kd) else float(loss_kd)
            alpha_val = model.alpha.item() if hasattr(model, 'alpha') else float('nan')
            print(f"Epoch {epoch+1}{warmup} | Recon: {loss_recon.item():.4f} | "
                  f"Conv: {conv_val:.4f} | KD: {kd_val:.4f} | "
                  f"Qshift: {qshift_val:.4f} | "
                  f"VQ: {vq_loss.item():.4f} | Perp: {perp.item():.0f}/{NUM_CODES} | "
                  f"alpha: {alpha_val:.3f} | ECAPA_val: {ecapa_val:.4f}")

        if save_metric < best_val:
            best_val = save_metric
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'avg_quality_pre': avg_q_pre, 'avg_quality_post': avg_q_post,
                'adapter_weights': model.get_adapter_weights(),
                'val_loss': save_metric,
                'cycle_val': cycle_val,
                'ecapa_val': ecapa_val,
                'config': {'feat_dim': FEAT_DIM, 'code_dim': CODE_DIM,
                           'num_codes': NUM_CODES, 'num_heads': NUM_HEADS,
                           'quality_dim': QUALITY_DIM, 'num_wavlm_layers': wavlm.num_layers,
                           'use_residual_output': True},
            }, ckpt_path)
            print(f"  -> Saved (epoch {epoch+1}, ECAPA_val={ecapa_val:.4f})")
        elif epoch >= WARMUP_EPOCHS:
            patience_counter += 1
            past_min = (epoch + 1) >= MIN_EPOCHS
            if patience_counter >= PATIENCE and past_min:
                print(f"Early stop at epoch {epoch+1}")
                break
            elif patience_counter >= PATIENCE and not past_min:
                print(f"  Patience exhausted but continuing (epoch {epoch+1}<{MIN_EPOCHS})")
                patience_counter = 0  # reset to keep training

    print(f"\nBest val: {best_val:.4f}")

    # ═══ Evaluate ═══
    print(f"\n{'='*70}")
    print(f"  Evaluating DLA-VC on test + train patients")
    print(f"{'='*70}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_eval = DLAVCModel(
        feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
        num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
        num_wavlm_layers=wavlm.num_layers,
        commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
        entropy_weight=ENTROPY_WEIGHT, dropout=0.0, content_noise_std=0.0,
    ).to(device)
    model_eval.load_state_dict(ckpt['model'])
    model_eval.eval()
    avg_q_post = ckpt['avg_quality_post']

    def get_emb(path):
        sig, sr = torchaudio.load(path)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return ecapa.encode_batch(sig.to(device)).squeeze()

    conv_dir = os.path.join(args.output, 'converted')
    os.makedirs(conv_dir, exist_ok=True)

    def evaluate_patients(patient_indices, tag):
        results = []
        for i in patient_indices:
            audio, sr = torchaudio.load(pre_files[i])
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = audio[0].unsqueeze(0).to(device)
            post_audio, post_sr = torchaudio.load(post_files[i])
            if post_sr != SAMPLE_RATE:
                post_audio = torchaudio.functional.resample(post_audio, post_sr, SAMPLE_RATE)
            post_audio = post_audio[0].unsqueeze(0).to(device)
            with torch.no_grad():
                hidden, _ = wavlm.extract(audio)
                h_post_i, _ = wavlm.extract(post_audio)
                q_post_i = model_eval.encode_quality(h_post_i)
                converted = model_eval.convert(hidden, q_post_i)
            wav = knn_vc.vocode(converted.squeeze(0).t()[None]).cpu().squeeze()
            out_path = os.path.join(conv_dir, names[i] + '.wav')
            torchaudio.save(out_path, wav.unsqueeze(0), SAMPLE_RATE)

            emb_c = get_emb(out_path)
            emb_post = get_emb(post_files[i])
            emb_pre = get_emb(pre_files[i])
            sp = F.cosine_similarity(emb_c.unsqueeze(0), emb_post.unsqueeze(0)).item()
            sr_ = F.cosine_similarity(emb_c.unsqueeze(0), emb_pre.unsqueeze(0)).item()
            bl = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()
            results.append({'name': names[i], 'sim_post': sp, 'sim_pre': sr_, 'baseline': bl})
            print(f"  [{tag}] {names[i]}: conv->post={sp:.3f}  baseline={bl:.3f}  delta={sp-bl:+.3f}")
        return results

    print("\n--- TRAIN (in-training eval; real test eval is run by run_eval.py) ---")
    results_test = []
    results_train = evaluate_patients(train_idx, "TRAIN")

    trp = [r['sim_post'] for r in results_train]
    trb = [r['baseline'] for r in results_train]

    print(f"\n{'='*70}")
    print(f"  DLA-VC — {args.surgery} — SUMMARY")
    print(f"{'='*70}")
    print(f"  TRAIN ({len(train_idx)}):")
    print(f"    Baseline:     {np.mean(trb):.3f} +/- {np.std(trb):.3f}")
    print(f"    Conv vs post: {np.mean(trp):.3f} +/- {np.std(trp):.3f}")
    print(f"    Improvement:  {np.mean(trp) - np.mean(trb):+.3f}")
    print(f"{'='*70}")

    with open(os.path.join(args.output, 'results.json'), 'w') as f:
        json.dump({'method': 'DLA-VC', 'surgery': args.surgery,
                   'test': results_test, 'train': results_train}, f, indent=2)


if __name__ == '__main__':
    main()
