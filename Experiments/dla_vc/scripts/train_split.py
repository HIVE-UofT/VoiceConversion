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
EPOCHS = 400
LR = 1e-4
SEGMENT_SAMPLES = 40000
SEGMENT_HOP_SAMPLES = 20000
WARMUP_EPOCHS = 30
PATIENCE = 40

LAMBDA_RECON = 10.0        # Direct WavLM feature MSE (high during warmup)
LAMBDA_RECON_PHASE2 = 2.0  # Reduced recon weight after warmup (so ECAPA can work)
LAMBDA_VQ = 1.0
LAMBDA_CYCLE = 2.0
ECAPA_EVERY = 3            # ECAPA loss every N training steps
LAMBDA_ECAPA = 3.0         # ECAPA speaker similarity weight
ECAPA_STOP_THRESH = 0.25   # Don't early-stop until ECAPA loss drops below this
MIN_EPOCHS = 150           # Minimum epochs before early stopping is allowed


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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), '..',
                                    f'results_{args.surgery.lower()}_split')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output, exist_ok=True)

    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "3")
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
    random.shuffle(cv_idx)
    n_val = max(1, int(0.15 * len(cv_idx)))
    val_idx = sorted(cv_idx[:n_val])
    train_idx = sorted(cv_idx[n_val:])

    test_names = [names[i] for i in test_idx]
    print(f"\n{args.surgery}: {n} patients | train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Test: {test_names}")

    with open(os.path.join(args.output, 'split_info.json'), 'w') as f:
        json.dump({'test': test_names, 'train': [names[i] for i in train_idx],
                   'val': [names[i] for i in val_idx], 'seed': args.seed}, f, indent=2)

    # WavLM
    wavlm = WavLMFeatureExtractor(device)

    # Datasets — train only on train patients
    train_pre = [pre_files[i] for i in train_idx]
    train_post = [post_files[i] for i in train_idx]
    val_pre = [pre_files[i] for i in val_idx]
    val_post = [post_files[i] for i in val_idx]

    ds_pre = AudioSegmentDatasetFromFiles(train_pre, label=0,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_HOP_SAMPLES, augment=True)
    ds_post = AudioSegmentDatasetFromFiles(train_post, label=1,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_HOP_SAMPLES, augment=True)
    print(f"  Train: {len(ds_pre)} pre + {len(ds_post)} post segments")

    loader_pre = DataLoader(ds_pre, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=2, pin_memory=True)
    loader_post = DataLoader(ds_post, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    paired_loader = PairedDomainLoader(loader_pre, loader_post)

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

        for audio_pre, _, audio_post, _ in paired_loader:
            audio_all = torch.cat([audio_pre, audio_post], dim=0).to(device)
            B_half = audio_pre.shape[0]
            hidden_all, target_all = wavlm.extract(audio_all)

            loss_cycle = torch.tensor(0.0, device=device)

            opt_model.zero_grad()
            recon, vq_loss, perp, content_z, quality = model(hidden_all, target_all)
            # Content consistency: re-extract content from output, compare to input content
            recon_enc_out, _ = model.unet_encoder(recon)
            recon_content = model.content_proj(recon_enc_out)
            loss_recon = F.mse_loss(recon_content, content_z.detach())

            if epoch >= WARMUP_EPOCHS:
                h_pre, h_post = hidden_all[:B_half], hidden_all[B_half:]
                t_pre, t_post = target_all[:B_half], target_all[B_half:]
                cq_pre, sk_pre = model.encode_content(h_pre)
                cq_post, sk_post = model.encode_content(h_post)
                q_pre = model.encode_quality(h_pre)
                q_post = model.encode_quality(h_post)

                x_a2b = model._match_time(model.unet_decoder(cq_pre, q_post, sk_pre), t_pre)
                x_b2a = model._match_time(model.unet_decoder(cq_post, q_pre, sk_post), t_post)

                re_a2b, _ = model.unet_encoder(x_a2b)
                re_cq_a2b, _, _ = model.vq(model.content_proj(re_a2b))
                re_b2a, _ = model.unet_encoder(x_b2a)
                re_cq_b2a, _, _ = model.vq(model.content_proj(re_b2a))

                loss_cycle = (F.l1_loss(re_cq_a2b, cq_pre.detach()) +
                              F.l1_loss(re_cq_b2a, cq_post.detach()))

            if epoch < WARMUP_EPOCHS:
                loss = LAMBDA_RECON * loss_recon + LAMBDA_VQ * vq_loss
            else:
                loss = (LAMBDA_RECON_PHASE2 * loss_recon + LAMBDA_VQ * vq_loss +
                        LAMBDA_CYCLE * loss_cycle)

            # ECAPA speaker loss on full utterance (every N steps, after warmup)
            global_step += 1
            if epoch >= WARMUP_EPOCHS and global_step % ECAPA_EVERY == 0:
                utt_idx = random.choice(train_idx)
                utt_audio, utt_sr = torchaudio.load(pre_files[utt_idx])
                if utt_sr != SAMPLE_RATE:
                    utt_audio = torchaudio.functional.resample(utt_audio, utt_sr, SAMPLE_RATE)
                utt_audio = utt_audio[0].unsqueeze(0).to(device)  # (1, T)

                utt_post_audio, utt_sr2 = torchaudio.load(post_files[utt_idx])
                if utt_sr2 != SAMPLE_RATE:
                    utt_post_audio = torchaudio.functional.resample(utt_post_audio, utt_sr2, SAMPLE_RATE)
                utt_post_audio = utt_post_audio[0].unsqueeze(0).to(device)

                with torch.no_grad():
                    h_pre_utt, _ = wavlm.extract(utt_audio)
                    h_post_utt, _ = wavlm.extract(utt_post_audio)

                cq_utt, sk_utt = model.encode_content(h_pre_utt)
                q_post_utt = model.encode_quality(h_post_utt)
                conv_feats = model.unet_decoder(cq_utt, q_post_utt, sk_utt)
                conv_feats = conv_feats[:, :, :h_pre_utt.shape[-1]]  # match time

                # Vocode (differentiable) and get ECAPA embedding
                audio_conv = vocode_differentiable(vocoder, conv_feats.transpose(1, 2))
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

        # Validate (content-code consistency)
        model.eval()
        vr = 0; nv = 0
        with torch.no_grad():
            for av, _ in val_loader:
                hv, tv = wavlm.extract(av.to(device))
                rv, _, _, cz_v, _ = model(hv, tv)
                rv_enc_out, _ = model.unet_encoder(rv)
                rv_content = model.content_proj(rv_enc_out)
                vr += F.mse_loss(rv_content, cz_v).item(); nv += 1
        avg_val = vr / max(nv, 1)

        # Reset early stopping when warmup ends so the best-val from
        # the trivially-easy warmup phase doesn't block real training
        if epoch == WARMUP_EPOCHS:
            best_val = float('inf')
            patience_counter = 0

        warmup = " [WARMUP]" if epoch < WARMUP_EPOCHS else ""
        avg_ecapa = np.mean(ecapa_losses) if ecapa_losses else 0.0
        ecapa_losses = []
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}{warmup} | Recon: {loss_recon.item():.4f} | "
                  f"VQ: {vq_loss.item():.4f} | Perp: {perp.item():.0f}/{NUM_CODES} | "
                  f"ECAPA_l: {avg_ecapa:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            avg_q_pre = compute_avg_quality_from_files(model, wavlm, train_pre, device)
            avg_q_post = compute_avg_quality_from_files(model, wavlm, train_post, device)
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'avg_quality_pre': avg_q_pre, 'avg_quality_post': avg_q_post,
                'adapter_weights': model.get_adapter_weights(),
                'val_loss': avg_val,
                'config': {'feat_dim': FEAT_DIM, 'code_dim': CODE_DIM,
                           'num_codes': NUM_CODES, 'num_heads': NUM_HEADS,
                           'quality_dim': QUALITY_DIM, 'num_wavlm_layers': wavlm.num_layers},
            }, ckpt_path)
        else:
            patience_counter += 1
            ecapa_ok = avg_ecapa < ECAPA_STOP_THRESH
            past_min = (epoch + 1) >= MIN_EPOCHS
            if patience_counter >= PATIENCE and ecapa_ok and past_min:
                print(f"Early stop at epoch {epoch+1}")
                break
            elif patience_counter >= PATIENCE and not (ecapa_ok and past_min):
                reason = []
                if not ecapa_ok:
                    reason.append(f"ECAPA={avg_ecapa:.3f}>{ECAPA_STOP_THRESH}")
                if not past_min:
                    reason.append(f"epoch {epoch+1}<{MIN_EPOCHS}")
                print(f"  Patience exhausted but continuing ({', '.join(reason)})")
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

    print("\n--- TEST ---")
    results_test = evaluate_patients(test_idx, "TEST")
    print("\n--- TRAIN ---")
    results_train = evaluate_patients(train_idx, "TRAIN")

    tp = [r['sim_post'] for r in results_test]
    tb = [r['baseline'] for r in results_test]
    trp = [r['sim_post'] for r in results_train]
    trb = [r['baseline'] for r in results_train]

    print(f"\n{'='*70}")
    print(f"  DLA-VC — {args.surgery} — SUMMARY")
    print(f"{'='*70}")
    print(f"  TEST ({len(test_idx)}):")
    print(f"    Baseline:     {np.mean(tb):.3f} +/- {np.std(tb):.3f}")
    print(f"    Conv vs post: {np.mean(tp):.3f} +/- {np.std(tp):.3f}")
    print(f"    Improvement:  {np.mean(tp) - np.mean(tb):+.3f}")
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
