"""
Evaluate FreeVC + ECAPA shift + ECAPA→FreeVC bridge on 5 held-out test patients.

Compares several conversion strategies (all using frozen FreeVC generator):

  A. avg_post_freevc      : zero-shot FreeVC with avg training post FreeVC-spk emb
  B. bridge(avg_post_ecapa): generic post in ECAPA space → bridge to FreeVC space
  C. bridge(pre_ecapa + delta_ecapa): per-patient linear shift in ECAPA space
  D. bridge(shift(pre_ecapa)): learned nonlinear shift in ECAPA space — our method

Metric: ECAPA cosine similarity between converted audio and real post audio.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

FREEVC_DIR = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc/FreeVC'
EXP_FREEVC = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc'
EXP_ECAPA  = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_ecapa'
sys.path.insert(0, FREEVC_DIR)
sys.path.insert(0, os.path.join(EXP_ECAPA, 'scripts'))
os.chdir(FREEVC_DIR)

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from shift_models import ShiftEcapa, BridgeEcapaToFreeVC

TEST_PATIENTS = ['0045', '0085', '0110', '0122', '0132']
CUCO_BASE = '/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios'
SR = 16000


def collect_paths(surgery='Tonsill'):
    def one(sess):
        pat = os.path.join(CUCO_BASE, surgery, 'Speech', sess, '*.wav')
        return {Path(p).stem.split('_')[-1]: p for p in sorted(glob.glob(pat))}
    return one('1'), one('2')


def load_wav(path, device):
    sig, sr = torchaudio.load(path)
    if sig.shape[0] > 1: sig = sig.mean(dim=0, keepdim=True)
    if sr != SR: sig = torchaudio.functional.resample(sig, sr, SR)
    return sig.to(device)


def load_wav_np(path, sr=SR):
    return librosa.load(path, sr=sr)[0]


def norm_np(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def freevc_generate(net_g, cmodel, pre_wav, spk_emb_np, device):
    """Run frozen FreeVC generator with given 256-d speaker embedding."""
    with torch.no_grad():
        c = utils.get_content(cmodel, pre_wav)
        g = torch.from_numpy(spk_emb_np).float().unsqueeze(0).to(device)
        # Ensure it's on unit sphere as FreeVC expects
        g = F.normalize(g, dim=-1)
        o = net_g.infer(c, g=g)
    return o.squeeze().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surgery', type=str, default='Tonsill')
    ap.add_argument('--config', type=str, default='configs/freevc.json')
    ap.add_argument('--freevc_ckpt', type=str,
                    default=os.path.join(EXP_FREEVC, 'checkpoints/freevc.pth'))
    ap.add_argument('--bridge_ckpt', type=str,
                    default=os.path.join(EXP_ECAPA, 'checkpoints/bridge.pt'))
    ap.add_argument('--shift_ckpt', type=str,
                    default=os.path.join(EXP_ECAPA, 'checkpoints/shift.pt'))
    ap.add_argument('--avg_pre_ecapa', type=str,
                    default=os.path.join(EXP_ECAPA, 'checkpoints/avg_pre_ecapa.npy'))
    ap.add_argument('--avg_post_ecapa', type=str,
                    default=os.path.join(EXP_ECAPA, 'checkpoints/avg_post_ecapa.npy'))
    ap.add_argument('--avg_post_freevc', type=str,
                    default=os.path.join(EXP_ECAPA, 'checkpoints/avg_post_freevc_spk.npy'))
    ap.add_argument('--out_dir', type=str,
                    default=os.path.join(EXP_ECAPA, 'converted_test'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    hps = utils.get_hparams_from_file(args.config)

    # ---------- Load frozen components ----------
    print('Loading frozen FreeVC synthesizer...')
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device).eval()
    _ = utils.load_checkpoint(args.freevc_ckpt, net_g, None, True)

    print('Loading frozen WavLM content encoder...')
    cmodel = utils.get_cmodel(0)

    print('Loading frozen FreeVC speaker encoder (for reference only)...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print('Loading frozen ECAPA-TDNN (metric + input to our shift)...')
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    for p in ecapa.mods.parameters(): p.requires_grad = False

    # ---------- Load our trained bridge + shift ----------
    print('Loading learned bridge (ECAPA → FreeVC_spk)...')
    bridge_ckpt = torch.load(args.bridge_ckpt, map_location=device, weights_only=False)
    bridge = BridgeEcapaToFreeVC(**bridge_ckpt['config']).to(device).eval()
    bridge.load_state_dict(bridge_ckpt['state_dict'])
    print(f'  bridge CV val_loss: {bridge_ckpt.get("cv_val_loss_mean", float("nan")):.4f}')

    print('Loading learned shift_ecapa (pre → post in ECAPA space)...')
    shift_ckpt = torch.load(args.shift_ckpt, map_location=device, weights_only=False)
    shift = ShiftEcapa(**shift_ckpt['config']).to(device).eval()
    shift.load_state_dict(shift_ckpt['state_dict'])
    print(f'  shift CV val_loss: {shift_ckpt.get("cv_val_loss_mean", float("nan")):.4f}')

    # ---------- Precomputed averages ----------
    avg_pre_ecapa   = np.load(args.avg_pre_ecapa)
    avg_post_ecapa  = np.load(args.avg_post_ecapa)
    avg_post_freevc = np.load(args.avg_post_freevc)
    mean_delta_ecapa = norm_np(avg_post_ecapa - avg_pre_ecapa)

    def ecapa_emb_of(signal_or_path):
        if isinstance(signal_or_path, str):
            sig, sr = torchaudio.load(signal_or_path)
            if sig.shape[0] > 1: sig = sig.mean(dim=0, keepdim=True)
            if sr != SR: sig = torchaudio.functional.resample(sig, sr, SR)
        else:
            sig = signal_or_path if signal_or_path.dim() == 2 else signal_or_path.unsqueeze(0)
        with torch.no_grad():
            return ecapa.encode_batch(sig.to(device)).squeeze().cpu()

    def bridge_from_ecapa(ecapa_np):
        """Bridge an ECAPA embedding to FreeVC speaker space."""
        with torch.no_grad():
            x = torch.from_numpy(norm_np(ecapa_np)).float().unsqueeze(0).to(device)
            y = bridge(x).squeeze(0).cpu().numpy()
        return y  # already L2-normed by bridge

    # ---------- Evaluate on test patients ----------
    pre_paths, post_paths = collect_paths(args.surgery)
    strategies = ['A_avg_post_freevc', 'B_bridge_avg_post', 'C_bridge_pre+delta', 'D_bridge_shift']
    rows = {s: [] for s in strategies}

    for pid in TEST_PATIENTS:
        if pid not in pre_paths or pid not in post_paths:
            continue
        pre_path = pre_paths[pid]; post_path = post_paths[pid]
        pre_wav = load_wav(pre_path, device)

        # Per-test-patient ECAPA embedding (for C, D)
        pre_wav_np = load_wav_np(pre_path)
        pre_ecapa = norm_np(ecapa.encode_batch(
            torch.from_numpy(pre_wav_np).unsqueeze(0).to(device)).squeeze().cpu().numpy())

        # Strategy speaker embeddings (all 256-d FreeVC space):
        spk_A = avg_post_freevc                                 # population avg
        spk_B = bridge_from_ecapa(avg_post_ecapa)               # ECAPA avg → bridge
        spk_C = bridge_from_ecapa(norm_np(pre_ecapa + mean_delta_ecapa))  # per-patient linear
        with torch.no_grad():
            pe_t = torch.from_numpy(pre_ecapa).float().unsqueeze(0).to(device)
            post_ecapa_pred = shift(pe_t).squeeze(0).cpu().numpy()
        spk_D = bridge_from_ecapa(post_ecapa_pred)              # learned shift → bridge

        # Metric baseline: pre vs real post
        emb_pre  = ecapa_emb_of(pre_path)
        emb_post = ecapa_emb_of(post_path)
        sim_base = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()

        for tag, spk in zip(strategies, [spk_A, spk_B, spk_C, spk_D]):
            out = freevc_generate(net_g, cmodel, pre_wav, spk, device)
            out_path = os.path.join(args.out_dir, f'{pid}_{tag}.wav')
            torchaudio.save(out_path, out.unsqueeze(0), hps.data.sampling_rate)
            emb_conv = ecapa_emb_of(out.unsqueeze(0))
            sim_conv = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
            rows[tag].append((pid, sim_base, sim_conv, sim_conv - sim_base))

    # ---------- Report ----------
    print(f'\n{"="*74}')
    print(f'  FreeVC + ECAPA-space shift + bridge — test evaluation')
    print(f'{"="*74}')
    print(f'\n  Strategy                 Baseline   Converted     Delta (mean ± std)')
    print(f'  ' + '-' * 68)
    for tag in strategies:
        b = np.array([r[1] for r in rows[tag]])
        c = np.array([r[2] for r in rows[tag]])
        d = np.array([r[3] for r in rows[tag]])
        print(f'  {tag:24} {b.mean():.4f}     {c.mean():.4f}     {d.mean():+.4f} ± {d.std():.4f}')

    print(f'\n  Per-patient deltas:')
    print(f'  {"PID":6}  ' + '  '.join(f'{s:22}' for s in strategies))
    for i, pid in enumerate(TEST_PATIENTS):
        vals = '  '.join(f'{rows[s][i][3]:+.4f}               ' for s in strategies)
        print(f'  {pid:6}  {vals}')

    with open(os.path.join(args.out_dir, '..', 'test_results.json'), 'w') as f:
        json.dump({s: [{'pid': r[0], 'base': r[1], 'conv': r[2], 'delta': r[3]}
                       for r in rows[s]] for s in strategies}, f, indent=2)
    print(f'\nSaved results.json and wavs to {args.out_dir}')


if __name__ == '__main__':
    main()
