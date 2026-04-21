"""
Evaluate: frozen FreeVC + learned SurgeryShift on test patients.

For each test patient, compare 4 inference strategies (all with the same
frozen FreeVC generator and WavLM content encoder):

  A. avg_post_spk            – population mean post embedding (generic post)
  B. pre_spk + mean_delta    – per-patient pre + mean surgery shift in embedding space
  C. avg_post_spk (matched)  – nearest-train-patient's post embedding by pre similarity
  D. shift_net(pre_spk)      – learned per-patient post embedding (our method)

Metric: ECAPA-TDNN cosine similarity between converted audio and real post audio.
Baseline: cosine between real pre and real post (same patient).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

FREEVC_DIR = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc/FreeVC'
EXP_FREE_VC = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc'
EXP_SHIFT = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_shift'
sys.path.insert(0, FREEVC_DIR)
sys.path.insert(0, os.path.join(EXP_SHIFT, 'scripts'))
os.chdir(FREEVC_DIR)

import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from shift_model import SurgeryShift

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
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != SR:
        sig = torchaudio.functional.resample(sig, sr, SR)
    return sig.to(device)


def load_wav_np(path, sr=SR):
    wav, _ = librosa.load(path, sr=sr); return wav


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def freevc_infer(net_g, cmodel, pre_wav, spk_emb, device):
    """Run frozen FreeVC conversion with given target speaker embedding."""
    with torch.no_grad():
        c = utils.get_content(cmodel, pre_wav)                   # (1, 1024, T)
        g = torch.from_numpy(spk_emb).float().unsqueeze(0).to(device)
        o = net_g.infer(c, g=g)
    return o.squeeze().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surgery', type=str, default='Tonsill')
    ap.add_argument('--config', type=str, default='configs/freevc.json')
    ap.add_argument('--freevc_ckpt', type=str,
                    default=os.path.join(EXP_FREE_VC, 'checkpoints/freevc.pth'))
    ap.add_argument('--shift_ckpt', type=str,
                    default=os.path.join(EXP_SHIFT, 'checkpoints/shift.pt'))
    ap.add_argument('--avg_post_spk', type=str,
                    default=os.path.join(EXP_SHIFT, 'checkpoints/avg_post_spk.npy'))
    ap.add_argument('--avg_pre_spk', type=str,
                    default=os.path.join(EXP_SHIFT, 'checkpoints/avg_pre_spk.npy'))
    ap.add_argument('--out_dir', type=str,
                    default=os.path.join(EXP_SHIFT, 'converted_test'))
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

    print('Loading frozen FreeVC speaker encoder...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # ---------- Load learned shift ----------
    print('Loading learned SurgeryShift...')
    shift_ckpt = torch.load(args.shift_ckpt, map_location=device, weights_only=False)
    cfg = shift_ckpt['config']
    shift_net = SurgeryShift(**cfg).to(device).eval()
    shift_net.load_state_dict(shift_ckpt['state_dict'])
    print(f'  Trained shift (CV val_loss = {shift_ckpt.get("cv_val_loss_mean", "?"):.4f})')

    # Load population-level baselines
    avg_post = np.load(args.avg_post_spk)
    avg_pre  = np.load(args.avg_pre_spk)
    mean_delta = norm(avg_post - avg_pre)   # surgery direction vector

    # ---------- Precompute training pre/post embeddings (for method C) ----------
    pre_paths, post_paths = collect_paths(args.surgery)
    train_pids = sorted(p for p in pre_paths if p not in TEST_PATIENTS and p in post_paths)
    print(f'\n{len(train_pids)} train patients for matching, {len(TEST_PATIENTS)} test patients')

    train_pre_embs, train_post_embs = {}, {}
    for pid in train_pids:
        train_pre_embs[pid]  = norm(smodel.embed_utterance(load_wav_np(pre_paths[pid])))
        train_post_embs[pid] = norm(smodel.embed_utterance(load_wav_np(post_paths[pid])))

    # ---------- ECAPA for metric ----------
    print('\nLoading ECAPA-TDNN (for metric)...')
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    for p in ecapa.mods.parameters(): p.requires_grad = False

    def ecapa_emb(src):
        if isinstance(src, str):
            sig, sr = torchaudio.load(src)
            if sig.shape[0] > 1: sig = sig.mean(dim=0, keepdim=True)
            if sr != SR: sig = torchaudio.functional.resample(sig, sr, SR)
        else:
            sig = src if src.dim() == 2 else src.unsqueeze(0)
        with torch.no_grad():
            return ecapa.encode_batch(sig.to(device)).squeeze().cpu()

    # ---------- Evaluate ----------
    strategies = ['A_avg_post', 'B_pre_plus_delta', 'C_nn_match', 'D_learned_shift']
    rows = {s: [] for s in strategies}

    for pid in TEST_PATIENTS:
        if pid not in pre_paths or pid not in post_paths:
            continue
        pre_path  = pre_paths[pid]
        post_path = post_paths[pid]
        pre_wav   = load_wav(pre_path, device)
        pre_spk   = norm(smodel.embed_utterance(load_wav_np(pre_path)))  # (256,)

        emb_pre  = ecapa_emb(pre_path)
        emb_post = ecapa_emb(post_path)
        sim_base = F.cosine_similarity(emb_pre.unsqueeze(0),
                                        emb_post.unsqueeze(0)).item()

        # A. generic avg post
        spk_A = avg_post
        # B. pre + mean surgery delta (re-normalised)
        spk_B = norm(pre_spk + mean_delta)
        # C. nearest-neighbour matched post
        sims = {p: float(pre_spk @ train_pre_embs[p]) for p in train_pids}
        nn_pid = max(sims, key=sims.get)
        spk_C = train_post_embs[nn_pid]
        # D. learned shift
        with torch.no_grad():
            spk_D = shift_net(torch.from_numpy(pre_spk).float().unsqueeze(0).to(device))
            spk_D = spk_D.squeeze(0).cpu().numpy()

        for tag, spk in zip(strategies, [spk_A, spk_B, spk_C, spk_D]):
            out_wav = freevc_infer(net_g, cmodel, pre_wav, spk, device)
            out_path = os.path.join(args.out_dir, f'{pid}_{tag}.wav')
            torchaudio.save(out_path, out_wav.unsqueeze(0), hps.data.sampling_rate)
            emb_conv = ecapa_emb(out_wav.unsqueeze(0))
            sim_conv = F.cosine_similarity(emb_conv.unsqueeze(0),
                                            emb_post.unsqueeze(0)).item()
            rows[tag].append((pid, sim_base, sim_conv, sim_conv - sim_base))

    # ---------- Report ----------
    print(f'\n{"="*70}')
    print(f'  FreeVC + SurgeryShift — ECAPA eval on test patients')
    print(f'{"="*70}')
    print(f'\n  Strategy                  Baseline   Converted     Delta (mean ± std)')
    print(f'  ' + '-'*64)
    for tag in strategies:
        bases  = np.array([r[1] for r in rows[tag]])
        convs  = np.array([r[2] for r in rows[tag]])
        deltas = np.array([r[3] for r in rows[tag]])
        print(f'  {tag:24}  {bases.mean():.4f}     {convs.mean():.4f}     {deltas.mean():+.4f} ± {deltas.std():.4f}')

    print(f'\n  Per-patient deltas:')
    print(f'  {"PID":6}  ' + '  '.join(f'{s:16}' for s in strategies))
    for i, pid in enumerate(TEST_PATIENTS):
        vals = '  '.join(f'{rows[s][i][3]:+.4f}          ' for s in strategies)
        print(f'  {pid:6}  {vals}')

    print(f'\nConverted wavs saved to: {args.out_dir}')

    # Save metrics JSON
    with open(os.path.join(args.out_dir, '..', 'test_results.json'), 'w') as f:
        json.dump({s: [{'pid': r[0], 'base': r[1], 'conv': r[2], 'delta': r[3]}
                       for r in rows[s]] for s in strategies}, f, indent=2)


if __name__ == '__main__':
    main()
