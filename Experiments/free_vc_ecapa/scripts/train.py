"""
Train two small MLPs on top of frozen FreeVC:

  1. bridge_net    : ECAPA emb (192-d) → FreeVC speaker emb (256-d)
     Trained on ALL training audio (pre + post), on per-crop embeddings.

  2. shift_ecapa   : ECAPA pre emb → ECAPA post emb
     Trained on paired (pre, post) crops from the same patient.

Both are simple MLPs trained by MSE + cosine loss. Patient-level k-fold CV
for each, then a final model trained on all training data.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

FREEVC_DIR  = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc/FreeVC'
EXP_DIR     = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_ecapa'
sys.path.insert(0, FREEVC_DIR)
sys.path.insert(0, os.path.join(EXP_DIR, 'scripts'))
os.chdir(FREEVC_DIR)

from speaker_encoder.voice_encoder import SpeakerEncoder
from shift_models import ShiftEcapa, BridgeEcapaToFreeVC

TEST_PATIENTS = {'0045', '0085', '0110', '0122', '0132'}
CUCO_BASE = '/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios'
SR = 16000


def collect_pairs(surgery='Tonsill'):
    def one(sess):
        pat = os.path.join(CUCO_BASE, surgery, 'Speech', sess, '*.wav')
        return {Path(p).stem.split('_')[-1]: p for p in sorted(glob.glob(pat))}
    pre, post = one('1'), one('2')
    return {pid: {'pre': pre[pid], 'post': post[pid]}
            for pid in pre if pid in post}


def load_wav_np(path, sr=SR):
    return librosa.load(path, sr=sr)[0]


def norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def extract_ecapa_crops(ecapa_model, wav_np, n_crops, crop_sec, device, sr=SR):
    """Extract N ECAPA embeddings from evenly-spaced crops of one audio."""
    crop_samples = int(crop_sec * sr)
    total = len(wav_np)
    if total <= crop_samples:
        sig = torch.from_numpy(wav_np).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ecapa_model.encode_batch(sig).squeeze().cpu().numpy()
        return [emb]
    starts = np.linspace(0, total - crop_samples, n_crops).astype(int)
    embs = []
    for s in starts:
        sig = torch.from_numpy(wav_np[s:s+crop_samples]).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = ecapa_model.encode_batch(sig).squeeze().cpu().numpy()
        embs.append(emb)
    return embs


def extract_freevc_spk_crops(spk_model, wav_np, n_crops, crop_sec, sr=SR):
    """Extract N FreeVC speaker embeddings from evenly-spaced crops."""
    crop_samples = int(crop_sec * sr)
    total = len(wav_np)
    if total <= crop_samples:
        return [spk_model.embed_utterance(wav_np)]
    starts = np.linspace(0, total - crop_samples, n_crops).astype(int)
    return [spk_model.embed_utterance(wav_np[s:s+crop_samples]) for s in starts]


def train_mlp(model, X_train, Y_train, X_val, Y_val, epochs, batch_size,
              lr, weight_decay, cos_weight, device, tag=''):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val, best_state = float('inf'), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        ep_loss, n = 0.0, 0
        for s in range(0, len(X_train), batch_size):
            idx = perm[s:s+batch_size]
            xb, yb = X_train[idx], Y_train[idx]
            yh = model(xb)
            mse = F.mse_loss(yh, yb)
            cos = 1.0 - F.cosine_similarity(yh, yb, dim=-1).mean()
            loss = mse + cos_weight * cos
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            v = model(X_val)
            v_mse = F.mse_loss(v, Y_val).item()
            v_cos = F.cosine_similarity(v, Y_val, dim=-1).mean().item()
            v_loss = v_mse + cos_weight * (1 - v_cos)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f'  {tag} ep {ep+1:3d}  train={ep_loss/n:.4f}  val_mse={v_mse:.4f}  val_cos={v_cos:.4f}')
    return best_val, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surgery', type=str, default='Tonsill')
    ap.add_argument('--n_crops', type=int, default=10)
    ap.add_argument('--crop_sec', type=float, default=3.0)
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-2)
    ap.add_argument('--cos_weight', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str,
                    default=os.path.join(EXP_DIR, 'checkpoints'))
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = collect_pairs(args.surgery)
    train_pids = sorted(p for p in pairs if p not in TEST_PATIENTS)
    print(f'Train patients: {len(train_pids)}')

    # ---------- Load encoders (frozen) ----------
    print('\nLoading ECAPA-TDNN (frozen, our metric space)...')
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)})
    for p in ecapa.mods.parameters(): p.requires_grad = False

    print('Loading FreeVC speaker encoder (frozen)...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # ---------- Extract all embeddings ----------
    print(f'\nExtracting paired embeddings ({args.n_crops} crops × {args.crop_sec}s per utt)...')
    # For each crop of each utterance, get BOTH ECAPA and FreeVC_spk
    # Store as: per-patient lists of (ecapa, freevc_spk) for pre and post
    data = {}  # pid -> {'pre': [(e, s), ...], 'post': [(e, s), ...]}
    for pid in train_pids:
        data[pid] = {}
        for side in ('pre', 'post'):
            wav_np = load_wav_np(pairs[pid][side])
            ecapas  = extract_ecapa_crops(ecapa, wav_np, args.n_crops, args.crop_sec, device)
            spks    = extract_freevc_spk_crops(smodel, wav_np, args.n_crops, args.crop_sec)
            # Match lengths in case of short audio
            n = min(len(ecapas), len(spks))
            data[pid][side] = [(norm(ecapas[i]), norm(spks[i])) for i in range(n)]

    # Bridge training data: all (ECAPA, FreeVC_spk) pairs (both pre and post)
    bridge_X, bridge_Y, bridge_pid = [], [], []
    for pid in train_pids:
        for side in ('pre', 'post'):
            for e, s in data[pid][side]:
                bridge_X.append(e); bridge_Y.append(s); bridge_pid.append(pid)

    # Shift training data: (pre_ECAPA, post_ECAPA) pairs cross-product within each patient
    shift_X, shift_Y, shift_pid = [], [], []
    for pid in train_pids:
        for ep, _ in data[pid]['pre']:
            for eo, _ in data[pid]['post']:
                shift_X.append(ep); shift_Y.append(eo); shift_pid.append(pid)

    bX = torch.from_numpy(np.stack(bridge_X)).float().to(device)  # (N, 192)
    bY = torch.from_numpy(np.stack(bridge_Y)).float().to(device)  # (N, 256)
    bP = np.array(bridge_pid)
    sX = torch.from_numpy(np.stack(shift_X)).float().to(device)
    sY = torch.from_numpy(np.stack(shift_Y)).float().to(device)
    sP = np.array(shift_pid)
    print(f'  Bridge: {len(bX)} pairs  |  Shift: {len(sX)} pairs')

    # ---------- Patient-level K-fold for each ----------
    random.Random(args.seed).shuffle(train_pids)
    fold_size = max(1, len(train_pids) // args.k_folds)
    folds_by_pid = [set(train_pids[i*fold_size:(i+1)*fold_size if i < args.k_folds-1 else None])
                    for i in range(args.k_folds)]

    def kfold(X, Y, P, model_factory, name):
        losses = []
        for k, val_set in enumerate(folds_by_pid):
            val_mask = np.array([p in val_set for p in P])
            tr_idx = np.where(~val_mask)[0]
            va_idx = np.where(val_mask)[0]
            mdl = model_factory().to(device)
            loss, _ = train_mlp(mdl, X[tr_idx], Y[tr_idx], X[va_idx], Y[va_idx],
                                args.epochs, args.batch_size, args.lr,
                                args.weight_decay, args.cos_weight, device,
                                tag=f'{name} Fold {k+1}')
            losses.append(loss)
            print(f'{name} Fold {k+1}/{args.k_folds}: val_loss={loss:.4f}')
        m = np.mean(losses); s = np.std(losses)
        print(f'{name} CV: {m:.4f} ± {s:.4f}')
        return m, s

    print(f'\n{"="*60}\n  Bridge K-fold CV\n{"="*60}')
    bridge_mean, bridge_std = kfold(bX, bY, bP, BridgeEcapaToFreeVC, 'Bridge')

    print(f'\n{"="*60}\n  Shift K-fold CV\n{"="*60}')
    shift_mean, shift_std = kfold(sX, sY, sP, ShiftEcapa, 'Shift')

    # ---------- Final models on all training data ----------
    print(f'\nTraining final models on all training data...')
    rng = np.random.default_rng(args.seed)

    def final(X, Y, model_factory, name):
        idx = np.arange(len(X)); rng.shuffle(idx)
        n_mon = max(1, int(0.05 * len(idx)))
        m = model_factory().to(device)
        _, state = train_mlp(m, X[idx[n_mon:]], Y[idx[n_mon:]],
                             X[idx[:n_mon]], Y[idx[:n_mon]],
                             args.epochs, args.batch_size, args.lr,
                             args.weight_decay, args.cos_weight, device, tag=name)
        return state

    bridge_state = final(bX, bY, BridgeEcapaToFreeVC, 'Bridge Final')
    shift_state  = final(sX, sY, ShiftEcapa,          'Shift Final')

    torch.save({'state_dict': bridge_state,
                'config': {'in_dim': 192, 'out_dim': 256,
                           'hidden_dim': 256, 'n_blocks': 2, 'dropout': 0.2},
                'cv_val_loss_mean': bridge_mean, 'cv_val_loss_std': bridge_std,
                }, os.path.join(args.out_dir, 'bridge.pt'))
    torch.save({'state_dict': shift_state,
                'config': {'emb_dim': 192, 'hidden_dim': 128,
                           'n_blocks': 2, 'dropout': 0.3},
                'cv_val_loss_mean': shift_mean, 'cv_val_loss_std': shift_std,
                }, os.path.join(args.out_dir, 'shift.pt'))
    print('\nSaved bridge.pt, shift.pt')

    # Also save population averages for ablation baselines
    avg_pre_ecapa  = norm(np.stack([e for pid in train_pids for e, _ in data[pid]['pre']]).mean(0))
    avg_post_ecapa = norm(np.stack([e for pid in train_pids for e, _ in data[pid]['post']]).mean(0))
    avg_post_freevc = norm(np.stack([s for pid in train_pids for _, s in data[pid]['post']]).mean(0))
    np.save(os.path.join(args.out_dir, 'avg_pre_ecapa.npy'), avg_pre_ecapa)
    np.save(os.path.join(args.out_dir, 'avg_post_ecapa.npy'), avg_post_ecapa)
    np.save(os.path.join(args.out_dir, 'avg_post_freevc_spk.npy'), avg_post_freevc)
    print('Saved population averages for baseline comparison.')


if __name__ == '__main__':
    main()
