"""
Train SurgeryShift: a small MLP mapping pre -> post speaker embedding in
FreeVC's speaker encoder space. Entire FreeVC foundation stays frozen.

For each (pre_wav, post_wav) pair in training patients:
  target_post_spk = SpeakerEncoder(post_wav)    (frozen, precomputed)
  pre_spk         = SpeakerEncoder(pre_wav)     (frozen, precomputed)
  predicted       = shift_net(pre_spk)
  loss            = MSE(predicted, target_post_spk) + cos_loss

Data augmentation: multiple temporal crops per utterance → many embeddings.
K-fold CV on train patients for model selection; final model trained on all.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

# FreeVC speaker encoder lives in the sibling free_vc/FreeVC dir
FREEVC_DIR = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc/FreeVC'
sys.path.insert(0, FREEVC_DIR)
os.chdir(FREEVC_DIR)  # speaker encoder needs relative paths

from speaker_encoder.voice_encoder import SpeakerEncoder

# Now add our dir too so we can import the shift model
SHIFT_DIR = '/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_shift'
sys.path.insert(0, os.path.join(SHIFT_DIR, 'scripts'))
from shift_model import SurgeryShift

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
    wav, _ = librosa.load(path, sr=sr)
    return wav


def extract_crops_embeddings(smodel, wav_path, n_crops=8, crop_sec=3.0, sr=SR):
    """Extract n_crops evenly-spaced speaker embeddings from a single audio file."""
    wav = load_wav_np(wav_path, sr)
    crop_samples = int(crop_sec * sr)
    total = len(wav)
    if total <= crop_samples:
        return [smodel.embed_utterance(wav)]
    starts = np.linspace(0, total - crop_samples, n_crops).astype(int)
    return [smodel.embed_utterance(wav[s:s + crop_samples]) for s in starts]


def normalise(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


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
    ap.add_argument('--hidden_dim', type=int, default=128)
    ap.add_argument('--n_blocks', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str,
                    default='/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion/Experiments/free_vc_shift/checkpoints')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------ Data ------------
    pairs = collect_pairs(args.surgery)
    train_pids = sorted(p for p in pairs if p not in TEST_PATIENTS)
    test_pids  = sorted(p for p in pairs if p in TEST_PATIENTS)
    print(f'Train patients: {len(train_pids)} | Test (held out): {sorted(test_pids)}')

    # Extract embeddings for all training pairs (many crops per utterance)
    print('\nLoading FreeVC speaker encoder (frozen)...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    print(f'\nExtracting speaker embeddings ({args.n_crops} crops/utt, {args.crop_sec}s each)...')
    train_X, train_Y, train_pid_per_crop = [], [], []  # X=pre, Y=post
    for pid in train_pids:
        pre_embs  = extract_crops_embeddings(smodel, pairs[pid]['pre'],
                                              args.n_crops, args.crop_sec)
        post_embs = extract_crops_embeddings(smodel, pairs[pid]['post'],
                                              args.n_crops, args.crop_sec)
        # Pair cross-product (every pre crop with every post crop) → surgery direction
        # averages over temporal variation within each recording.
        for pe in pre_embs:
            for po in post_embs:
                train_X.append(pe); train_Y.append(po); train_pid_per_crop.append(pid)

    X = normalise(np.stack(train_X))     # (N, 256) L2-normed
    Y = normalise(np.stack(train_Y))     # (N, 256)
    pids_arr = np.array(train_pid_per_crop)
    print(f'  Total training pairs: {len(X)} (from {len(train_pids)} patients)')

    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)

    # ------------ K-fold CV ------------
    def train_one(train_idx, val_idx, tag='', epochs=args.epochs, verbose=False):
        shift = SurgeryShift(emb_dim=256, hidden_dim=args.hidden_dim,
                             n_blocks=args.n_blocks, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(shift.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        Xt, Yt = X_t[train_idx], Y_t[train_idx]
        Xv, Yv = X_t[val_idx], Y_t[val_idx]

        best_val, best_state = float('inf'), None
        for ep in range(epochs):
            shift.train()
            perm = torch.randperm(len(Xt), device=device)
            ep_loss = 0.0; n = 0
            for s in range(0, len(Xt), args.batch_size):
                idx = perm[s:s + args.batch_size]
                xb, yb = Xt[idx], Yt[idx]
                yb_pred = shift(xb)
                mse = F.mse_loss(yb_pred, yb)
                cos = 1.0 - F.cosine_similarity(yb_pred, yb, dim=-1).mean()
                loss = mse + args.cos_weight * cos
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(shift.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); n += 1
            sched.step()

            shift.eval()
            with torch.no_grad():
                v = shift(Xv)
                val_mse = F.mse_loss(v, Yv).item()
                val_cos = F.cosine_similarity(v, Yv, dim=-1).mean().item()
                val_loss = val_mse + args.cos_weight * (1 - val_cos)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in shift.state_dict().items()}
            if verbose and ((ep + 1) % 50 == 0 or ep == 0):
                print(f'  {tag} ep {ep+1:3d}  train={ep_loss/n:.4f}  val_mse={val_mse:.4f}  val_cos={val_cos:.4f}')
        return best_val, best_state

    # Patient-level k-fold
    random.Random(args.seed).shuffle(train_pids)
    fold_size = max(1, len(train_pids) // args.k_folds)
    folds_by_pid = [set(train_pids[i*fold_size:(i+1)*fold_size if i < args.k_folds-1 else None])
                    for i in range(args.k_folds)]

    print(f'\n{"="*60}\n  {args.k_folds}-fold CV on {len(train_pids)} train patients\n{"="*60}')
    cv_val_losses = []
    for k, val_set in enumerate(folds_by_pid):
        val_mask  = np.array([pid in val_set for pid in pids_arr])
        train_idx = np.where(~val_mask)[0]
        val_idx   = np.where(val_mask)[0]
        val_loss, _ = train_one(train_idx, val_idx, tag=f'Fold {k+1}', verbose=True)
        cv_val_losses.append(val_loss)
        print(f'Fold {k+1}/{args.k_folds}: best val_loss = {val_loss:.4f}')
    print(f'\nCV mean val_loss: {np.mean(cv_val_losses):.4f} (+/- {np.std(cv_val_losses):.4f})')

    # ------------ Final model on all training data ------------
    print(f'\nTraining final model on all {len(train_pids)} train patients...')
    # No real val set for final; hold out a tiny random slice just for early-stop monitoring
    all_idx = np.arange(len(X))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_idx)
    n_mon = max(1, int(0.05 * len(all_idx)))
    val_idx = all_idx[:n_mon]
    train_idx = all_idx[n_mon:]
    _, final_state = train_one(train_idx, val_idx, tag='Final', verbose=True)

    # Save checkpoint
    ckpt_path = os.path.join(args.out_dir, 'shift.pt')
    torch.save({
        'state_dict': final_state,
        'config': {
            'emb_dim': 256,
            'hidden_dim': args.hidden_dim,
            'n_blocks': args.n_blocks,
            'dropout': args.dropout,
        },
        'cv_val_loss_mean': float(np.mean(cv_val_losses)),
        'cv_val_loss_std':  float(np.std(cv_val_losses)),
    }, ckpt_path)
    print(f'\nSaved shift checkpoint: {ckpt_path}')

    # Also save avg_post and avg_pre for comparison baselines
    avg_pre  = X.mean(axis=0); avg_pre  /= np.linalg.norm(avg_pre)  + 1e-8
    avg_post = Y.mean(axis=0); avg_post /= np.linalg.norm(avg_post) + 1e-8
    np.save(os.path.join(args.out_dir, 'avg_pre_spk.npy'),  avg_pre)
    np.save(os.path.join(args.out_dir, 'avg_post_spk.npy'), avg_post)
    print(f'Saved avg_pre_spk, avg_post_spk for baseline comparison.')


if __name__ == '__main__':
    main()
