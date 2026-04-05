"""
K-NN Voice Mapper: pre-surgery -> post-surgery embedding.

Instead of training a model, at test time we:
  1. Find the k most similar training patients by pre-surgery ECAPA similarity
  2. Predict the test patient's post-surgery embedding as a weighted average
     of those k neighbours' actual post-surgery embeddings
     (weights = softmax of cosine similarities)

Advantages over MLP with 23 training patients:
  - No training, no overfitting — uses only observed post-surgery embeddings
  - Naturally patient-specific: similar pre-surgery voice → similar post-surgery
  - k and temperature selected by leave-one-out CV on the training pool

Uses ECAPA for retrieval (best absolute similarity scores of all features tried)
and also evaluates librosa acoustic features as an alternative retrieval space.

Usage:
    python knn_mapper.py
    python knn_mapper.py --surgery Tonsill --n_test 5 --seed 42
    python knn_mapper.py --k 3 --temp 0.1
"""

import argparse
import glob
import json
import os
import random
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path

SAMPLE_RATE = 16000
CUCO_BASE   = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"

N_CROPS  = 10
CROP_SEC = 3.0

# k and temperature grid for LOO-CV selection
K_GRID    = [1, 2, 3, 4, 5, 7, 10, 15]
TEMP_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


# ─────────────────────────────────────────────────────────────
# ECAPA extraction
# ─────────────────────────────────────────────────────────────

def load_ecapa():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"})


def extract_ecapa_multicrop(ecapa, wav_path, n_crops, crop_sec):
    sig, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        sig = torchaudio.functional.resample(sig, sr, SAMPLE_RATE)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)

    total       = sig.shape[1]
    crop_frames = int(crop_sec * SAMPLE_RATE)

    if n_crops == 1 or total <= crop_frames:
        with torch.no_grad():
            emb = ecapa.encode_batch(sig)
        return [emb.squeeze().cpu()]

    starts = np.linspace(0, total - crop_frames, n_crops).astype(int)
    embs = []
    for s in starts:
        crop = sig[:, s : s + crop_frames]
        with torch.no_grad():
            emb = ecapa.encode_batch(crop)
        embs.append(emb.squeeze().cpu())
    return embs


# ─────────────────────────────────────────────────────────────
# Librosa acoustic feature extraction
# ─────────────────────────────────────────────────────────────

def _librosa_features(audio: np.ndarray, sr: int) -> torch.Tensor:
    feats = []
    mfcc  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    dmfcc = librosa.feature.delta(mfcc)
    for mat in (mfcc, dmfcc):
        feats.extend(mat.mean(axis=1).tolist())
        feats.extend(mat.std(axis=1).tolist())
    sc = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    feats.extend(sc.mean(axis=1).tolist())
    feats.extend(sc.std(axis=1).tolist())
    for arr in (
        librosa.feature.spectral_centroid(y=audio, sr=sr)[0],
        librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0],
        librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0],
        librosa.feature.spectral_flatness(y=audio)[0],
        librosa.feature.zero_crossing_rate(audio)[0],
        librosa.feature.rms(y=audio)[0],
    ):
        feats.extend([float(arr.mean()), float(arr.std())])
    return torch.tensor(feats, dtype=torch.float32)


def extract_librosa_multicrop(wav_path, n_crops, crop_sec):
    sig, sr = torchaudio.load(wav_path)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        sig = torchaudio.functional.resample(sig, sr, SAMPLE_RATE)
    audio       = sig.squeeze().numpy()
    total       = len(audio)
    crop_frames = int(crop_sec * SAMPLE_RATE)
    if n_crops == 1 or total <= crop_frames:
        return [_librosa_features(audio, SAMPLE_RATE)]
    starts = np.linspace(0, total - crop_frames, n_crops).astype(int)
    return [_librosa_features(audio[s : s + crop_frames], SAMPLE_RATE)
            for s in starts]


# ─────────────────────────────────────────────────────────────
# K-NN prediction
# ─────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def knn_predict(query_pre, db_pre, db_post, k, temp, exclude_idx=None):
    """
    Predict post-surgery embedding for query_pre.

    Args:
        query_pre:   (D,) query pre-surgery embedding
        db_pre:      (N, D) database of pre-surgery embeddings
        db_post:     (N, D) corresponding post-surgery embeddings
        k:           number of neighbours
        temp:        softmax temperature (lower = sharper, higher = more uniform)
        exclude_idx: index to exclude from db (for LOO-CV)

    Returns: (D,) predicted post-surgery embedding
    """
    sims = F.cosine_similarity(
        query_pre.unsqueeze(0).expand(len(db_pre), -1), db_pre)  # (N,)

    if exclude_idx is not None:
        sims[exclude_idx] = -float('inf')

    # Top-k neighbours
    k_eff   = min(k, (sims > -float('inf')).sum().item())
    topk    = torch.topk(sims, k_eff)
    top_idx = topk.indices
    top_sim = topk.values

    # Softmax weights
    weights = F.softmax(top_sim / temp, dim=0)   # (k,)

    # Weighted sum of post-surgery embeddings
    pred = (weights.unsqueeze(1) * db_post[top_idx]).sum(0)  # (D,)
    return pred


def loo_cv(pre_embs, post_embs, k_grid, temp_grid):
    """
    Leave-one-out CV over the training pool to select best (k, temp).
    Returns (best_k, best_temp, best_loo_sim, results_table).
    """
    n        = len(pre_embs)
    db_pre   = torch.stack(pre_embs)
    db_post  = torch.stack(post_embs)

    best_sim   = -float('inf')
    best_k     = k_grid[0]
    best_temp  = temp_grid[0]
    results    = []

    for k in k_grid:
        for temp in temp_grid:
            sims = []
            for i in range(n):
                pred     = knn_predict(db_pre[i], db_pre, db_post,
                                       k, temp, exclude_idx=i)
                sims.append(cosine_sim(pred, db_post[i]))
            mean_sim = np.mean(sims)
            results.append({'k': k, 'temp': temp, 'loo_sim': mean_sim})
            if mean_sim > best_sim:
                best_sim  = mean_sim
                best_k    = k
                best_temp = temp

    return best_k, best_temp, best_sim, results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--n_test',  type=int, default=5)
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--output',  type=str, default=None)
    parser.add_argument('--n_crops', type=int, default=N_CROPS)
    parser.add_argument('--k',       type=int, default=None,
                        help='Fix k instead of LOO-CV selection')
    parser.add_argument('--temp',    type=float, default=None,
                        help='Fix temperature instead of LOO-CV selection')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), f'results_{args.surgery.lower()}_knn')
    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  K-NN Voice Mapper: pre (ses1) -> post (ses2)")
    print(f"  Surgery: {args.surgery}  n_crops={args.n_crops}")
    print(f"{'='*60}")

    # ── Patient split ──
    pre_dir  = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files  = sorted(glob.glob(os.path.join(pre_dir,  "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files)
    n     = len(pre_files)
    names = [Path(f).stem for f in pre_files]

    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    test_idx = sorted(indices[:args.n_test])
    cv_idx   = sorted(indices[args.n_test:])

    print(f"\n{n} patients total")
    print(f"  CV pool ({len(cv_idx)}): {[names[i] for i in cv_idx]}")
    print(f"  Test   ({len(test_idx)}): {[names[i] for i in test_idx]}")

    # ── Extract ECAPA embeddings ──
    print(f"\nLoading ECAPA-TDNN...")
    ecapa = load_ecapa()
    print(f"\nExtracting ECAPA embeddings ({args.n_crops} crops × {CROP_SEC}s)...")

    ecapa_pre, ecapa_post = {}, {}
    for i in range(n):
        ecapa_pre[i]  = torch.stack(extract_ecapa_multicrop(
            ecapa, pre_files[i], args.n_crops, CROP_SEC)).mean(0)
        ecapa_post[i] = torch.stack(extract_ecapa_multicrop(
            ecapa, post_files[i], args.n_crops, CROP_SEC)).mean(0)
        print(f"  {names[i]}")

    # ── Extract librosa features ──
    print(f"\nExtracting librosa acoustic features ({args.n_crops} crops × {CROP_SEC}s)...")
    lib_pre, lib_post = {}, {}
    for i in range(n):
        lib_pre[i]  = torch.stack(extract_librosa_multicrop(
            pre_files[i], args.n_crops, CROP_SEC)).mean(0)
        lib_post[i] = torch.stack(extract_librosa_multicrop(
            post_files[i], args.n_crops, CROP_SEC)).mean(0)
        print(f"  {names[i]}")

    # Normalise librosa features (z-score on CV pool)
    cv_lib = torch.stack([lib_pre[i]  for i in cv_idx] +
                         [lib_post[i] for i in cv_idx])
    lib_mean = cv_lib.mean(0)
    lib_std  = cv_lib.std(0).clamp(min=1e-8)
    for i in range(n):
        lib_pre[i]  = (lib_pre[i]  - lib_mean) / lib_std
        lib_post[i] = (lib_post[i] - lib_mean) / lib_std

    # ── Baseline ──
    baseline_test = [cosine_sim(ecapa_pre[i], ecapa_post[i]) for i in test_idx]
    baseline_cv   = [cosine_sim(ecapa_pre[i], ecapa_post[i]) for i in cv_idx]
    print(f"\nBaseline cosine sim (ECAPA, no transform):")
    print(f"  CV pool: {np.mean(baseline_cv):.4f} ± {np.std(baseline_cv):.4f}")
    print(f"  Test:    {np.mean(baseline_test):.4f} ± {np.std(baseline_test):.4f}")

    # Mean shift
    delta = torch.stack([ecapa_post[i] - ecapa_pre[i] for i in cv_idx]).mean(0)
    ms_test = [cosine_sim(ecapa_pre[i] + delta, ecapa_post[i]) for i in test_idx]
    print(f"  Mean shift test: {np.mean(ms_test):.4f}  "
          f"delta={np.mean(ms_test) - np.mean(baseline_test):+.4f}")

    # ── LOO-CV to select k and temp for each retrieval space ──
    results_all = {}

    for space_name, pre_dict, post_dict in [
        ("ECAPA",   ecapa_pre, ecapa_post),
        ("librosa", lib_pre,   lib_post),
        ("ECAPA-retrieve + ECAPA-predict", ecapa_pre, ecapa_post),  # same
    ]:
        # We only do two meaningful spaces: ECAPA-retrieve and librosa-retrieve
        # both predicting in ECAPA post space
        pass

    # Space 1: retrieve by ECAPA pre, predict ECAPA post
    # Space 2: retrieve by librosa pre, predict ECAPA post
    # Space 3: retrieve by ECAPA pre, predict librosa post (less interesting)

    for space_name, retrieve_pre, predict_post in [
        ("ECAPA→ECAPA",   ecapa_pre, ecapa_post),
        ("librosa→ECAPA", lib_pre,   ecapa_post),
    ]:
        cv_pre  = [retrieve_pre[i]  for i in cv_idx]
        cv_post = [predict_post[i]  for i in cv_idx]

        if args.k is not None and args.temp is not None:
            best_k, best_temp = args.k, args.temp
            best_loo = None
            loo_results = []
        else:
            print(f"\nLOO-CV for {space_name} "
                  f"(k ∈ {K_GRID}, temp ∈ {TEMP_GRID})...")
            best_k, best_temp, best_loo, loo_results = loo_cv(
                cv_pre, cv_post, K_GRID, TEMP_GRID)
            print(f"  Best: k={best_k}  temp={best_temp}  "
                  f"LOO sim={best_loo:.4f}")

        # Predict test patients using full CV pool
        db_pre_t  = torch.stack([retrieve_pre[i]  for i in cv_idx])
        db_post_t = torch.stack([predict_post[i]  for i in cv_idx])

        preds = []
        for i in test_idx:
            pred = knn_predict(retrieve_pre[i], db_pre_t, db_post_t,
                               best_k, best_temp)
            preds.append(pred)

        test_sims = [cosine_sim(preds[j], ecapa_post[test_idx[j]])
                     for j in range(len(test_idx))]

        results_all[space_name] = {
            'best_k':    best_k,
            'best_temp': best_temp,
            'best_loo':  float(best_loo) if best_loo is not None else None,
            'test_sims': test_sims,
            'preds':     preds,
        }

    # ── Test results ──
    print(f"\n{'='*60}")
    print(f"  Test Results")
    print(f"{'='*60}")
    print(f"\n  {'Patient':40s}  {'base':>6}  {'ms':>6}  "
          f"{'knn-ecapa':>10}  {'knn-lib':>9}")
    print(f"  {'-'*40}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*9}")

    knn_ecapa_sims = results_all["ECAPA→ECAPA"]["test_sims"]
    knn_lib_sims   = results_all["librosa→ECAPA"]["test_sims"]

    for j, i in enumerate(test_idx):
        sim_base = baseline_test[j]
        print(f"  {names[i]:40s}  "
              f"{sim_base:6.4f}  "
              f"{ms_test[j]:6.4f}  "
              f"{knn_ecapa_sims[j]:10.4f}  "
              f"{knn_lib_sims[j]:9.4f}")

    mean_base      = np.mean(baseline_test)
    mean_ms        = np.mean(ms_test)
    mean_knn_ecapa = np.mean(knn_ecapa_sims)
    mean_knn_lib   = np.mean(knn_lib_sims)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:        {mean_base:.4f}")
    print(f"  Mean shift:      {mean_ms:.4f}  "
          f"delta={mean_ms - mean_base:+.4f}")
    print(f"  k-NN (ECAPA):    {mean_knn_ecapa:.4f}  "
          f"delta={mean_knn_ecapa - mean_base:+.4f}  "
          f"(k={results_all['ECAPA→ECAPA']['best_k']}  "
          f"temp={results_all['ECAPA→ECAPA']['best_temp']})")
    print(f"  k-NN (librosa):  {mean_knn_lib:.4f}  "
          f"delta={mean_knn_lib - mean_base:+.4f}  "
          f"(k={results_all['librosa→ECAPA']['best_k']}  "
          f"temp={results_all['librosa→ECAPA']['best_temp']})")
    print(f"{'='*60}")

    # ── Save ──
    out_json = os.path.join(args.output, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'surgery':  args.surgery,
            'n_crops':  args.n_crops,
            'n_cv':     len(cv_idx),
            'n_test':   len(test_idx),
            'split': {
                'cv_pool': [names[i] for i in cv_idx],
                'test':    [names[i] for i in test_idx],
            },
            'summary': {
                'baseline':   float(mean_base),
                'mean_shift': float(mean_ms),
                'knn_ecapa':  float(mean_knn_ecapa),
                'knn_librosa': float(mean_knn_lib),
            },
            'knn_ecapa': {
                'best_k':    results_all["ECAPA→ECAPA"]['best_k'],
                'best_temp': results_all["ECAPA→ECAPA"]['best_temp'],
                'loo_sim':   results_all["ECAPA→ECAPA"]['best_loo'],
            },
            'knn_librosa': {
                'best_k':    results_all["librosa→ECAPA"]['best_k'],
                'best_temp': results_all["librosa→ECAPA"]['best_temp'],
                'loo_sim':   results_all["librosa→ECAPA"]['best_loo'],
            },
            'per_patient_test': [
                {
                    'name':        names[test_idx[j]],
                    'baseline':    float(baseline_test[j]),
                    'mean_shift':  float(ms_test[j]),
                    'knn_ecapa':   float(knn_ecapa_sims[j]),
                    'knn_librosa': float(knn_lib_sims[j]),
                }
                for j in range(len(test_idx))
            ],
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")
    print("Done!")


if __name__ == '__main__':
    main()
