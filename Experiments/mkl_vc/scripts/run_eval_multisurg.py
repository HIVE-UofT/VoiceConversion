"""
MKL-VC — Multi-Surgery Evaluation

Same as run_eval.py but the OT map is computed from ALL four surgery types:
  Tonsill (train patients only, test excluded) + Fess + Sept + Contr

Tests on the same 5 Tonsill held-out patients to check if extra surgery data helps.

Usage:
    python scripts/run_eval_multisurg.py
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from scipy.linalg import sqrtm

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
    get_all_audio_pairs
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test_multisurg')
K = 2
EXTRA_SURGERIES = ["Fess", "Sept", "Contr"]


def extract_all_features(knn_vc, wav_paths):
    all_f = []
    for wav_path in wav_paths:
        f = knn_vc.get_features(str(wav_path)).cpu()
        all_f.append(f)
    combined = torch.cat(all_f, dim=0)
    print(f"  Total: {combined.shape[0]} frames")
    return combined


def compute_mkl_map(X_source, X_target, k=K):
    X_s = X_source.numpy().astype("float64")
    X_t = X_target.numpy().astype("float64")
    D   = X_s.shape[1]

    var_s     = np.var(X_s, axis=0)
    dim_order = np.argsort(-var_s)
    X_s = X_s[:, dim_order]
    X_t = X_t[:, dim_order]

    mu_s = X_s.mean(0)
    mu_t = X_t.mean(0)
    X_s -= mu_s;  X_t -= mu_t

    A_blocks = []
    for g in range(D // k):
        s, e = g * k, (g + 1) * k
        Ss = X_s[:, s:e].T @ X_s[:, s:e] / (X_s.shape[0] - 1) + 1e-6 * np.eye(k)
        St = X_t[:, s:e].T @ X_t[:, s:e] / (X_t.shape[0] - 1) + 1e-6 * np.eye(k)
        Ss_sqrt = sqrtm(Ss).real
        Ss_inv  = np.linalg.inv(Ss_sqrt)
        A = Ss_inv @ sqrtm(Ss_sqrt @ St @ Ss_sqrt).real @ Ss_inv
        A_blocks.append(torch.from_numpy(A).float())

    if D % k:
        A_blocks.append(torch.eye(D % k))

    return (torch.from_numpy(mu_s).float(),
            torch.from_numpy(mu_t).float(),
            A_blocks,
            torch.from_numpy(dim_order).long())


def apply_mkl(features, mu_s, mu_t, A_blocks, dim_order):
    dev = features.device
    x = features[:, dim_order] - mu_s.to(dev)
    parts, idx = [], 0
    for A in A_blocks:
        bs = A.shape[0]
        parts.append(x[:, idx:idx + bs] @ A.to(dev).t())
        idx += bs
    y = torch.cat(parts, dim=1) + mu_t.to(dev)
    return y[:, torch.argsort(dim_order)]


def collect_paths():
    """Collect (pre, post) paths from all surgery types."""
    pre_paths, post_paths = [], []
    tonsill = get_all_audio_pairs("Tonsill", exclude=TEST_PATIENTS)
    for pid in sorted(tonsill):
        for pre, post in tonsill[pid]:
            pre_paths.append(pre)
            post_paths.append(post)
    print(f"  Tonsill (train):  {len(pre_paths)} file pairs")
    for surg in EXTRA_SURGERIES:
        pairs = get_all_audio_pairs(surg)
        n_before = len(pre_paths)
        for pid in sorted(pairs):
            for pre, post in pairs[pid]:
                pre_paths.append(pre)
                post_paths.append(post)
        print(f"  {surg}: {len(pre_paths) - n_before} file pairs")
    return pre_paths, post_paths


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    knn_vc = load_finetuned_knnvc(device)

    print(f"\nCollecting training pairs (Tonsill train + {EXTRA_SURGERIES})...")
    pre_paths, post_paths = collect_paths()
    print(f"  Total: {len(pre_paths)} file pairs")

    print("\nExtracting training pre features...")
    feats_pre  = extract_all_features(knn_vc, pre_paths)
    print("\nExtracting training post features...")
    feats_post = extract_all_features(knn_vc, post_paths)

    print(f"\nComputing MKL map (K={K})...")
    mu_s, mu_t, A_blocks, dim_order = compute_mkl_map(feats_pre, feats_post, k=K)
    print("  Done.")

    # Test patient files (Tonsill only)
    test_pre  = {pid: p for pid, p in
                 get_wav_files(surgery="Tonsill", session="1").items()
                 if pid in TEST_PATIENTS}
    test_post = {pid: p for pid, p in
                 get_wav_files(surgery="Tonsill", session="2").items()
                 if pid in TEST_PATIENTS}

    print("\nLoading ECAPA-TDNN...")
    ecapa = load_ecapa(device)

    pids, sims_conv, sims_base = [], [], []
    for pid in sorted(test_pre):
        pre_path  = test_pre[pid]
        post_path = test_post[pid]
        out_path  = os.path.join(OUT_DIR, f"{pid}_mklvc_multisurg.wav")

        features  = knn_vc.get_features(str(pre_path))
        converted = apply_mkl(features, mu_s, mu_t, A_blocks, dim_order)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("MKL-VC (MultiSurg)", pids, sims_conv, sims_base)

    # Training patients (Tonsill only)
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} Tonsill training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        features  = knn_vc.get_features(str(pre_path))
        converted = apply_mkl(features, mu_s, mu_t, A_blocks, dim_order)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("MKL-VC (MultiSurg) [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
