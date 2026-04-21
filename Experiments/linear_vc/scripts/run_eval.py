"""
LinearVC — Test Set Evaluation with Fine-Tuned HiFi-GAN

1. Relearns the ridge regression projection W from TRAINING patients only.
2. Converts each test patient's pre-surgery recording via X @ W.
3. Synthesises with the fine-tuned HiFi-GAN.
4. Evaluates ECAPA-TDNN: converted→post vs baseline pre→post.

Usage:
    python scripts/run_eval.py
"""

import os
import sys
import torch
import torchaudio
import numpy as np

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
    get_all_audio_pairs
)

OUT_DIR   = os.path.join(os.path.dirname(__file__), '..', 'converted_test')
REG       = 1e-3   # ridge regularisation strength


def extract_all_features(knn_vc, wav_paths):
    all_f = []
    for wav_path in wav_paths:
        f = knn_vc.get_features(str(wav_path)).cpu()
        all_f.append(f)
    combined = torch.cat(all_f, dim=0)
    print(f"  Total: {combined.shape[0]} frames")
    return combined


def pair_knn(X, Y, chunk=5000):
    """Pair each row in X to its nearest neighbour in Y (cosine similarity)."""
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Yn = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    indices = []
    for i in range(0, len(Xn), chunk):
        sim = Xn[i:i + chunk] @ Yn.t()
        indices.append(sim.argmax(dim=1))
    return X, Y[torch.cat(indices)]


def solve_ridge(X, Y, reg=REG):
    Xnp = X.numpy().astype("float64")
    Ynp = Y.numpy().astype("float64")
    XtX = Xnp.T @ Xnp
    XtY = Xnp.T @ Ynp
    W   = np.linalg.solve(XtX + reg * np.eye(XtX.shape[0]), XtY)
    mse = np.mean((Xnp @ W - Ynp) ** 2)
    print(f"  Training MSE: {mse:.6f}")
    return torch.from_numpy(W).float()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    knn_vc = load_finetuned_knnvc(device)

    # Learn projection from TRAINING patients only (all audio types)
    _train_pairs   = get_all_audio_pairs("Tonsill", exclude=TEST_PATIENTS)
    train_pre_paths  = [pre  for pid in sorted(_train_pairs) for pre,  _ in _train_pairs[pid]]
    train_post_paths = [post for pid in sorted(_train_pairs) for _,  post in _train_pairs[pid]]
    print(f"Training files: {len(train_pre_paths)} (excluding {TEST_PATIENTS})")

    print("\nExtracting training pre features...")
    feats_pre  = extract_all_features(knn_vc, train_pre_paths)
    print("\nExtracting training post features...")
    feats_post = extract_all_features(knn_vc, train_post_paths)

    print("\nPairing frames (cosine NN, pre→post)...")
    X, Y = pair_knn(feats_pre, feats_post)
    print(f"  {X.shape[0]} frame pairs")

    print("\nSolving ridge regression...")
    W = solve_ridge(X, Y, reg=REG).to(device)

    # Test patient files
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
        out_path  = os.path.join(OUT_DIR, f"{pid}_linearvc.wav")

        features  = knn_vc.get_features(str(pre_path))   # (T, 1024)
        converted = features @ W.to(features.device)      # (T, 1024)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("LinearVC", pids, sims_conv, sims_base)

    # ── Training patients evaluation ──────────────────────────────────────────────
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        features  = knn_vc.get_features(str(pre_path))
        converted = features @ W.to(features.device)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("LinearVC [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
