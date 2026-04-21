"""
Mean-Shift VC — Test Set Evaluation with Fine-Tuned HiFi-GAN

1. Recomputes domain mean delta from TRAINING patients only (test patients excluded).
2. Converts each test patient's pre-surgery recording.
3. Synthesises audio with the fine-tuned HiFi-GAN.
4. Evaluates ECAPA-TDNN cosine similarity: converted→post vs baseline pre→post.

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

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')


def compute_delta(knn_vc, train_pre_paths, train_post_paths):
    """Compute mean delta (mean_post - mean_pre) from training files only."""
    def mean_feats(paths):
        all_f = []
        for wav_path in paths:
            f = knn_vc.get_features(str(wav_path)).cpu()  # (T, 1024)
            all_f.append(f)
        return torch.cat(all_f, dim=0).mean(dim=0)  # (1024,)

    print(f"\nExtracting training pre-surgery features ({len(train_pre_paths)} files)...")
    mean_pre  = mean_feats(train_pre_paths)
    print(f"\nExtracting training post-surgery features ({len(train_post_paths)} files)...")
    mean_post = mean_feats(train_post_paths)
    delta = mean_post - mean_pre
    print(f"\nDelta norm: {delta.norm():.4f}")
    return delta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    knn_vc = load_finetuned_knnvc(device)

    # Compute delta from TRAINING patients only (all audio types)
    _train_pairs   = get_all_audio_pairs("Tonsill", exclude=TEST_PATIENTS)
    train_pre_paths  = [pre  for pid in sorted(_train_pairs) for pre,  _ in _train_pairs[pid]]
    train_post_paths = [post for pid in sorted(_train_pairs) for _,  post in _train_pairs[pid]]
    print(f"Training files: {len(train_pre_paths)} (excluding {TEST_PATIENTS})")
    delta = compute_delta(knn_vc, train_pre_paths, train_post_paths).to(device)

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
        out_path  = os.path.join(OUT_DIR, f"{pid}_meanshift.wav")

        features  = knn_vc.get_features(str(pre_path))       # (T, 1024)
        converted = features + delta.to(features.device)      # (T, 1024)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()   # (T_audio,)
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("Mean-Shift", pids, sims_conv, sims_base)

    # ── Training patients evaluation ──────────────────────────────────────────────
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        features  = knn_vc.get_features(str(pre_path))
        converted = features + delta.to(features.device)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("Mean-Shift [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
