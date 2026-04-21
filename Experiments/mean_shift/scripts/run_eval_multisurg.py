"""
Mean-Shift VC — Multi-Surgery Evaluation

Same as run_eval.py but the mean delta is computed from ALL four surgery types:
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

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
    get_all_audio_pairs
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test_multisurg')
EXTRA_SURGERIES = ["Fess", "Sept", "Contr"]


def compute_delta(knn_vc, pre_paths, post_paths):
    def mean_feats(paths):
        all_f = []
        for wav_path in paths:
            f = knn_vc.get_features(str(wav_path)).cpu()
            all_f.append(f)
        return torch.cat(all_f, dim=0).mean(dim=0)

    print(f"\nExtracting pre-surgery features ({len(pre_paths)} files)...")
    mean_pre  = mean_feats(pre_paths)
    print(f"\nExtracting post-surgery features ({len(post_paths)} files)...")
    mean_post = mean_feats(post_paths)
    delta = mean_post - mean_pre
    print(f"\nDelta norm: {delta.norm():.4f}")
    return delta


def collect_paths():
    """Collect (pre, post) paths from all surgery types."""
    pre_paths, post_paths = [], []
    # Tonsill: exclude test patients
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
    delta = compute_delta(knn_vc, pre_paths, post_paths).to(device)

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
        out_path  = os.path.join(OUT_DIR, f"{pid}_meanshift_multisurg.wav")

        features  = knn_vc.get_features(str(pre_path))
        converted = features + delta.to(features.device)
        out_wav   = knn_vc.vocode(converted[None]).cpu().squeeze()
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("Mean-Shift (MultiSurg)", pids, sims_conv, sims_base)

    # Training patients (Tonsill only)
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} Tonsill training patients...")
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

    print_ecapa_summary("Mean-Shift (MultiSurg) [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
