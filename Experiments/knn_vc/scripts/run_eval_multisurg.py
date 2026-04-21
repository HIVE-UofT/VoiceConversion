"""
kNN-VC — Multi-Surgery Evaluation

Same as run_eval.py but the matching set is built from ALL four surgery types:
  Tonsill (train patients only, test excluded) + Fess + Sept + Contr

Tests on the same 5 Tonsill held-out patients to check if extra surgery data helps.

Usage:
    python scripts/run_eval_multisurg.py
    python scripts/run_eval_multisurg.py --topk 4
"""

import argparse
import os
import sys
import numpy as np
import torch
import torchaudio

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
    get_all_audio_pairs
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test_multisurg')
EXTRA_SURGERIES = ["Fess", "Sept", "Contr"]


def build_matching_set(knn_vc, post_paths, device):
    all_feats = []
    for wav_path in post_paths:
        feats = knn_vc.get_features(str(wav_path))
        all_feats.append(feats.cpu())
    ms = torch.cat(all_feats, dim=0)
    print(f"  Matching set: {ms.shape[0]:,} frames ({ms.shape[0]*0.02/60:.1f} min)")
    return ms.to(device)


def collect_post_paths():
    """Collect post-surgery paths from all surgery types."""
    paths = []
    # Tonsill: exclude test patients
    tonsill = get_all_audio_pairs("Tonsill", exclude=TEST_PATIENTS)
    for pid in sorted(tonsill):
        for _, post in tonsill[pid]:
            paths.append(post)
    print(f"  Tonsill (train):  {len(paths)} files")
    # Other surgeries: all patients
    for surg in EXTRA_SURGERIES:
        pairs = get_all_audio_pairs(surg)
        n_before = len(paths)
        for pid in sorted(pairs):
            for _, post in pairs[pid]:
                paths.append(post)
        print(f"  {surg}: {len(paths) - n_before} files")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    knn_vc = load_finetuned_knnvc(device)

    print(f"\nBuilding matching set (Tonsill train + {EXTRA_SURGERIES})...")
    post_paths = collect_post_paths()
    print(f"  Total: {len(post_paths)} files")
    matching_set = build_matching_set(knn_vc, post_paths, device)

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
        out_path  = os.path.join(OUT_DIR, f"{pid}_knnvc_multisurg.wav")

        query_seq = knn_vc.get_features(str(pre_path))
        out_wav   = knn_vc.match(query_seq, matching_set, topk=args.topk)
        torchaudio.save(out_path, out_wav.unsqueeze(0).cpu(), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0).cpu(), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("kNN-VC (MultiSurg)", pids, sims_conv, sims_base)

    # Training patients
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} Tonsill training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        query_seq = knn_vc.get_features(str(pre_path))
        out_wav   = knn_vc.match(query_seq, matching_set, topk=args.topk)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0).cpu(), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("kNN-VC (MultiSurg) [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
