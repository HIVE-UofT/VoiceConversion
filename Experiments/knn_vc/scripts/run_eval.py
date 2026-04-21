"""
kNN-VC — Test Set Evaluation with Fine-Tuned HiFi-GAN

1. Builds the matching set from the 23 TRAINING post-surgery patients only
   (the 5 test patients are excluded, preventing data leakage).
2. Converts each test patient's pre-surgery recording via k-NN feature matching.
3. Synthesises audio with the fine-tuned HiFi-GAN.
4. Evaluates ECAPA-TDNN cosine similarity: converted→post vs baseline pre→post.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --topk 4
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

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')


def build_matching_set(knn_vc, post_paths, device):
    all_feats = []
    for wav_path in post_paths:
        feats = knn_vc.get_features(str(wav_path))   # (T, 1024)
        all_feats.append(feats.cpu())
    ms = torch.cat(all_feats, dim=0)
    print(f"  Matching set: {ms.shape[0]:,} frames ({ms.shape[0]*0.02/60:.1f} min)")
    return ms.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load kNN-VC with fine-tuned HiFi-GAN
    knn_vc = load_finetuned_knnvc(device)

    # Build matching set from TRAINING patients only (all audio types)
    print(f"\nBuilding matching set (excluding {TEST_PATIENTS})...")
    _train_pairs = get_all_audio_pairs("Tonsill", exclude=TEST_PATIENTS)
    train_post_paths = [post for pid in sorted(_train_pairs) for _, post in _train_pairs[pid]]
    print(f"Using {len(train_post_paths)} training files for matching set")
    matching_set = build_matching_set(knn_vc, train_post_paths, device)

    # Test patient files
    test_pre  = {pid: p for pid, p in
                 get_wav_files(surgery="Tonsill", session="1").items()
                 if pid in TEST_PATIENTS}
    test_post = {pid: p for pid, p in
                 get_wav_files(surgery="Tonsill", session="2").items()
                 if pid in TEST_PATIENTS}

    # Load ECAPA
    print("\nLoading ECAPA-TDNN...")
    ecapa = load_ecapa(device)

    pids, sims_conv, sims_base = [], [], []
    for pid in sorted(test_pre):
        pre_path  = test_pre[pid]
        post_path = test_post[pid]
        out_path  = os.path.join(OUT_DIR, f"{pid}_knnvc.wav")

        # kNN-VC conversion + synthesis with fine-tuned HiFi-GAN
        query_seq = knn_vc.get_features(str(pre_path))         # (T, 1024)
        out_wav   = knn_vc.match(query_seq, matching_set,
                                  topk=args.topk)              # (T_audio,)
        torchaudio.save(out_path, out_wav.unsqueeze(0).cpu(), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0).cpu(), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("kNN-VC", pids, sims_conv, sims_base)

    # ── Training patients evaluation ──────────────────────────────────────────────
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} training patients...")
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

    print_ecapa_summary("kNN-VC [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f"\nConverted files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
