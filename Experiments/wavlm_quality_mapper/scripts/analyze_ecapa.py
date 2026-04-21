"""
ECAPA-TDNN Speaker Analysis (Speech segments only)

Segments each speech recording into chunks and compares:
  - Pre vs Pre:   chunks within same pre-surgery recording (same-speaker baseline)
  - Post vs Post: chunks within same post-surgery recording (same-speaker baseline, post)
  - Pre vs Post:  chunks from pre vs post recordings (surgery effect)
  - Between-speaker: chunks from different patients (different-speaker reference)

The gap between pre-vs-pre and pre-vs-post quantifies the voice change from surgery.
Post-vs-Post shows whether intra-speaker consistency is preserved after surgery.

Usage:
    python scripts/analyze_ecapa.py
    python scripts/analyze_ecapa.py --surgery Tonsill
    python scripts/analyze_ecapa.py --surgery Tonsill --chunk_sec 5
"""

import argparse
import os
import glob
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"
SAMPLE_RATE = 16000
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots', 'analysis')


def load_and_segment(wav_path, chunk_sec, sample_rate=SAMPLE_RATE):
    """Load audio and split into non-overlapping chunks."""
    sig, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        sig = torchaudio.functional.resample(sig, sr, sample_rate)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)

    chunk_samples = int(chunk_sec * sample_rate)
    total_samples = sig.shape[1]
    duration = total_samples / sample_rate

    chunks = []
    for start in range(0, total_samples - chunk_samples + 1, chunk_samples):
        chunks.append(sig[:, start:start + chunk_samples])

    return chunks, duration


def get_chunk_embeddings(ecapa, chunks, device):
    """Get ECAPA embedding for each chunk."""
    embeddings = []
    for chunk in chunks:
        emb = ecapa.encode_batch(chunk.to(device)).squeeze().cpu()
        embeddings.append(emb)
    return embeddings


def pairwise_cosine_sims(embs_a, embs_b):
    """Compute all pairwise cosine similarities between two lists of embeddings."""
    sims = []
    for a in embs_a:
        for b in embs_b:
            sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            sims.append(sim)
    return sims


def within_cosine_sims(embs):
    """Compute all pairwise cosine similarities within a single list."""
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
            sims.append(sim)
    return sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--chunk_sec', type=float, default=5.0,
                        help='Chunk duration in seconds (default: 5)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Chunk duration: {args.chunk_sec}s")
    os.makedirs(PLOT_DIR, exist_ok=True)

    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading ECAPA-TDNN...")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/lustre06/project/6086959/sepharfi/pretrained_models/ecapa-voxceleb",
        run_opts={"device": str(device)})

    base = os.path.join(CUCO_BASE, args.surgery)

    # Gather Speech files (session 1 = pre, session 2 = post)
    pre_files = sorted(glob.glob(os.path.join(base, "Speech", "1", "*.wav")))
    post_files = sorted(glob.glob(os.path.join(base, "Speech", "2", "*.wav")))

    def get_patient_id(path):
        return Path(path).stem.split('_')[-1]

    pre_by_patient = {get_patient_id(f): f for f in pre_files}
    post_by_patient = {get_patient_id(f): f for f in post_files}

    patients_with_both = sorted(set(pre_by_patient) & set(post_by_patient))
    print(f"Patients with pre: {len(pre_by_patient)}")
    print(f"Patients with post: {len(post_by_patient)}")
    print(f"Patients with both: {len(patients_with_both)}")

    # Segment and extract embeddings
    print("\nSegmenting and extracting ECAPA embeddings...")
    pre_chunk_embs = {}   # pid -> list of embeddings
    post_chunk_embs = {}

    for pid in patients_with_both:
        chunks_pre, dur_pre = load_and_segment(pre_by_patient[pid], args.chunk_sec)
        chunks_post, dur_post = load_and_segment(post_by_patient[pid], args.chunk_sec)

        pre_chunk_embs[pid] = get_chunk_embeddings(ecapa, chunks_pre, device)
        post_chunk_embs[pid] = get_chunk_embeddings(ecapa, chunks_post, device)

        print(f"  Patient {pid}: pre {dur_pre:.1f}s ({len(chunks_pre)} chunks), "
              f"post {dur_post:.1f}s ({len(chunks_post)} chunks)")

    # ══════════════════════════════════════════════
    #  Analysis 1: Pre vs Pre (same-speaker baseline)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Pre vs Pre (same speaker baseline, within-recording)")
    print("=" * 60)

    pre_pre_per_patient = {}
    all_pre_pre = []

    for pid in patients_with_both:
        sims = within_cosine_sims(pre_chunk_embs[pid])
        if sims:
            pre_pre_per_patient[pid] = sims
            all_pre_pre.extend(sims)
            print(f"  Patient {pid}: {np.mean(sims):.4f} +/- {np.std(sims):.4f} ({len(sims)} pairs)")

    pre_pre_mean = np.mean(all_pre_pre)
    print(f"\n  Overall pre vs pre: {pre_pre_mean:.4f} +/- {np.std(all_pre_pre):.4f}")

    # ══════════════════════════════════════════════
    #  Analysis 2: Post vs Post (same-speaker baseline, post-surgery)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Post vs Post (same speaker baseline, within-post-recording)")
    print("=" * 60)

    post_post_per_patient = {}
    all_post_post = []

    for pid in patients_with_both:
        sims = within_cosine_sims(post_chunk_embs[pid])
        if sims:
            post_post_per_patient[pid] = sims
            all_post_post.extend(sims)
            print(f"  Patient {pid}: {np.mean(sims):.4f} +/- {np.std(sims):.4f} ({len(sims)} pairs)")

    post_post_mean = np.mean(all_post_post)
    print(f"\n  Overall post vs post: {post_post_mean:.4f} +/- {np.std(all_post_post):.4f}")

    # ══════════════════════════════════════════════
    #  Analysis 3: Pre vs Post (surgery effect)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Pre vs Post (same patient, cross-session surgery effect)")
    print("=" * 60)

    pre_post_per_patient = {}
    all_pre_post = []

    for pid in patients_with_both:
        sims = pairwise_cosine_sims(pre_chunk_embs[pid], post_chunk_embs[pid])
        if sims:
            pre_post_per_patient[pid] = sims
            all_pre_post.extend(sims)
            print(f"  Patient {pid}: {np.mean(sims):.4f} +/- {np.std(sims):.4f} ({len(sims)} pairs)")

    pre_post_mean = np.mean(all_pre_post)
    print(f"\n  Overall pre vs post: {pre_post_mean:.4f} +/- {np.std(all_pre_post):.4f}")

    # ══════════════════════════════════════════════
    #  Analysis 4: Between-speaker (different patients, pre chunks)
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Between-Speaker (different patients, pre-surgery)")
    print("=" * 60)

    all_between = []
    pids = list(patients_with_both)
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            sims = pairwise_cosine_sims(pre_chunk_embs[pids[i]], pre_chunk_embs[pids[j]])
            all_between.extend(sims)

    between_mean = np.mean(all_between)
    print(f"  Between-speaker similarity: {between_mean:.4f} +/- {np.std(all_between):.4f}")

    # ══════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  ECAPA Summary (Speech, {:.0f}s chunks)".format(args.chunk_sec))
    print("=" * 60)
    print(f"  Pre vs Pre    (same speaker, pre):     {pre_pre_mean:.4f} +/- {np.std(all_pre_pre):.4f}")
    print(f"  Post vs Post  (same speaker, post):   {post_post_mean:.4f} +/- {np.std(all_post_post):.4f}")
    print(f"  Pre vs Post   (surgery effect):       {pre_post_mean:.4f} +/- {np.std(all_pre_post):.4f}")
    print(f"  Between-speaker (diff person):        {between_mean:.4f} +/- {np.std(all_between):.4f}")
    print(f"  Voice change   (pre_pre - pre_post):  {pre_pre_mean - pre_post_mean:+.4f}")
    print(f"  Post stability (post_post - pre_post):{post_post_mean - pre_post_mean:+.4f}")
    print(f"  Speaker discr  (pre_post - between):  {pre_post_mean - between_mean:+.4f}")

    # ══════════════════════════════════════════════
    #  Plots
    # ══════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Generating Plots")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Four-way boxplot
    ax = axes[0]
    data = [all_pre_pre, all_post_post, all_pre_post, all_between]
    labels_box = ['Pre vs Pre\n(pre baseline)', 'Post vs Post\n(post baseline)',
                  'Pre vs Post\n(surgery)', 'Between\nspeakers']
    colors = ['steelblue', 'mediumpurple', 'seagreen', 'coral']
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(labels_box, fontsize=8)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"ECAPA-TDNN Speaker Similarity (Speech, {args.chunk_sec}s chunks)")
    for i, d in enumerate(data):
        ax.text(i + 1, np.median(d) + 0.01, f"μ={np.mean(d):.3f}", ha='center', fontsize=9)

    # Plot 2: Per-patient comparison (pre-pre, post-post, pre-post)
    ax = axes[1]
    pids_plot = [pid for pid in patients_with_both
                 if pid in pre_pre_per_patient and pid in post_post_per_patient
                 and pid in pre_post_per_patient]
    patient_pre_pre_means  = [np.mean(pre_pre_per_patient[pid])  for pid in pids_plot]
    patient_post_post_means = [np.mean(post_post_per_patient[pid]) for pid in pids_plot]
    patient_pre_post_means = [np.mean(pre_post_per_patient[pid]) for pid in pids_plot]

    y = np.arange(len(pids_plot))
    height = 0.25
    ax.barh(y - height, patient_pre_pre_means,   height, color='steelblue',    alpha=0.8, label='Pre vs Pre')
    ax.barh(y,          patient_post_post_means,  height, color='mediumpurple', alpha=0.8, label='Post vs Post')
    ax.barh(y + height, patient_pre_post_means,  height, color='seagreen',     alpha=0.8, label='Pre vs Post')
    ax.set_yticks(y)
    ax.set_yticklabels([f"P{pid}" for pid in pids_plot], fontsize=7)
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Per-Patient: Intra-Session vs Surgery Effect")
    ax.axvline(x=between_mean, color='coral', linestyle='--', label=f'Between-spk ({between_mean:.3f})')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_analysis.png'), dpi=150)
    plt.close()
    print(f"  Saved ecapa_analysis.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
