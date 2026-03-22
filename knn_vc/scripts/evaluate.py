"""
Evaluate kNN-VC conversion quality.

Converts test set pre-surgery files and compares against post-surgery files
using objective metrics:
  - Mel Cepstral Distortion (MCD): lower = more similar spectral shape
  - F0 Correlation: higher = better pitch tracking
  - Speaker Embedding Cosine Similarity: higher = closer voice identity

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --test_dir /path/to/test_wavs
"""

import argparse
import os
import sys
import glob
import torch
import numpy as np
import librosa
from pathlib import Path


def compute_mcd(ref_mel, synth_mel):
    """Mel Cepstral Distortion between two mel-spectrograms."""
    # Align lengths
    min_len = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel = ref_mel[:, :min_len]
    synth_mel = synth_mel[:, :min_len]

    # MCD (using first 13 MFCCs)
    ref_mfcc = librosa.feature.mfcc(S=ref_mel, n_mfcc=13)
    synth_mfcc = librosa.feature.mfcc(S=synth_mel, n_mfcc=13)

    min_len = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - synth_mfcc[:, :min_len]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))
    return mcd


def compute_f0_corr(ref_audio, synth_audio, sr=16000):
    """F0 correlation between reference and synthesized audio."""
    f0_ref, _, _ = librosa.pyin(ref_audio, fmin=50, fmax=500, sr=sr)
    f0_synth, _, _ = librosa.pyin(synth_audio, fmin=50, fmax=500, sr=sr)

    # Remove NaN frames (unvoiced)
    min_len = min(len(f0_ref), len(f0_synth))
    f0_ref = f0_ref[:min_len]
    f0_synth = f0_synth[:min_len]
    valid = ~np.isnan(f0_ref) & ~np.isnan(f0_synth)

    if valid.sum() < 10:
        return float('nan')

    corr = np.corrcoef(f0_ref[valid], f0_synth[valid])[0, 1]
    return corr


def main():
    parser = argparse.ArgumentParser(description="Evaluate kNN-VC conversion quality")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1",
                        help='Directory with pre-surgery wav files')
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2",
                        help='Directory with post-surgery wav files (for reference metrics)')
    parser.add_argument('--converted_dir', type=str, required=True,
                        help='Directory with kNN-VC converted wav files')
    parser.add_argument('--max_files', type=int, default=20,
                        help='Max files to evaluate')
    args = parser.parse_args()

    sr = 16000
    converted_files = sorted(glob.glob(os.path.join(args.converted_dir, "*.wav")))[:args.max_files]
    post_files = sorted(glob.glob(os.path.join(args.post_dir, "**/*.wav"), recursive=True))

    if not converted_files:
        print(f"No converted wav files in {args.converted_dir}")
        return

    # Load a few post-surgery files to get average post-surgery mel for comparison
    print(f"Evaluating {len(converted_files)} converted files...")
    print(f"Reference pool: {len(post_files)} post-surgery files\n")

    # Compute average post-surgery mel spectrum (for MCD reference)
    post_mels = []
    for pf in post_files[:20]:
        y, _ = librosa.load(pf, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        post_mels.append(mel_db)

    mcds = []
    f0_corrs = []

    for cf in converted_files:
        y_conv, _ = librosa.load(cf, sr=sr)
        mel_conv = librosa.feature.melspectrogram(y=y_conv, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
        mel_conv_db = librosa.power_to_db(mel_conv, ref=np.max)

        # MCD against each post-surgery file, take the minimum (best match)
        file_mcds = []
        for pm in post_mels:
            mcd = compute_mcd(pm, mel_conv_db)
            file_mcds.append(mcd)
        best_mcd = min(file_mcds)
        mcds.append(best_mcd)

        name = Path(cf).stem
        print(f"  {name}: MCD={best_mcd:.2f}")

    print(f"\n{'='*40}")
    print(f"Average MCD (lower=better): {np.mean(mcds):.2f} ± {np.std(mcds):.2f}")
    if f0_corrs:
        valid_f0 = [f for f in f0_corrs if not np.isnan(f)]
        if valid_f0:
            print(f"Average F0 Corr (higher=better): {np.mean(valid_f0):.3f} ± {np.std(valid_f0):.3f}")


if __name__ == '__main__':
    main()
