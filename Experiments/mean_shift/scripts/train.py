"""
Mean Shift Voice Conversion — Training (Compute Domain Statistics)

Learns the simplest possible domain transform: shift all WavLM features
by the difference in domain means.

    converted = source_features + (mean_post - mean_pre)

No neural network, no training loop — just compute and save two mean vectors.

Usage:
    python scripts/train.py
"""

import argparse
import os
import sys
import glob
import torch
import torchaudio
import numpy as np
from pathlib import Path


SAMPLE_RATE = 16000


def load_knnvc(device):
    """Load kNN-VC model (WavLM encoder + HiFi-GAN vocoder)."""
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)
    return knn_vc


def extract_all_features(knn_vc, wav_dir, sr=SAMPLE_RATE):
    """Extract WavLM features from all WAV files in a directory."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    all_features = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)  # (T, 1024)
        all_features.append(features.cpu())
        n_frames = features.shape[0]
        duration = n_frames * 0.02
        print(f"  {Path(wf).name}: {n_frames} frames ({duration:.1f}s)")

    combined = torch.cat(all_features, dim=0)  # (N_total, 1024)
    print(f"  Total: {combined.shape[0]} frames ({combined.shape[0] * 0.02 / 60:.1f} min)")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Mean Shift VC — Compute domain statistics")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1",
                        help='Directory with pre-surgery WAV files')
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2",
                        help='Directory with post-surgery WAV files')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'domain_stats.pt'),
                        help='Output path for domain statistics')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC (WavLM encoder)
    print("Loading kNN-VC model...")
    knn_vc = load_knnvc(device)

    # Extract features from both domains
    print(f"\nExtracting pre-surgery features from {args.pre_dir}")
    features_pre = extract_all_features(knn_vc, args.pre_dir)

    print(f"\nExtracting post-surgery features from {args.post_dir}")
    features_post = extract_all_features(knn_vc, args.post_dir)

    # Compute domain means
    mean_pre = features_pre.mean(dim=0)    # (1024,)
    mean_post = features_post.mean(dim=0)  # (1024,)
    delta = mean_post - mean_pre           # (1024,)

    print(f"\nDomain statistics:")
    print(f"  Pre-surgery mean norm:  {mean_pre.norm():.4f}")
    print(f"  Post-surgery mean norm: {mean_post.norm():.4f}")
    print(f"  Delta norm:             {delta.norm():.4f}")
    print(f"  Delta L1:               {delta.abs().mean():.6f}")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'mean_pre': mean_pre,
        'mean_post': mean_post,
        'delta': delta,
        'n_frames_pre': features_pre.shape[0],
        'n_frames_post': features_post.shape[0],
    }, args.output)
    print(f"\nSaved domain statistics to {args.output}")


if __name__ == '__main__':
    main()
