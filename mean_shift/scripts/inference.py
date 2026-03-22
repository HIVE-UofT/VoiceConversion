"""
Mean Shift Voice Conversion — Inference

Converts pre-surgery audio to post-surgery by shifting WavLM features
by the pre-computed domain mean difference.

No reference audio needed at inference — the shift is fixed.

Usage:
    python scripts/inference.py --input source.wav --output converted.wav
    python scripts/inference.py --input_dir /path/to/wavs/ --output_dir /path/to/converted/
"""

import argparse
import os
import glob
import torch
import torchaudio
from pathlib import Path


SAMPLE_RATE = 16000


def convert_file(knn_vc, delta, input_path, output_path):
    """Convert a single file by shifting WavLM features."""
    features = knn_vc.get_features(input_path)  # (T, 1024)

    # Apply mean shift
    converted_features = features + delta.to(features.device)

    # Vocode back to audio using HiFi-GAN
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()  # vocode expects (1, T, 1024)

    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Mean Shift VC — Inference")
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--stats', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'domain_stats.pt'),
                        help='Path to domain_stats.pt')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='A2B=pre->post, B2A=post->pre')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load domain statistics
    stats = torch.load(args.stats, map_location=device, weights_only=True)
    delta = stats['delta']  # (1024,)
    if args.direction == 'B2A':
        delta = -delta
    print(f"Loaded domain stats (delta norm: {delta.norm():.4f})")
    print(f"Direction: {'pre -> post' if args.direction == 'A2B' else 'post -> pre'}")

    # Convert
    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, delta, args.input, args.output)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, delta, wf, out_path)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
