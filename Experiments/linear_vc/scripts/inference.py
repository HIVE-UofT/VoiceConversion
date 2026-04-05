"""
LinearVC — Inference

Converts audio by applying the learned linear projection matrix W
to WavLM features: converted = source_features @ W

No reference audio needed at inference.

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


def convert_file(knn_vc, W, input_path, output_path):
    """Convert a single file by applying linear transform to WavLM features."""
    features = knn_vc.get_features(input_path)  # (T, 1024)

    # Apply linear transform
    converted_features = features @ W.to(features.device)  # (T, 1024)

    # Vocode
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()

    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="LinearVC — Inference")
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--transform', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'linear_transform.pt'))
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='A2B=pre->post, B2A=post->pre')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load transform
    data = torch.load(args.transform, map_location=device, weights_only=True)
    W = data['W_a2b'] if args.direction == 'A2B' else data['W_b2a']
    print(f"Loaded linear transform (direction: {args.direction}, norm: {W.norm():.4f})")

    # Convert
    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, W, args.input, args.output)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, W, wf, out_path)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
