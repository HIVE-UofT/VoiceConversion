"""
MKL-VC — Factorized Optimal Transport Voice Conversion (Inference)

Applies the pre-computed MKL optimal transport map to convert WavLM features:
    T(x) = mu_t + A_block @ (x_sorted - mu_s)  (per subgroup)

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


def apply_mkl_transform(features, mu_s, mu_t, A_blocks, dim_order, K):
    """
    Apply factorized MKL optimal transport map to features.

    features: (T, D) tensor
    Returns: (T, D) transformed tensor
    """
    device = features.device
    D = features.shape[1]

    # Reorder dimensions by variance
    x_sorted = features[:, dim_order]  # (T, D)

    # Center
    x_centered = x_sorted - mu_s.to(device).unsqueeze(0)  # (T, D)

    # Apply block-diagonal transform
    out_parts = []
    idx = 0
    for A in A_blocks:
        block_size = A.shape[0]
        x_block = x_centered[:, idx:idx + block_size]  # (T, block_size)
        y_block = x_block @ A.to(device).t()            # (T, block_size)
        out_parts.append(y_block)
        idx += block_size

    y_centered = torch.cat(out_parts, dim=1)  # (T, D)

    # Add target mean
    y_sorted = y_centered + mu_t.to(device).unsqueeze(0)

    # Undo dimension reordering
    inv_order = torch.argsort(dim_order)
    y = y_sorted[:, inv_order]

    return y


def convert_file(knn_vc, transform_data, input_path, output_path, direction='A2B'):
    """Convert a single file using MKL transform."""
    features = knn_vc.get_features(input_path)  # (T, 1024)

    if direction == 'A2B':
        mu_s = transform_data['mu_s']
        mu_t = transform_data['mu_t']
        A_blocks = transform_data['A_blocks']
        dim_order = transform_data['dim_order']
    else:
        mu_s = transform_data['mu_s_rev']
        mu_t = transform_data['mu_t_rev']
        A_blocks = transform_data['A_blocks_rev']
        dim_order = transform_data['dim_order_rev']

    K = transform_data['K']

    converted_features = apply_mkl_transform(features, mu_s, mu_t, A_blocks, dim_order, K)

    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()

    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="MKL-VC — Inference")
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--transform', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'mkl_transform.pt'))
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='A2B=pre->post, B2A=post->pre')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    transform_data = torch.load(args.transform, map_location=device, weights_only=True)
    print(f"Loaded MKL transform (K={transform_data['K']}, direction: {args.direction})")

    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, transform_data, args.input, args.output, args.direction)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, transform_data, wf, out_path, args.direction)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
