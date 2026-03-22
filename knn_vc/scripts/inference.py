"""
kNN-VC Inference: Convert pre-surgery voice to post-surgery voice.

Uses a pre-computed matching set (no post-surgery audio needed at inference).

Usage:
  python scripts/inference.py --input /path/to/pre_surgery.wav --output /path/to/output.wav
  python scripts/inference.py --input_dir /path/to/wavs/ --output_dir /path/to/converted/
"""

import argparse
import os
import sys
import torch
import torchaudio
from pathlib import Path


def convert_file(knn_vc, matching_set, input_path, output_path, topk=4):
    """Convert a single audio file using kNN-VC."""
    query_seq = knn_vc.get_features(input_path)
    out_wav = knn_vc.match(query_seq, matching_set, topk=topk)

    # out_wav is a 1D tensor at 16kHz
    out_wav = out_wav.unsqueeze(0).cpu()  # (1, T)
    torchaudio.save(output_path, out_wav, 16000)
    duration = out_wav.shape[1] / 16000
    print(f"  {Path(input_path).name} → {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="kNN-VC Inference: Pre→Post Surgery Voice Conversion")
    parser.add_argument('--input', type=str, help='Path to input wav file')
    parser.add_argument('--output', type=str, help='Path to output wav file')
    parser.add_argument('--input_dir', type=str, help='Directory of input wav files')
    parser.add_argument('--output_dir', type=str, help='Directory for output wav files')
    parser.add_argument('--matching_set', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'matching_sets', 'post_surgery_matching_set.pt'),
                        help='Path to pre-computed matching set')
    parser.add_argument('--topk', type=int, default=4,
                        help='Number of nearest neighbors (default: 4)')
    parser.add_argument('--prematched', action='store_true', default=True,
                        help='Use prematched HiFi-GAN variant')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC
    print("Loading kNN-VC...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=args.prematched, device=device)

    # Load pre-computed matching set
    print(f"Loading matching set: {args.matching_set}")
    matching_set = torch.load(args.matching_set, map_location=device, weights_only=True).float()
    n_frames = matching_set.shape[0]
    print(f"  {n_frames:,} frames ({n_frames * 0.02 / 60:.1f} min of post-surgery speech)")

    # Single file
    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, matching_set, args.input, args.output, topk=args.topk)

    # Directory
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        input_dir = Path(args.input_dir)
        wav_files = sorted(input_dir.glob('*.wav'))
        print(f"Found {len(wav_files)} wav files to convert\n")

        for wav_path in wav_files:
            out_path = os.path.join(args.output_dir, wav_path.name)
            convert_file(knn_vc, matching_set, str(wav_path), out_path, topk=args.topk)

        print(f"\nDone. Converted {len(wav_files)} files → {args.output_dir}")
    else:
        parser.print_help()
        print("\nProvide either (--input + --output) or (--input_dir + --output_dir)")


if __name__ == '__main__':
    main()
