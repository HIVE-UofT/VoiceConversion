"""
UNet-VC — Nonlinear Feature-Space Voice Conversion (Inference)

Applies the trained residual U-Net to transform WavLM features,
then vocodes back to audio using HiFi-GAN.

Usage:
    python scripts/inference.py --input source.wav --output converted.wav
    python scripts/inference.py --input_dir /path/to/wavs/ --output_dir /path/to/converted/
"""

import argparse
import os
import sys
import glob
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D


SAMPLE_RATE = 16000


def convert_file(knn_vc, model, input_path, output_path, device):
    """Convert a single file using the trained U-Net transform."""
    features = knn_vc.get_features(input_path)  # (T, 1024)

    # Run through U-Net: (1, 1024, T)
    with torch.no_grad():
        x = features.t().unsqueeze(0).to(device)  # (1, 1024, T)
        y = model(x)                                # (1, 1024, T)
        converted_features = y.squeeze(0).t()       # (T, 1024)

    # Vocode
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()

    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="UNet-VC — Inference")
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_model.pt'))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC (WavLM + HiFi-GAN)
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load trained U-Net
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    model = ResUNet1D(
        feat_dim=config['feat_dim'],
        hidden_dim=config['hidden_dim'],
        n_levels=config['n_levels'],
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.6f}, alpha={ckpt['alpha']:.4f})")

    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, model, args.input, args.output, device)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, model, wf, out_path, device)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
