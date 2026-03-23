"""
VQVAE Experiment 5 — Inference

Converts pre-surgery audio to post-surgery using:
1. WavLM feature extraction (frozen)
2. Content encoding + VQ (strips quality info)
3. Injection of average post-surgery quality vector
4. Decoding back to WavLM features
5. HiFi-GAN vocoding to audio

Usage:
    python scripts/inference_exp5.py
    python scripts/inference_exp5.py --input_dir /path/to/wavs --output_dir /path/to/converted
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import torch
import torchaudio
from pathlib import Path

from model.vqvae_wavlm import VQVAEWavLM


SAMPLE_RATE = 16000


def convert_file(knn_vc, model, avg_quality, input_path, output_path, device):
    """Convert a single file."""
    # Extract WavLM features
    features = knn_vc.get_features(input_path)  # (T, 1024)

    # Run through VQVAE: content from source + target quality
    with torch.no_grad():
        x = features.t().unsqueeze(0).to(device)  # (1, 1024, T)
        quality = avg_quality.unsqueeze(0).to(device)  # (1, quality_dim)
        converted = model.convert(x, quality)  # (1, 1024, T)
        converted_features = converted.squeeze(0).t()  # (T, 1024)

    # Vocode
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()

    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="VQVAE Exp5 — Inference")
    parser.add_argument('--input', type=str, help='Input WAV file')
    parser.add_argument('--output', type=str, help='Output WAV file')
    parser.add_argument('--input_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'converted_exp5'))
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints_exp5', 'best_vqvae_wavlm.pth'))
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='A2B=pre->post, B2A=post->pre')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC (WavLM + HiFi-GAN)
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load trained VQVAE
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ckpt['config']

    model = VQVAEWavLM(
        feat_dim=config['feat_dim'], code_dim=config['code_dim'],
        num_codes=config['num_codes'], num_heads=config['num_heads'],
        quality_dim=config['quality_dim'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Select quality vector based on direction
    if args.direction == 'A2B':
        avg_quality = ckpt['avg_quality_post']
        print("Direction: pre -> post (using avg post-surgery quality)")
    else:
        avg_quality = ckpt['avg_quality_pre']
        print("Direction: post -> pre (using avg pre-surgery quality)")

    print(f"Loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(knn_vc, model, avg_quality, args.input, args.output, device)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, model, avg_quality, wf, out_path, device)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
