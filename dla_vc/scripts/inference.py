"""
DLA-VC: Dual Layer Adapter VC — Inference

Converts pre-surgery audio to post-surgery using:
1. WavLM-Large (frozen) → all 24 hidden states
2. Content adapter → U-Net encoder → VQ → quantized content
3. Inject avg post-surgery quality vector → FiLM-modulated U-Net decoder
4. Output: WavLM layer 6 features → knn-vc HiFi-GAN → audio

Usage:
    python scripts/inference.py
    python scripts/inference.py --input_dir /path/to/wavs --output_dir /path/to/out
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import torch
import torchaudio
from pathlib import Path

from model.dla_vc import DLAVCModel


SAMPLE_RATE = 16000


def load_wavlm_extractor(device):
    """Load WavLM-Large for multi-layer feature extraction."""
    from transformers import WavLMModel
    print("Loading WavLM-Large...")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_hidden_states(wavlm_model, audio, device):
    """Extract all hidden states from audio tensor."""
    with torch.no_grad():
        outputs = wavlm_model(audio.to(device), output_hidden_states=True)
    all_layers = torch.stack(outputs.hidden_states[1:], dim=1)  # (B, 24, T, 1024)
    all_layers = all_layers.permute(0, 1, 3, 2)  # (B, 24, 1024, T)
    return all_layers


def convert_file(wavlm_model, model, avg_quality, input_path, output_path, device, vocoder):
    """Convert a single file."""
    audio, sr = torchaudio.load(input_path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    audio = audio[0].unsqueeze(0)  # (1, T)

    hidden_states = extract_hidden_states(wavlm_model, audio, device)
    quality = avg_quality.unsqueeze(0).to(device)

    with torch.no_grad():
        converted = model.convert(hidden_states, quality)  # (1, 1024, T)
        converted_features = converted.squeeze(0).t()  # (T, 1024)

    out_wav = vocoder.vocode(converted_features[None]).cpu().squeeze()
    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="DLA-VC — Inference")
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--input_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'converted'))
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_dla_vc.pth'))
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load WavLM for feature extraction
    wavlm_model = load_wavlm_extractor(device)

    # Load knn-vc for HiFi-GAN vocoder only
    print("Loading kNN-VC vocoder (HiFi-GAN)...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load trained model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ckpt['config']

    model = DLAVCModel(
        feat_dim=config['feat_dim'], code_dim=config['code_dim'],
        num_codes=config['num_codes'], num_heads=config['num_heads'],
        quality_dim=config['quality_dim'],
        num_wavlm_layers=config['num_wavlm_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    if args.direction == 'A2B':
        avg_quality = ckpt['avg_quality_post']
        print("Direction: pre -> post")
    else:
        avg_quality = ckpt['avg_quality_pre']
        print("Direction: post -> pre")

    print(f"Loaded from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")
    print(f"Adapter weights - Content top layers: "
          f"{sorted(range(24), key=lambda i: -ckpt['adapter_weights']['content'][i])[:5]}")
    print(f"Adapter weights - Quality top layers: "
          f"{sorted(range(24), key=lambda i: -ckpt['adapter_weights']['quality'][i])[:5]}")

    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        convert_file(wavlm_model, model, avg_quality, args.input, args.output, device, knn_vc)
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"Converting {len(wav_files)} files...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(wavlm_model, model, avg_quality, wf, out_path, device, knn_vc)
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()
