"""
Inference script for VQVAE Voice Conversion.

Converts audio files from one domain to another by:
1. Extracting content from source audio (via VQ encoder)
2. Extracting voice quality from target domain (averaged from reference samples)
3. Decoding with swapped quality vector

Usage:
  # Single file (uses averaged post-surgery quality)
  python scripts/inference.py --input /path/to/pre_surgery.wav --output /path/to/output.wav

  # With specific target reference
  python scripts/inference.py --input source.wav --output out.wav --target_ref /path/to/post_surgery.wav

  # Batch directory
  python scripts/inference.py --input_dir /path/to/wavs/ --output_dir /path/to/converted/

  # Reverse direction (post → pre)
  python scripts/inference.py --input file.wav --output out.wav --direction B2A
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import pickle

from model.vqvae import VQVAE


# Audio parameters (must match dataset_processing.py)
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
SEGMENT_DURATION = 5  # seconds
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_DURATION
TARGET_LEN = 400


def audio_to_mel(audio, sr=SAMPLE_RATE):
    """Convert audio to normalized mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return mel_norm  # (80, T)


def mel_to_audio(mel_norm, sr=SAMPLE_RATE):
    """Convert normalized mel-spectrogram back to audio via Griffin-Lim."""
    mel_db = mel_norm * 80.0 - 80.0
    mel_power = librosa.db_to_power(mel_db)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=64
    )
    return audio


def compute_average_quality(model, pkl_path, label, device, target_len=TARGET_LEN, max_samples=50):
    """
    Compute average quality vector for a domain from the dataset.
    This captures the 'average' voice quality of pre or post surgery.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    domain_data = [d for d in data if d['label'] == label]
    if len(domain_data) > max_samples:
        indices = np.random.RandomState(42).choice(len(domain_data), max_samples, replace=False)
        domain_data = [domain_data[i] for i in indices]

    qualities = []
    model.eval()
    with torch.no_grad():
        for d in domain_data:
            mel = d['mel_spectrogram'].copy()
            if mel.shape[1] > target_len:
                mel = mel[:, :target_len]
            elif mel.shape[1] < target_len:
                pad = target_len - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')

            mel_t = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
            q = model.quality_encoder(mel_t)  # (1, quality_dim)
            qualities.append(q)

    avg_quality = torch.stack(qualities).mean(dim=0)  # (1, quality_dim)
    return avg_quality


def convert_audio(model, audio, target_quality, device):
    """Convert a full audio signal using the VQVAE model."""
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Process in segments
    segments = []
    for start in range(0, len(audio), SEGMENT_SAMPLES):
        segment = audio[start:start + SEGMENT_SAMPLES]
        if len(segment) < SAMPLE_RATE:  # skip < 1 second
            continue
        if len(segment) < SEGMENT_SAMPLES:
            segment = np.pad(segment, (0, SEGMENT_SAMPLES - len(segment)))

        mel = audio_to_mel(segment)  # (80, T)

        # Pad/crop to TARGET_LEN
        if mel.shape[1] > TARGET_LEN:
            mel = mel[:, :TARGET_LEN]
        elif mel.shape[1] < TARGET_LEN:
            pad = TARGET_LEN - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode='constant')

        mel_t = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)

        # Convert: source content + target quality
        with torch.no_grad():
            content_z = model.content_encoder(mel_t)
            content_q, _, _ = model.vq(content_z)
            converted = model.decoder(content_q, target_quality)
            converted = model._match_size(converted, mel_t)

        converted_mel = converted[0, 0].cpu().numpy()
        converted_audio = mel_to_audio(converted_mel)
        segments.append(converted_audio[:SEGMENT_SAMPLES])

    if not segments:
        return np.zeros(SEGMENT_SAMPLES)

    return np.concatenate(segments)


def main():
    parser = argparse.ArgumentParser(description="VQVAE Voice Conversion Inference")
    parser.add_argument('--input', type=str, help="Input audio file")
    parser.add_argument('--output', type=str, help="Output audio file")
    parser.add_argument('--input_dir', type=str, help="Input directory for batch conversion")
    parser.add_argument('--output_dir', type=str, help="Output directory for batch conversion")
    parser.add_argument('--target_ref', type=str, default=None,
                        help="Target reference audio (if not provided, uses averaged quality from dataset)")
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help="A2B=pre->post, B2A=post->pre")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_vqvae.pth'))
    parser.add_argument('--data_pkl', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/train_dataset.pkl",
                        help="Dataset pickle for computing average quality")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = VQVAE(code_dim=64, num_codes=16, num_heads=4, quality_dim=32).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    # Get target quality vector
    if args.target_ref:
        # From a specific reference file
        ref_audio, _ = librosa.load(args.target_ref, sr=SAMPLE_RATE)
        ref_audio, _ = librosa.effects.trim(ref_audio, top_db=30)
        ref_mel = audio_to_mel(ref_audio[:SEGMENT_SAMPLES])
        if ref_mel.shape[1] > TARGET_LEN:
            ref_mel = ref_mel[:, :TARGET_LEN]
        elif ref_mel.shape[1] < TARGET_LEN:
            pad = TARGET_LEN - ref_mel.shape[1]
            ref_mel = np.pad(ref_mel, ((0, 0), (0, pad)), mode='constant')
        ref_t = torch.from_numpy(ref_mel).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            target_quality = model.quality_encoder(ref_t)
    else:
        # Average quality from dataset
        target_label = 1 if args.direction == 'A2B' else 0
        target_quality = compute_average_quality(model, args.data_pkl, target_label, device)
        direction_str = "post-surgery" if args.direction == 'A2B' else "pre-surgery"
        print(f"Using averaged {direction_str} quality vector")

    # Convert
    if args.input and args.output:
        audio, _ = librosa.load(args.input, sr=SAMPLE_RATE)
        converted = convert_audio(model, audio, target_quality, device)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        sf.write(args.output, converted, SAMPLE_RATE)
        print(f"Saved: {args.output}")

    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        wav_files = [f for f in os.listdir(args.input_dir) if f.endswith('.wav')]
        print(f"Converting {len(wav_files)} files...")
        for fname in wav_files:
            audio, _ = librosa.load(os.path.join(args.input_dir, fname), sr=SAMPLE_RATE)
            converted = convert_audio(model, audio, target_quality, device)
            sf.write(os.path.join(args.output_dir, fname), converted, SAMPLE_RATE)
            print(f"  {fname}")
        print("Done.")
    else:
        parser.error("Provide --input/--output or --input_dir/--output_dir")


if __name__ == '__main__':
    main()