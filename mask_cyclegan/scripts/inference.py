"""
Inference script for MaskCycleGAN-VC.

Takes a pre-surgery audio file and converts it to post-surgery voice.
No reference audio needed — the mapping is baked into the trained generator.

Usage:
  python scripts/inference.py --input /path/to/pre_surgery.wav --output /path/to/output.wav
  python scripts/inference.py --input_dir /path/to/wavs/ --output_dir /path/to/converted/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from model.mask_cyclegan import Generator


# ──────────────────────────────────────────────
# Audio processing (matches dataset_processing.py)
# ──────────────────────────────────────────────

SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
SEGMENT_DURATION = 5
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION
TARGET_LEN = 400


def audio_to_mel(y, sr=SAMPLE_RATE):
    """Convert audio waveform to normalized mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80) / 80
    mel_norm = np.clip(mel_norm, 0, 1)
    return mel_norm.astype(np.float32)


def mel_to_audio(mel_norm, sr=SAMPLE_RATE):
    """Convert normalized mel-spectrogram back to audio using Griffin-Lim."""
    # Undo normalization
    mel_db = mel_norm * 80 - 80
    mel_power = librosa.db_to_power(mel_db)
    # Griffin-Lim reconstruction
    y = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=64
    )
    return y


def convert_file(model, input_path, output_path, device):
    """Convert a single audio file from pre-surgery to post-surgery voice."""
    # Load audio
    y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=30)

    # Process in segments (same as training)
    converted_segments = []
    step = SAMPLES_PER_SEGMENT

    for start in range(0, len(y), step):
        segment = y[start:start + step]

        # Skip very short trailing segments
        if len(segment) < SAMPLE_RATE * 2:
            continue

        # Pad if needed
        if len(segment) < step:
            segment = np.pad(segment, (0, step - len(segment)), mode='constant')
            was_padded = True
            original_len = len(y[start:start + step])
        else:
            was_padded = False

        # To mel
        mel = audio_to_mel(segment)  # (80, T)

        # Pad mel to TARGET_LEN
        if mel.shape[1] < TARGET_LEN:
            mel = np.pad(mel, ((0, 0), (0, TARGET_LEN - mel.shape[1])), mode='constant')
        elif mel.shape[1] > TARGET_LEN:
            mel = mel[:, :TARGET_LEN]

        # Convert
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 80, T)
        mask = torch.ones_like(mel_tensor)  # No masking at inference

        with torch.no_grad():
            converted_mel = model(mel_tensor, mask)

        converted_mel = converted_mel.squeeze().cpu().numpy()  # (80, T)

        # Back to audio
        audio_out = mel_to_audio(converted_mel)

        # Trim padding if segment was padded
        if was_padded:
            # Approximate number of samples for original length
            orig_samples = int(original_len / step * len(audio_out))
            audio_out = audio_out[:orig_samples]

        converted_segments.append(audio_out)

    # Concatenate all segments
    if converted_segments:
        full_audio = np.concatenate(converted_segments)
        sf.write(output_path, full_audio, SAMPLE_RATE)
        print(f"Saved: {output_path} ({len(full_audio)/SAMPLE_RATE:.1f}s)")
    else:
        print(f"Warning: No segments to convert for {input_path}")


def main():
    parser = argparse.ArgumentParser(description="MaskCycleGAN-VC Inference: Pre→Post Surgery")
    parser.add_argument('--input', type=str, help='Path to input wav file')
    parser.add_argument('--output', type=str, help='Path to output wav file')
    parser.add_argument('--input_dir', type=str, help='Directory of input wav files')
    parser.add_argument('--output_dir', type=str, help='Directory for output wav files')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'best_mask_cyclegan.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--direction', type=str, default='A2B', choices=['A2B', 'B2A'],
                        help='A2B = pre→post, B2A = post→pre')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (base_channels=64 matches retrained checkpoints)
    generator = Generator(base_channels=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    if args.direction == 'A2B':
        generator.load_state_dict(checkpoint['G_A2B'])
        print("Loaded G_A2B (pre→post surgery)")
    else:
        generator.load_state_dict(checkpoint['G_B2A'])
        print("Loaded G_B2A (post→pre surgery)")
    generator.eval()

    # Single file
    if args.input and args.output:
        convert_file(generator, args.input, args.output, device)

    # Directory
    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        input_dir = Path(args.input_dir)
        wav_files = list(input_dir.glob('*.wav'))
        print(f"Found {len(wav_files)} wav files")

        for wav_path in wav_files:
            out_path = os.path.join(args.output_dir, wav_path.name)
            convert_file(generator, str(wav_path), out_path, device)
    else:
        parser.print_help()
        print("\nProvide either (--input + --output) or (--input_dir + --output_dir)")


if __name__ == '__main__':
    main()
