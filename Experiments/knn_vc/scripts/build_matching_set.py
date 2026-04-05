"""
Build the kNN-VC matching set from post-surgery recordings.

This is the OFFLINE step — run once. It extracts WavLM features from all
post-surgery audio files and saves them as a single tensor (the "matching set").

Usage:
  python scripts/build_matching_set.py
  python scripts/build_matching_set.py --data_dir /path/to/post_surgery_wavs
  python scripts/build_matching_set.py --use_pkl  # use processed pkl instead of raw wav
"""

import argparse
import os
import sys
import glob
import torch
import pickle
import numpy as np
import soundfile as sf
import tempfile


def build_from_wavs(knn_vc, wav_paths, device):
    """Build matching set directly from wav files."""
    print(f"Building matching set from {len(wav_paths)} wav files...")
    matching_set = knn_vc.get_matching_set(wav_paths)
    return matching_set


def build_from_pkl(knn_vc, pkl_path, label, device):
    """Build matching set from processed pkl dataset.

    Converts mel-spectrograms back to temporary wav files, then extracts
    WavLM features. This is a workaround since kNN-VC needs raw audio.
    """
    import librosa

    with open(pkl_path, 'rb') as f:
        all_data = pickle.load(f)
    target_data = [d for d in all_data if d['label'] == label]
    print(f"Found {len(target_data)} segments with label={label} in {pkl_path}")

    # kNN-VC needs raw wav files, so we write temp wavs from mel-spectrograms
    # NOTE: This is lossy (Griffin-Lim reconstruction). Using raw wavs is preferred.
    tmp_dir = tempfile.mkdtemp()
    tmp_paths = []

    print("Reconstructing audio from mel-spectrograms (Griffin-Lim)...")
    for i, item in enumerate(target_data):
        mel_norm = item['mel_spectrogram']  # (80, T), normalized [0,1]
        mel_db = mel_norm * 80 - 80
        mel_power = librosa.db_to_power(mel_db)
        y = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=16000, n_fft=2048, hop_length=512, n_iter=32
        )
        tmp_path = os.path.join(tmp_dir, f"seg_{i:04d}.wav")
        sf.write(tmp_path, y, 16000)
        tmp_paths.append(tmp_path)

    matching_set = knn_vc.get_matching_set(tmp_paths)

    # Cleanup temp files
    for p in tmp_paths:
        os.remove(p)
    os.rmdir(tmp_dir)

    return matching_set


def main():
    parser = argparse.ArgumentParser(description="Build kNN-VC matching set from post-surgery audio")
    parser.add_argument('--data_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2",
                        help='Directory containing post-surgery wav files')
    parser.add_argument('--use_pkl', action='store_true',
                        help='Use processed pkl files instead of raw wavs')
    parser.add_argument('--pkl_path', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/processed_data/train_dataset.pkl",
                        help='Path to pkl dataset (used with --use_pkl)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'matching_sets', 'post_surgery_matching_set.pt'),
                        help='Output path for matching set tensor')
    parser.add_argument('--prematched', action='store_true', default=True,
                        help='Use prematched HiFi-GAN variant (recommended)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC
    print("Loading kNN-VC (WavLM-Large + HiFi-GAN)...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=args.prematched, device=device)

    if args.use_pkl:
        matching_set = build_from_pkl(knn_vc, args.pkl_path, label=1, device=device)
    else:
        # Find all wav files in post-surgery directory
        wav_paths = sorted(glob.glob(os.path.join(args.data_dir, "**/*.wav"), recursive=True))
        if not wav_paths:
            print(f"No wav files found in {args.data_dir}")
            sys.exit(1)
        matching_set = build_from_wavs(knn_vc, wav_paths, device)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(matching_set.cpu().half(), args.output)

    n_frames = matching_set.shape[0]
    duration_min = n_frames * 0.02 / 60  # 20ms per frame
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nMatching set saved: {args.output}")
    print(f"  Frames: {n_frames:,} ({duration_min:.1f} min of speech)")
    print(f"  Shape: {matching_set.shape}")
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
