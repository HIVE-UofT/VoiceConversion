"""
Evaluate domain-shift voice conversion quality.

Metrics (comparable across all methods):
  - MCD to target: converted vs real post-surgery (lower = better conversion)
  - Content preservation MCD: source vs converted (lower = better content preservation)
  - F0 Correlation: pitch tracking source vs converted (higher = better)
  - Baseline MCD: real pre vs real post (for reference)

Paired by patient ID: ses1_XXXX (pre) -> ses2_XXXX (post)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --converted_dir /path/to/converted --skip_f0
"""

import argparse
import os
import glob
import numpy as np
import librosa
from pathlib import Path


SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80


def compute_mcd(ref_mel, synth_mel):
    """Mel Cepstral Distortion between two mel-spectrograms (dB scale)."""
    min_len = min(ref_mel.shape[1], synth_mel.shape[1])
    ref_mel = ref_mel[:, :min_len]
    synth_mel = synth_mel[:, :min_len]

    ref_mfcc = librosa.feature.mfcc(S=ref_mel, n_mfcc=13)
    synth_mfcc = librosa.feature.mfcc(S=synth_mel, n_mfcc=13)

    min_len = min(ref_mfcc.shape[1], synth_mfcc.shape[1])
    diff = ref_mfcc[:, :min_len] - synth_mfcc[:, :min_len]
    return np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=0)))


def compute_f0_corr(audio_a, audio_b, sr=SAMPLE_RATE):
    """F0 correlation between two audio signals."""
    f0_a, _, _ = librosa.pyin(audio_a, fmin=50, fmax=500, sr=sr)
    f0_b, _, _ = librosa.pyin(audio_b, fmin=50, fmax=500, sr=sr)

    min_len = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:min_len], f0_b[:min_len]
    valid = ~np.isnan(f0_a) & ~np.isnan(f0_b)

    if valid.sum() < 10:
        return float('nan')
    return np.corrcoef(f0_a[valid], f0_b[valid])[0, 1]


def audio_to_mel_db(audio, sr=SAMPLE_RATE):
    """Convert audio to dB-scale mel-spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max)


def main():
    parser = argparse.ArgumentParser(description="Evaluate voice conversion quality")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1",
                        help='Pre-surgery WAV files (source)')
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2",
                        help='Post-surgery WAV files (ground-truth target)')
    parser.add_argument('--converted_dir', type=str, required=True,
                        help='Converted WAV files')
    parser.add_argument('--method_name', type=str, default='UNet-VC',
                        help='Name for display')
    parser.add_argument('--skip_f0', action='store_true', help='Skip F0 correlation (slow)')
    args = parser.parse_args()

    sr = SAMPLE_RATE
    converted_files = sorted(glob.glob(os.path.join(args.converted_dir, "*.wav")))

    if not converted_files:
        print(f"No converted wav files in {args.converted_dir}")
        return

    print(f"Evaluating {len(converted_files)} converted files...")
    print(f"Source dir: {args.pre_dir}")
    print(f"Target dir: {args.post_dir}\n")

    mcd_to_target = []
    mcd_content_pres = []
    f0_corrs = []
    mcd_pre_vs_post = []

    for cf in converted_files:
        name = Path(cf).stem

        pre_path = os.path.join(args.pre_dir, name + '.wav')
        if not os.path.exists(pre_path):
            print(f"  SKIP {name}: no matching pre-surgery file")
            continue

        post_name = name.replace('ses1', 'ses2')
        post_path = os.path.join(args.post_dir, post_name + '.wav')
        if not os.path.exists(post_path):
            print(f"  SKIP {name}: no matching post-surgery file ({post_name})")
            continue

        y_conv, _ = librosa.load(cf, sr=sr)
        y_pre, _ = librosa.load(pre_path, sr=sr)
        y_post, _ = librosa.load(post_path, sr=sr)

        mel_conv = audio_to_mel_db(y_conv, sr)
        mel_pre = audio_to_mel_db(y_pre, sr)
        mel_post = audio_to_mel_db(y_post, sr)

        mcd_target = compute_mcd(mel_post, mel_conv)
        mcd_to_target.append(mcd_target)

        mcd_content = compute_mcd(mel_pre, mel_conv)
        mcd_content_pres.append(mcd_content)

        mcd_baseline = compute_mcd(mel_pre, mel_post)
        mcd_pre_vs_post.append(mcd_baseline)

        f0_str = ""
        if not args.skip_f0:
            f0_c = compute_f0_corr(y_pre, y_conv, sr)
            f0_corrs.append(f0_c)
            f0_str = f"  F0={f0_c:.3f}" if not np.isnan(f0_c) else "  F0=nan"

        print(f"  {name}: MCD(target)={mcd_target:.2f}  MCD(content)={mcd_content:.2f}  MCD(baseline)={mcd_baseline:.2f}{f0_str}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {args.method_name} — Evaluation")
    print(f"{'='*60}")

    print(f"\n  Baseline (real pre vs real post, no conversion):")
    print(f"    MCD: {np.mean(mcd_pre_vs_post):.2f} +/- {np.std(mcd_pre_vs_post):.2f}")

    print(f"\n  Pre -> Post conversion — {len(mcd_to_target)} paired samples:")
    print(f"    MCD to target (lower=better):      {np.mean(mcd_to_target):.2f} +/- {np.std(mcd_to_target):.2f}")
    print(f"    Content MCD (lower=preserved):     {np.mean(mcd_content_pres):.2f} +/- {np.std(mcd_content_pres):.2f}")

    if not args.skip_f0 and f0_corrs:
        valid_f0 = [f for f in f0_corrs if not np.isnan(f)]
        if valid_f0:
            print(f"    F0 Correlation (higher=better):    {np.mean(valid_f0):.3f} +/- {np.std(valid_f0):.3f}  ({len(valid_f0)}/{len(f0_corrs)} valid)")

    if mcd_to_target and mcd_pre_vs_post:
        baseline_avg = np.mean(mcd_pre_vs_post)
        converted_avg = np.mean(mcd_to_target)
        if baseline_avg > 0:
            reduction = (baseline_avg - converted_avg) / baseline_avg * 100
            print(f"\n  Conversion effectiveness:")
            print(f"    MCD reduction vs baseline: {reduction:+.1f}%")
            if reduction > 0:
                print(f"    (Converted is {reduction:.1f}% closer to post-surgery than original pre-surgery)")
            else:
                print(f"    (Converted is further from post-surgery than original pre-surgery)")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
