"""
Shared evaluation metrics for all voice conversion methods.

Metrics:
  - Speaker Cosine Similarity (ECAPA-TDNN): content-independent voice identity comparison.
    This is the PRIMARY metric — compares voice identity regardless of what is being said.
  - LSD (Log-Spectral Distance): RMS log-power spectral distance (dB).
  - SED (Spectral Envelope Distance): average mel-band energy difference (voice quality).
  - F0 Correlation: pitch contour correlation (prosody preservation).

All metrics are computed on paired files:
  Converted:    Tonsill_ses1_speech_XXXX.wav  (pre -> post converted)
  Pre-surgery:  Tonsill_ses1_speech_XXXX.wav  (source)
  Post-surgery: Tonsill_ses2_speech_XXXX.wav  (ground-truth target)
"""

import argparse
import os
import glob
import numpy as np
import librosa
import torch
import torchaudio
from pathlib import Path


SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80


# ──────────────────────────────────────────────
# Speaker embedding (ECAPA-TDNN via SpeechBrain)
# ──────────────────────────────────────────────

_spk_model = None

def _get_speaker_model():
    """Lazy-load ECAPA-TDNN speaker encoder (cached)."""
    global _spk_model
    if _spk_model is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )
        print("  [Loaded ECAPA-TDNN speaker encoder]")
    return _spk_model


def compute_speaker_embedding(audio, sr=SAMPLE_RATE):
    """Extract ECAPA-TDNN speaker embedding from audio (numpy array)."""
    model = _get_speaker_model()
    # SpeechBrain expects (1, T) torch tensor at 16kHz
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    with torch.no_grad():
        emb = model.encode_batch(waveform)  # (1, 1, 192)
    return emb.squeeze().numpy()  # (192,)


def compute_speaker_similarity(audio_a, audio_b, sr=SAMPLE_RATE):
    """
    Cosine similarity between speaker embeddings (ECAPA-TDNN).
    Content-independent: only compares voice identity.
    Range: -1 to 1 (higher = more similar voice identity).
    Typical same-speaker: >0.7, different speaker: <0.3
    """
    emb_a = compute_speaker_embedding(audio_a, sr)
    emb_b = compute_speaker_embedding(audio_b, sr)
    cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
    return float(cos_sim)


# ──────────────────────────────────────────────
# Spectral metrics
# ──────────────────────────────────────────────

def compute_lsd(audio_a, audio_b, sr=SAMPLE_RATE):
    """
    Log-Spectral Distance (dB) between two audio signals.
    Measures spectral envelope similarity. Typical values: 1-5 dB.
    """
    S_a = np.abs(librosa.stft(audio_a, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    S_b = np.abs(librosa.stft(audio_b, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2

    min_len = min(S_a.shape[1], S_b.shape[1])
    S_a = S_a[:, :min_len] + 1e-10
    S_b = S_b[:, :min_len] + 1e-10

    log_diff = np.log10(S_a) - np.log10(S_b)
    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return lsd


def compute_f0_corr(audio_a, audio_b, sr=SAMPLE_RATE):
    """F0 correlation between two audio signals (prosody preservation)."""
    f0_a, _, _ = librosa.pyin(audio_a, fmin=50, fmax=500, sr=sr)
    f0_b, _, _ = librosa.pyin(audio_b, fmin=50, fmax=500, sr=sr)

    min_len = min(len(f0_a), len(f0_b))
    f0_a, f0_b = f0_a[:min_len], f0_b[:min_len]
    valid = ~np.isnan(f0_a) & ~np.isnan(f0_b)

    if valid.sum() < 10:
        return float('nan')
    return np.corrcoef(f0_a[valid], f0_b[valid])[0, 1]


def compute_spectral_envelope_distance(audio_a, audio_b, sr=SAMPLE_RATE):
    """
    Mean difference in mel-band energies (dB).
    Captures voice quality characteristics (resonance, nasality).
    Averaged over time — measures global spectral shift.
    """
    mel_a = librosa.feature.melspectrogram(
        y=audio_a, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_b = librosa.feature.melspectrogram(
        y=audio_b, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)

    mel_a_db = librosa.power_to_db(mel_a + 1e-10, ref=1.0)
    mel_b_db = librosa.power_to_db(mel_b + 1e-10, ref=1.0)

    env_a = mel_a_db.mean(axis=1)  # (n_mels,)
    env_b = mel_b_db.mean(axis=1)

    return np.mean(np.abs(env_a - env_b))


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def run_evaluation(converted_dir, pre_dir, post_dir, method_name='Voice Conversion',
                   skip_f0=False):
    """Run full evaluation on converted files."""
    sr = SAMPLE_RATE
    converted_files = sorted(glob.glob(os.path.join(converted_dir, "*.wav")))

    if not converted_files:
        print(f"No converted wav files in {converted_dir}")
        return

    print(f"Evaluating {len(converted_files)} converted files...")
    print(f"Source dir: {pre_dir}")
    print(f"Target dir: {post_dir}\n")

    # Metrics storage
    results = {
        # Speaker similarity (cosine, higher = better match)
        'spk_sim_conv_vs_target': [],   # converted vs post-surgery (want HIGH)
        'spk_sim_conv_vs_source': [],   # converted vs pre-surgery (reference)
        'spk_sim_baseline': [],         # pre vs post (same patient baseline)
        # Spectral
        'lsd_target': [], 'lsd_content': [], 'lsd_baseline': [],
        'sed_target': [], 'sed_content': [], 'sed_baseline': [],
        # Prosody
        'f0_corrs': [],
    }

    for cf in converted_files:
        name = Path(cf).stem

        pre_path = os.path.join(pre_dir, name + '.wav')
        if not os.path.exists(pre_path):
            print(f"  SKIP {name}: no matching pre-surgery file")
            continue

        post_name = name.replace('ses1', 'ses2')
        post_path = os.path.join(post_dir, post_name + '.wav')
        if not os.path.exists(post_path):
            print(f"  SKIP {name}: no matching post-surgery file ({post_name})")
            continue

        y_conv, _ = librosa.load(cf, sr=sr)
        y_pre, _ = librosa.load(pre_path, sr=sr)
        y_post, _ = librosa.load(post_path, sr=sr)

        # Speaker similarity (content-independent voice identity)
        spk_ct = compute_speaker_similarity(y_conv, y_post, sr)
        spk_cs = compute_speaker_similarity(y_conv, y_pre, sr)
        spk_bl = compute_speaker_similarity(y_pre, y_post, sr)
        results['spk_sim_conv_vs_target'].append(spk_ct)
        results['spk_sim_conv_vs_source'].append(spk_cs)
        results['spk_sim_baseline'].append(spk_bl)

        # LSD
        lsd_t = compute_lsd(y_post, y_conv, sr)
        lsd_c = compute_lsd(y_pre, y_conv, sr)
        lsd_b = compute_lsd(y_pre, y_post, sr)
        results['lsd_target'].append(lsd_t)
        results['lsd_content'].append(lsd_c)
        results['lsd_baseline'].append(lsd_b)

        # Spectral Envelope Distance
        sed_t = compute_spectral_envelope_distance(y_post, y_conv, sr)
        sed_c = compute_spectral_envelope_distance(y_pre, y_conv, sr)
        sed_b = compute_spectral_envelope_distance(y_pre, y_post, sr)
        results['sed_target'].append(sed_t)
        results['sed_content'].append(sed_c)
        results['sed_baseline'].append(sed_b)

        # F0
        f0_str = ""
        if not skip_f0:
            f0_c = compute_f0_corr(y_pre, y_conv, sr)
            results['f0_corrs'].append(f0_c)
            f0_str = f"  F0={f0_c:.3f}" if not np.isnan(f0_c) else "  F0=nan"

        print(f"  {name}: SpkSim(t)={spk_ct:.3f} SpkSim(s)={spk_cs:.3f} LSD(t)={lsd_t:.2f} SED(t)={sed_t:.1f}{f0_str}")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print(f"  {method_name} — Evaluation Summary")
    print(f"{'='*70}")

    n = len(results['spk_sim_conv_vs_target'])
    if n == 0:
        print("  No paired samples found.")
        return

    def fmt(vals):
        return f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"

    def fmt2(vals):
        return f"{np.mean(vals):.2f} +/- {np.std(vals):.2f}"

    print(f"\n  Baseline (real pre vs real post — same patient, no conversion):")
    print(f"    Speaker Similarity:  {fmt(results['spk_sim_baseline'])}")
    print(f"    LSD:                 {fmt2(results['lsd_baseline'])} dB")
    print(f"    SED:                 {fmt2(results['sed_baseline'])} dB")

    print(f"\n  Voice Identity Match — {n} paired samples:")
    print(f"    Converted vs Target post (higher=better):  {fmt(results['spk_sim_conv_vs_target'])}")
    print(f"    Converted vs Source pre  (reference):      {fmt(results['spk_sim_conv_vs_source'])}")

    print(f"\n  Spectral Quality:")
    print(f"    LSD to target (lower=better):       {fmt2(results['lsd_target'])} dB")
    print(f"    SED to target (lower=better):       {fmt2(results['sed_target'])} dB")

    print(f"\n  Content Preservation (source vs converted):")
    print(f"    LSD (lower=preserved):              {fmt2(results['lsd_content'])} dB")
    print(f"    SED (lower=preserved):              {fmt2(results['sed_content'])} dB")

    if not skip_f0 and results['f0_corrs']:
        valid_f0 = [f for f in results['f0_corrs'] if not np.isnan(f)]
        if valid_f0:
            print(f"    F0 Correlation (higher=better):     {np.mean(valid_f0):.3f} +/- {np.std(valid_f0):.3f}  ({len(valid_f0)}/{len(results['f0_corrs'])} valid)")

    # Conversion effectiveness
    baseline_spk = np.mean(results['spk_sim_baseline'])
    converted_spk = np.mean(results['spk_sim_conv_vs_target'])
    source_spk = np.mean(results['spk_sim_conv_vs_source'])

    print(f"\n  Conversion Effectiveness:")
    print(f"    Baseline (pre vs post same patient):        {baseline_spk:.3f}")
    print(f"    Converted vs target post-surgery:           {converted_spk:.3f}")
    print(f"    Converted vs source pre-surgery:            {source_spk:.3f}")

    if converted_spk > source_spk:
        print(f"    -> Converted voice is CLOSER to post-surgery than to pre-surgery")
    else:
        print(f"    -> Converted voice is still closer to pre-surgery (minimal conversion)")

    baseline_sed = np.mean(results['sed_baseline'])
    converted_sed = np.mean(results['sed_target'])
    if baseline_sed > 0:
        reduction_sed = (baseline_sed - converted_sed) / baseline_sed * 100
        print(f"    SED reduction vs baseline: {reduction_sed:+.1f}%")

    print(f"{'='*70}")


def make_parser(method_name='Voice Conversion', default_converted_dir=None):
    """Create standard argument parser for evaluation scripts."""
    parser = argparse.ArgumentParser(description=f"Evaluate {method_name} conversion quality")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--converted_dir', type=str,
                        default=default_converted_dir,
                        required=(default_converted_dir is None))
    parser.add_argument('--method_name', type=str, default=method_name)
    parser.add_argument('--skip_f0', action='store_true', help='Skip F0 correlation (slow)')
    return parser
