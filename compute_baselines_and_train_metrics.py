"""
Compute baseline SpkSim (no conversion) and UNet-VC train+test metrics.

For each surgery type:
1. Read split_info.json to get train/test patient lists
2. Compute baseline SpkSim (pre vs post, no conversion) for train and test separately
3. Convert ALL patients (train + test) with the final UNet model
4. Evaluate SpkSim for train and test patients separately

This gives:
- Baseline (no conversion) on test set
- Baseline (no conversion) on train set
- UNet-VC on test set (honest generalization)
- UNet-VC on train set (for comparison / overfitting check)

Usage:
    python compute_baselines_and_train_metrics.py
"""

import argparse
import os
import sys
import json
import glob
import torch
import torchaudio
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

SAMPLE_RATE = 16000
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"


def load_ecapa(device):
    """Load ECAPA-TDNN speaker encoder."""
    from speechbrain.inference.speaker import EncoderClassifier
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    print("  [Loaded ECAPA-TDNN speaker encoder]")
    return encoder


def get_speaker_embedding(encoder, wav_path):
    """Extract speaker embedding from a wav file."""
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    embedding = encoder.encode_batch(signal)
    return embedding.squeeze()


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def compute_spksim_for_patients(encoder, pre_files, post_files, indices, label=""):
    """Compute SpkSim (pre vs post) for a set of patients by index."""
    scores = []
    for i in indices:
        pre_path = pre_files[i]
        post_path = post_files[i]
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            print(f"  WARNING: Missing file for patient {i}, skipping")
            continue
        emb_pre = get_speaker_embedding(encoder, pre_path)
        emb_post = get_speaker_embedding(encoder, post_path)
        sim = cosine_sim(emb_pre, emb_post)
        name = Path(pre_path).stem
        scores.append(sim)
        print(f"  {label} {name}: {sim:.3f}")

    if scores:
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {label} SpkSim: {mean:.3f} +/- {std:.3f} (n={len(scores)})")
    return scores


def compute_converted_spksim(encoder, converted_dir, post_files, indices, pre_files, label=""):
    """Compute SpkSim (converted vs post) for a set of patients."""
    scores_vs_post = []
    scores_vs_pre = []
    for i in indices:
        pre_name = Path(pre_files[i]).name
        conv_path = os.path.join(converted_dir, pre_name)
        post_path = post_files[i]

        if not os.path.exists(conv_path):
            print(f"  WARNING: {conv_path} not found, skipping")
            continue

        emb_conv = get_speaker_embedding(encoder, conv_path)
        emb_post = get_speaker_embedding(encoder, post_path)
        emb_pre = get_speaker_embedding(encoder, pre_files[i])

        sim_post = cosine_sim(emb_conv, emb_post)
        sim_pre = cosine_sim(emb_conv, emb_pre)
        scores_vs_post.append(sim_post)
        scores_vs_pre.append(sim_pre)
        name = Path(pre_files[i]).stem
        print(f"  {label} {name}: vs_post={sim_post:.3f}  vs_pre={sim_pre:.3f}")

    if scores_vs_post:
        m1, s1 = np.mean(scores_vs_post), np.std(scores_vs_post)
        m2, s2 = np.mean(scores_vs_pre), np.std(scores_vs_pre)
        print(f"  {label} Conv vs Post: {m1:.3f} +/- {s1:.3f}")
        print(f"  {label} Conv vs Pre:  {m2:.3f} +/- {s2:.3f}")
    return scores_vs_post, scores_vs_pre


def run_unet_inference_all(knn_vc, model, pre_files, output_dir, device):
    """Convert ALL patients with UNet model."""
    os.makedirs(output_dir, exist_ok=True)
    for wf in pre_files:
        features = knn_vc.get_features(wf)
        with torch.no_grad():
            x = features.t().unsqueeze(0).to(device)
            y = model(x)
            converted = y.squeeze(0).t()
        out_wav = knn_vc.vocode(converted[None]).cpu().squeeze()
        out_path = os.path.join(output_dir, Path(wf).name)
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)
        print(f"  {Path(wf).name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, nargs='+', default=['Tonsill', 'Fess', 'Sept'])
    parser.add_argument('--unet_ckpt_pattern', type=str,
                        default='unet_vc/checkpoints_kfold_{surgery}/best_model.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_test', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ECAPA-TDNN
    encoder = load_ecapa(str(device))

    # Try to load kNN-VC and UNet for conversion
    print("Loading kNN-VC...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    for surgery in args.surgery:
        pre_dir = os.path.join(CUCO_BASE, surgery, "Speech", "1")
        post_dir = os.path.join(CUCO_BASE, surgery, "Speech", "2")

        pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
        post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
        n = len(pre_files)

        # Reproduce the same split
        import random
        random.seed(args.seed)
        indices = list(range(n))
        random.shuffle(indices)
        test_idx = sorted(indices[:args.n_test])
        train_idx = sorted(indices[args.n_test:])

        test_names = [Path(pre_files[i]).stem for i in test_idx]
        train_names = [Path(pre_files[i]).stem for i in train_idx]

        print(f"\n{'='*70}")
        print(f"  {surgery} — {n} patients (train={len(train_idx)}, test={len(test_idx)})")
        print(f"  Test: {test_names}")
        print(f"{'='*70}")

        # ═══ Baseline SpkSim (no conversion) ═══
        print(f"\n--- Baseline (no conversion) — TEST ---")
        baseline_test = compute_spksim_for_patients(
            encoder, pre_files, post_files, test_idx, label="[TEST]")

        print(f"\n--- Baseline (no conversion) — TRAIN ---")
        baseline_train = compute_spksim_for_patients(
            encoder, pre_files, post_files, train_idx, label="[TRAIN]")

        # ═══ UNet-VC: convert all patients, evaluate train and test separately ═══
        ckpt_path = args.unet_ckpt_pattern.replace('{surgery}', surgery.lower())
        if os.path.exists(ckpt_path):
            print(f"\n--- UNet-VC: loading {ckpt_path} ---")
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unet_vc'))
            from model.unet import ResUNet1D

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            config = ckpt['config']
            model = ResUNet1D(
                feat_dim=config['feat_dim'],
                hidden_dim=config['hidden_dim'],
                n_levels=config['n_levels'],
                dropout=0.0
            ).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            print(f"  Loaded epoch {ckpt['epoch']}, alpha={ckpt['alpha']:.4f}")

            conv_dir = f"results_all/{surgery}/unet_vc_all"
            print(f"\n--- UNet-VC: converting ALL {n} patients ---")
            run_unet_inference_all(knn_vc, model, pre_files, conv_dir, device)

            print(f"\n--- UNet-VC SpkSim — TEST ---")
            unet_test_post, unet_test_pre = compute_converted_spksim(
                encoder, conv_dir, post_files, test_idx, pre_files, label="[TEST]")

            print(f"\n--- UNet-VC SpkSim — TRAIN ---")
            unet_train_post, unet_train_pre = compute_converted_spksim(
                encoder, conv_dir, post_files, train_idx, pre_files, label="[TRAIN]")

            # ═══ Summary ═══
            print(f"\n{'='*70}")
            print(f"  {surgery} — SUMMARY")
            print(f"{'='*70}")
            print(f"  Baseline (no conversion):")
            print(f"    Test:  {np.mean(baseline_test):.3f} +/- {np.std(baseline_test):.3f} (n={len(baseline_test)})")
            print(f"    Train: {np.mean(baseline_train):.3f} +/- {np.std(baseline_train):.3f} (n={len(baseline_train)})")
            print(f"  UNet-VC (conv vs post):")
            print(f"    Test:  {np.mean(unet_test_post):.3f} +/- {np.std(unet_test_post):.3f} (n={len(unet_test_post)})")
            print(f"    Train: {np.mean(unet_train_post):.3f} +/- {np.std(unet_train_post):.3f} (n={len(unet_train_post)})")
            print(f"  UNet-VC (conv vs source pre):")
            print(f"    Test:  {np.mean(unet_test_pre):.3f} +/- {np.std(unet_test_pre):.3f}")
            print(f"    Train: {np.mean(unet_train_pre):.3f} +/- {np.std(unet_train_pre):.3f}")
            print(f"{'='*70}")
        else:
            print(f"\n  WARNING: UNet checkpoint not found at {ckpt_path}")
            print(f"  Run unet_vc/submit_kfold.sh first, then re-run this script.")


if __name__ == '__main__':
    main()
