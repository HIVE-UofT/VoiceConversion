"""
UNet-VC — Inference on held-out test patients only.

Reads test_files.json from the checkpoint directory to know which
patients were held out, and converts only those.

Usage:
    python scripts/inference_kfold.py
    python scripts/inference_kfold.py --checkpoint checkpoints_kfold/best_model.pt
"""

import argparse
import os
import sys
import json
import glob
import torch
import torchaudio
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D


SAMPLE_RATE = 16000


def convert_file(knn_vc, model, input_path, output_path, device):
    features = knn_vc.get_features(input_path)
    with torch.no_grad():
        x = features.t().unsqueeze(0).to(device)
        y = model(x)
        converted_features = y.squeeze(0).t()
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()
    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    duration = out_wav.shape[0] / SAMPLE_RATE
    print(f"  {Path(input_path).name} -> {Path(output_path).name} ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="UNet-VC — Inference on test patients")
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints_kfold', 'best_model.pt'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'converted_kfold'))
    parser.add_argument('--all_patients', action='store_true',
                        help='Convert all patients (not just test). Useful for comparison.')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Override input directory (converts all files in it)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load kNN-VC
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    model = ResUNet1D(
        feat_dim=config['feat_dim'],
        hidden_dim=config['hidden_dim'],
        n_levels=config['n_levels'],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (val={ckpt['val_loss']:.6f}, alpha={ckpt['alpha']:.4f})")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_dir:
        # Convert all files in the given directory
        wav_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        print(f"\nConverting all {len(wav_files)} files from {args.input_dir}...")
        for wf in wav_files:
            out_path = os.path.join(args.output_dir, Path(wf).name)
            convert_file(knn_vc, model, wf, out_path, device)
    else:
        # Read test patient list
        ckpt_dir = os.path.dirname(args.checkpoint)
        test_info_path = os.path.join(ckpt_dir, 'test_files.json')

        if not os.path.exists(test_info_path):
            print(f"ERROR: {test_info_path} not found. Run train_kfold.py first.")
            sys.exit(1)

        with open(test_info_path) as f:
            test_info = json.load(f)

        test_files = test_info['test_wav_files']
        test_patients = test_info['test_patients']

        if args.all_patients:
            # Also convert non-test patients (for comparison, clearly marked)
            split_path = os.path.join(ckpt_dir, 'split_info.json')
            with open(split_path) as f:
                split_info = json.load(f)

            pre_dir = os.path.dirname(test_files[0])
            all_wavs = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
            print(f"\nConverting ALL {len(all_wavs)} patients (test + train)...")
            for wf in all_wavs:
                name = Path(wf).stem
                is_test = name in test_patients
                out_path = os.path.join(args.output_dir, Path(wf).name)
                tag = "[TEST]" if is_test else "[TRAIN]"
                print(f"  {tag}", end="")
                convert_file(knn_vc, model, wf, out_path, device)
        else:
            print(f"\nConverting {len(test_files)} held-out test patients...")
            for wf in test_files:
                if not os.path.exists(wf):
                    print(f"  WARNING: {wf} not found, skipping")
                    continue
                out_path = os.path.join(args.output_dir, Path(wf).name)
                convert_file(knn_vc, model, wf, out_path, device)

    print("Done.")


if __name__ == '__main__':
    main()
