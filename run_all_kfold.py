"""
Run all methods with proper held-out test evaluation.

For each surgery type (Tonsill, Fess, Sept):
1. Hold out 5 test patients (same seed for all methods)
2. For methods that need "training" (computing stats/transforms), use only non-test patients
3. Convert only the held-out test patients
4. Evaluate on test patients

Methods: kNN-VC, Mean-Shift, MKL-VC, LinearVC, UNet-VC (uses its own kfold script)

Usage:
    python run_all_kfold.py
    python run_all_kfold.py --surgery Tonsill
"""

import argparse
import os
import sys
import glob
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from pathlib import Path
from scipy.linalg import sqrtm

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from shared_evaluate import run_evaluation


SAMPLE_RATE = 16000
SEED = 42
N_TEST = 5
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"


def get_patient_files(pre_dir, post_dir):
    """Get sorted lists of pre/post wav files. Returns (pre_files, post_files, patient_names)."""
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files), \
        f"Mismatch: {len(pre_files)} pre vs {len(post_files)} post"
    names = [Path(f).stem for f in pre_files]
    return pre_files, post_files, names


def split_patients(n_patients, n_test, seed):
    """Return (test_indices, train_indices) with fixed seed."""
    random.seed(seed)
    indices = list(range(n_patients))
    random.shuffle(indices)
    test_idx = sorted(indices[:n_test])
    train_idx = sorted(indices[n_test:])
    return test_idx, train_idx


def extract_features(knn_vc, wav_files):
    """Extract WavLM features for a list of wav files. Returns list of (T, 1024) tensors."""
    feats = []
    for wf in wav_files:
        f = knn_vc.get_features(wf).cpu()
        feats.append(f)
    return feats


def convert_and_save(knn_vc, converted_features, output_path):
    """Vocode and save converted features."""
    out_wav = knn_vc.vocode(converted_features[None]).cpu().squeeze()
    torchaudio.save(output_path, out_wav.unsqueeze(0), SAMPLE_RATE)
    return out_wav.shape[0] / SAMPLE_RATE


# ═══════════════════════════════════════════
# kNN-VC: build matching set from train post, convert test pre
# ═══════════════════════════════════════════
def run_knn_vc(knn_vc, pre_files, post_files, test_idx, train_idx, output_dir, topk=4):
    print("\n" + "="*60)
    print("  kNN-VC")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Build matching set from TRAIN post-surgery files only
    train_post_files = [post_files[i] for i in train_idx]
    print(f"  Building matching set from {len(train_post_files)} train post-surgery files...")
    matching_set = knn_vc.get_matching_set(train_post_files)
    print(f"  Matching set: {matching_set.shape[0]} frames")

    # Convert test pre-surgery files
    for i in test_idx:
        query = knn_vc.get_features(pre_files[i])
        out_wav = knn_vc.match(query, matching_set, topk=topk)
        out_path = os.path.join(output_dir, Path(pre_files[i]).name)
        torchaudio.save(out_path, out_wav.unsqueeze(0).cpu(), SAMPLE_RATE)
        print(f"  {Path(pre_files[i]).name} -> {Path(out_path).name}")


# ═══════════════════════════════════════════
# Mean-Shift: compute means from train, apply to test
# ═══════════════════════════════════════════
def run_mean_shift(knn_vc, pre_files, post_files, test_idx, train_idx, output_dir):
    print("\n" + "="*60)
    print("  Mean-Shift")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Compute means from TRAIN patients only
    train_pre_feats = []
    train_post_feats = []
    for i in train_idx:
        train_pre_feats.append(knn_vc.get_features(pre_files[i]).cpu())
        train_post_feats.append(knn_vc.get_features(post_files[i]).cpu())

    mean_pre = torch.cat(train_pre_feats, dim=0).mean(dim=0)
    mean_post = torch.cat(train_post_feats, dim=0).mean(dim=0)
    delta = mean_post - mean_pre
    print(f"  Delta norm: {delta.norm():.4f} (from {len(train_idx)} train patients)")

    # Convert test patients
    for i in test_idx:
        features = knn_vc.get_features(pre_files[i])
        converted = features + delta.to(features.device)
        out_path = os.path.join(output_dir, Path(pre_files[i]).name)
        convert_and_save(knn_vc, converted, out_path)
        print(f"  {Path(pre_files[i]).name} -> {Path(out_path).name}")


# ═══════════════════════════════════════════
# MKL-VC: compute OT map from train, apply to test
# ═══════════════════════════════════════════
def compute_mkl_map(X_source, X_target, K=2):
    """Compute factorized MKL optimal transport map."""
    X_s = X_source.numpy().astype(np.float64)
    X_t = X_target.numpy().astype(np.float64)
    D = X_s.shape[1]

    var_s = np.var(X_s, axis=0)
    dim_order = np.argsort(-var_s)

    X_s_sorted = X_s[:, dim_order]
    X_t_sorted = X_t[:, dim_order]

    mu_s = np.mean(X_s_sorted, axis=0)
    mu_t = np.mean(X_t_sorted, axis=0)

    n_groups = D // K
    remainder = D % K
    A_blocks = []

    for g in range(n_groups):
        start = g * K
        end = start + K
        X_sg = X_s_sorted[:, start:end] - mu_s[start:end]
        X_tg = X_t_sorted[:, start:end] - mu_t[start:end]

        Sigma_s = (X_sg.T @ X_sg) / (X_sg.shape[0] - 1) + 1e-6 * np.eye(K)
        Sigma_t = (X_tg.T @ X_tg) / (X_tg.shape[0] - 1) + 1e-6 * np.eye(K)

        Sigma_s_sqrt = sqrtm(Sigma_s).real
        Sigma_s_inv_sqrt = np.linalg.inv(Sigma_s_sqrt)
        inner = Sigma_s_sqrt @ Sigma_t @ Sigma_s_sqrt
        inner_sqrt = sqrtm(inner).real
        A = Sigma_s_inv_sqrt @ inner_sqrt @ Sigma_s_inv_sqrt
        A_blocks.append(torch.from_numpy(A).float())

    if remainder > 0:
        A_blocks.append(torch.eye(remainder))

    return (torch.from_numpy(mu_s).float(), torch.from_numpy(mu_t).float(),
            A_blocks, torch.from_numpy(dim_order).long())


def apply_mkl(features, mu_s, mu_t, A_blocks, dim_order):
    device = features.device
    x_sorted = features[:, dim_order]
    x_centered = x_sorted - mu_s.to(device).unsqueeze(0)
    out_parts = []
    idx = 0
    for A in A_blocks:
        bs = A.shape[0]
        out_parts.append(x_centered[:, idx:idx+bs] @ A.to(device).t())
        idx += bs
    y_sorted = torch.cat(out_parts, dim=1) + mu_t.to(device).unsqueeze(0)
    return y_sorted[:, torch.argsort(dim_order)]


def run_mkl_vc(knn_vc, pre_files, post_files, test_idx, train_idx, output_dir, K=2):
    print("\n" + "="*60)
    print("  MKL-VC")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Compute OT map from TRAIN patients only
    train_pre = torch.cat([knn_vc.get_features(pre_files[i]).cpu() for i in train_idx], dim=0)
    train_post = torch.cat([knn_vc.get_features(post_files[i]).cpu() for i in train_idx], dim=0)
    print(f"  Computing MKL map from {len(train_idx)} train patients "
          f"({train_pre.shape[0]} pre frames, {train_post.shape[0]} post frames)...")

    mu_s, mu_t, A_blocks, dim_order = compute_mkl_map(train_pre, train_post, K=K)

    # Convert test patients
    for i in test_idx:
        features = knn_vc.get_features(pre_files[i])
        converted = apply_mkl(features, mu_s, mu_t, A_blocks, dim_order)
        out_path = os.path.join(output_dir, Path(pre_files[i]).name)
        convert_and_save(knn_vc, converted, out_path)
        print(f"  {Path(pre_files[i]).name} -> {Path(out_path).name}")


# ═══════════════════════════════════════════
# LinearVC: fit ridge regression on train, apply to test
# ═══════════════════════════════════════════
def pair_frames_knn(X, Y):
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        sim = X_norm[i:i+chunk_size] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))
    return X, Y[torch.cat(all_indices)]


def run_linear_vc(knn_vc, pre_files, post_files, test_idx, train_idx, output_dir, reg=1e-3):
    print("\n" + "="*60)
    print("  LinearVC")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    # Extract and pair from TRAIN patients only
    train_pre = torch.cat([knn_vc.get_features(pre_files[i]).cpu() for i in train_idx], dim=0)
    train_post = torch.cat([knn_vc.get_features(post_files[i]).cpu() for i in train_idx], dim=0)
    print(f"  Pairing frames from {len(train_idx)} train patients...")

    X, Y = pair_frames_knn(train_pre, train_post)
    print(f"  Solving ridge regression ({X.shape[0]} pairs, lambda={reg})...")

    X_np = X.numpy().astype(np.float64)
    Y_np = Y.numpy().astype(np.float64)
    XtX = X_np.T @ X_np
    XtY = X_np.T @ Y_np
    W = np.linalg.solve(XtX + reg * np.eye(X_np.shape[1]), XtY)
    W_t = torch.from_numpy(W).float()

    mse = np.mean((X_np @ W - Y_np) ** 2)
    print(f"  Train MSE: {mse:.6f}")

    # Convert test patients
    for i in test_idx:
        features = knn_vc.get_features(pre_files[i])
        converted = features @ W_t.to(features.device)
        out_path = os.path.join(output_dir, Path(pre_files[i]).name)
        convert_and_save(knn_vc, converted, out_path)
        print(f"  {Path(pre_files[i]).name} -> {Path(out_path).name}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Run all baselines with held-out test evaluation")
    parser.add_argument('--surgery', type=str, nargs='+',
                        default=['Tonsill', 'Fess', 'Sept'],
                        help='Surgery types to evaluate')
    parser.add_argument('--n_test', type=int, default=N_TEST)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output_base', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'results_kfold'))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    for surgery in args.surgery:
        pre_dir = os.path.join(CUCO_BASE, surgery, "Speech", "1")
        post_dir = os.path.join(CUCO_BASE, surgery, "Speech", "2")

        print(f"\n{'#'*60}")
        print(f"#  {surgery}")
        print(f"#  Pre:  {pre_dir}")
        print(f"#  Post: {post_dir}")
        print(f"{'#'*60}")

        pre_files, post_files, names = get_patient_files(pre_dir, post_dir)
        n = len(pre_files)
        test_idx, train_idx = split_patients(n, args.n_test, args.seed)

        test_names = [names[i] for i in test_idx]
        train_names = [names[i] for i in train_idx]

        print(f"\n  Total patients: {n}")
        print(f"  Test ({len(test_idx)}): {test_names}")
        print(f"  Train ({len(train_idx)}): {len(train_idx)} patients")

        # Save split info
        out_base = os.path.join(args.output_base, surgery)
        os.makedirs(out_base, exist_ok=True)
        split_info = {
            'seed': args.seed, 'n_test': args.n_test,
            'test_patients': test_names, 'train_patients': train_names,
            'test_indices': test_idx, 'train_indices': train_idx,
        }
        with open(os.path.join(out_base, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)

        # Run each method
        methods = {
            'kNN-VC':     lambda od: run_knn_vc(knn_vc, pre_files, post_files, test_idx, train_idx, od),
            'Mean-Shift': lambda od: run_mean_shift(knn_vc, pre_files, post_files, test_idx, train_idx, od),
            'MKL-VC':     lambda od: run_mkl_vc(knn_vc, pre_files, post_files, test_idx, train_idx, od),
            'LinearVC':   lambda od: run_linear_vc(knn_vc, pre_files, post_files, test_idx, train_idx, od),
        }

        for method_name, run_fn in methods.items():
            method_dir = os.path.join(out_base, method_name.replace(' ', '_').replace('-', '_'))
            run_fn(method_dir)

            # Evaluate
            print(f"\n  Evaluating {method_name}...")
            # Only evaluate test patients (the converted dir only has test files)
            test_pre_dir = os.path.join(out_base, 'test_pre_tmp')
            test_post_dir = os.path.join(out_base, 'test_post_tmp')
            os.makedirs(test_pre_dir, exist_ok=True)
            os.makedirs(test_post_dir, exist_ok=True)

            # Symlink test files
            for i in test_idx:
                pre_src = pre_files[i]
                post_src = post_files[i]
                pre_dst = os.path.join(test_pre_dir, Path(pre_src).name)
                post_dst = os.path.join(test_post_dir, Path(post_src).name)
                if not os.path.exists(pre_dst):
                    os.symlink(pre_src, pre_dst)
                # Post file name might differ — handle session naming
                post_name = Path(pre_src).stem.replace('ses1', 'ses2').replace('_1_', '_2_')
                # Just symlink the post file with the pre file's name pattern
                if not os.path.exists(post_dst):
                    os.symlink(post_src, post_dst)

            run_evaluation(method_dir, test_pre_dir, test_post_dir,
                          f"{method_name} ({surgery}, test)", skip_f0=True)

        # Cleanup temp dirs
        import shutil
        for d in ['test_pre_tmp', 'test_post_tmp']:
            p = os.path.join(out_base, d)
            if os.path.exists(p):
                shutil.rmtree(p)

    print(f"\n{'='*60}")
    print(f"  All done! Results in {args.output_base}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
