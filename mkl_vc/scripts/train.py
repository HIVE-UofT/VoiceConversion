"""
MKL-VC — Factorized Optimal Transport Voice Conversion (Training)

Based on: "Training-Free Voice Conversion with Factorized Optimal Transport"
(Interspeech 2025).

Computes the Monge-Kantorovich Linear (MKL) optimal transport map between
pre-surgery and post-surgery WavLM feature distributions.

The key insight: WavLM features have non-uniform variance across dimensions.
We factorize the 1024-dim space into K-dimensional subgroups sorted by variance,
then compute the Gaussian OT map independently per subgroup.

The OT map for Gaussians has a closed-form solution:
    T(x) = mu_t + A @ (x - mu_s)
    where A = Sigma_s^{-1/2} @ (Sigma_s^{1/2} @ Sigma_t @ Sigma_s^{1/2})^{1/2} @ Sigma_s^{-1/2}

Usage:
    python scripts/train.py
"""

import argparse
import os
import glob
import torch
import numpy as np
from pathlib import Path
from scipy.linalg import sqrtm


SAMPLE_RATE = 16000


def extract_all_features(knn_vc, wav_dir):
    """Extract WavLM features from all WAV files."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    all_features = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)
        all_features.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")

    combined = torch.cat(all_features, dim=0)
    print(f"  Total: {combined.shape[0]} frames ({combined.shape[0] * 0.02 / 60:.1f} min)")
    return combined


def compute_mkl_map(X_source, X_target, K=2):
    """
    Compute the factorized MKL optimal transport map.

    1. Sort dimensions by variance
    2. Split into subgroups of size K
    3. For each subgroup, compute Gaussian OT map (closed-form)

    Returns: (mean_s, mean_t, A_blocks, dim_order, K) — everything needed to apply the map.
    """
    X_s = X_source.numpy().astype(np.float64)
    X_t = X_target.numpy().astype(np.float64)
    D = X_s.shape[1]  # 1024

    # Sort dimensions by variance (descending)
    var_s = np.var(X_s, axis=0)
    dim_order = np.argsort(-var_s)  # high variance first

    # Reorder
    X_s_sorted = X_s[:, dim_order]
    X_t_sorted = X_t[:, dim_order]

    # Global means
    mu_s = np.mean(X_s_sorted, axis=0)
    mu_t = np.mean(X_t_sorted, axis=0)

    # Compute OT map per K-dimensional subgroup
    n_groups = D // K
    remainder = D % K
    A_blocks = []

    for g in range(n_groups):
        start = g * K
        end = start + K

        X_sg = X_s_sorted[:, start:end] - mu_s[start:end]
        X_tg = X_t_sorted[:, start:end] - mu_t[start:end]

        # Covariance matrices
        Sigma_s = (X_sg.T @ X_sg) / (X_sg.shape[0] - 1)
        Sigma_t = (X_tg.T @ X_tg) / (X_tg.shape[0] - 1)

        # Regularize for numerical stability
        reg = 1e-6 * np.eye(K)
        Sigma_s += reg
        Sigma_t += reg

        # MKL map: A = Sigma_s^{-1/2} @ (Sigma_s^{1/2} @ Sigma_t @ Sigma_s^{1/2})^{1/2} @ Sigma_s^{-1/2}
        Sigma_s_sqrt = sqrtm(Sigma_s).real
        Sigma_s_inv_sqrt = np.linalg.inv(Sigma_s_sqrt)

        inner = Sigma_s_sqrt @ Sigma_t @ Sigma_s_sqrt
        inner_sqrt = sqrtm(inner).real

        A = Sigma_s_inv_sqrt @ inner_sqrt @ Sigma_s_inv_sqrt
        A_blocks.append(A)

    # Handle remainder dimensions (if D not divisible by K) with identity
    if remainder > 0:
        A_blocks.append(np.eye(remainder))

    return mu_s, mu_t, A_blocks, dim_order, K


def main():
    parser = argparse.ArgumentParser(description="MKL-VC — Compute optimal transport map")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'mkl_transform.pt'))
    parser.add_argument('--K', type=int, default=2,
                        help='Subgroup dimension for factorized OT (paper recommends K=2)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print(f"\nExtracting pre-surgery features...")
    features_pre = extract_all_features(knn_vc, args.pre_dir)

    print(f"\nExtracting post-surgery features...")
    features_post = extract_all_features(knn_vc, args.post_dir)

    # Compute forward map (pre -> post)
    print(f"\nComputing MKL map (pre -> post, K={args.K})...")
    mu_s, mu_t, A_blocks, dim_order, K = compute_mkl_map(features_pre, features_post, K=args.K)
    print(f"  {len(A_blocks)} subgroups of size {K}")

    # Compute reverse map (post -> pre)
    print(f"\nComputing MKL map (post -> pre, K={args.K})...")
    mu_s_rev, mu_t_rev, A_blocks_rev, dim_order_rev, _ = compute_mkl_map(features_post, features_pre, K=args.K)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        # Forward (pre -> post)
        'mu_s': torch.from_numpy(mu_s).float(),
        'mu_t': torch.from_numpy(mu_t).float(),
        'A_blocks': [torch.from_numpy(A).float() for A in A_blocks],
        'dim_order': torch.from_numpy(dim_order).long(),
        'K': K,
        # Reverse (post -> pre)
        'mu_s_rev': torch.from_numpy(mu_s_rev).float(),
        'mu_t_rev': torch.from_numpy(mu_t_rev).float(),
        'A_blocks_rev': [torch.from_numpy(A).float() for A in A_blocks_rev],
        'dim_order_rev': torch.from_numpy(dim_order_rev).long(),
    }, args.output)
    print(f"\nSaved MKL transforms to {args.output}")


if __name__ == '__main__':
    main()
