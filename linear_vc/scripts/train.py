"""
LinearVC — Learn a Linear Domain Transform in WavLM Space

Based on: "LinearVC: Linear transformations of self-supervised features
through the lens of voice conversion" (2025).

Learns a projection matrix W such that:
    converted_features = source_features @ W

The matrix W is learned by:
1. Extracting WavLM features from pre and post surgery audio
2. Pairing frames via nearest neighbors (pre <-> post)
3. Solving least squares: W = argmin ||Y - XW||^2

Usage:
    python scripts/train.py
"""

import argparse
import os
import glob
import torch
import numpy as np
from pathlib import Path


SAMPLE_RATE = 16000


def extract_all_features(knn_vc, wav_dir):
    """Extract WavLM features from all WAV files."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    all_features = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)  # (T, 1024)
        all_features.append(features.cpu())
        print(f"  {Path(wf).name}: {features.shape[0]} frames")

    combined = torch.cat(all_features, dim=0)
    print(f"  Total: {combined.shape[0]} frames ({combined.shape[0] * 0.02 / 60:.1f} min)")
    return combined


def pair_frames_knn(X, Y, k=1):
    """
    Pair source frames (X) to target frames (Y) via nearest neighbors.
    For each frame in X, find the closest frame in Y.

    Returns paired (X_paired, Y_paired) of same length.
    """
    print(f"  Pairing {X.shape[0]} source frames to {Y.shape[0]} target frames (k={k})...")

    # Normalize for cosine similarity
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)

    # Process in chunks to avoid OOM
    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        X_chunk = X_norm[i:i + chunk_size]
        sim = X_chunk @ Y_norm.t()  # (chunk, N_Y)
        if k == 1:
            indices = sim.argmax(dim=1)
        else:
            _, topk_idx = sim.topk(k, dim=1)
            indices = topk_idx[:, 0]  # take closest
        all_indices.append(indices)

    indices = torch.cat(all_indices)
    X_paired = X
    Y_paired = Y[indices]
    print(f"  Paired: {X_paired.shape[0]} frame pairs")
    return X_paired, Y_paired


def main():
    parser = argparse.ArgumentParser(description="LinearVC — Learn domain transform")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'linear_transform.pt'))
    parser.add_argument('--regularization', type=float, default=1e-3,
                        help='Ridge regularization strength (prevents overfitting)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Extract features
    print(f"\nExtracting pre-surgery features...")
    features_pre = extract_all_features(knn_vc, args.pre_dir)

    print(f"\nExtracting post-surgery features...")
    features_post = extract_all_features(knn_vc, args.post_dir)

    # Pair frames via nearest neighbors
    print(f"\nPairing frames (pre -> post)...")
    X, Y = pair_frames_knn(features_pre, features_post)

    # Solve ridge regression: W = (X^T X + lambda I)^{-1} X^T Y
    print(f"\nSolving linear regression (d={X.shape[1]}, n={X.shape[0]}, lambda={args.regularization})...")
    X_np = X.numpy().astype(np.float64)
    Y_np = Y.numpy().astype(np.float64)

    XtX = X_np.T @ X_np  # (1024, 1024)
    XtY = X_np.T @ Y_np  # (1024, 1024)
    reg = args.regularization * np.eye(X_np.shape[1])
    W = np.linalg.solve(XtX + reg, XtY)  # (1024, 1024)

    # Compute training error
    Y_pred = X_np @ W
    mse = np.mean((Y_pred - Y_np) ** 2)
    print(f"  Training MSE: {mse:.6f}")

    # Also solve reverse direction (post -> pre)
    print(f"\nPairing frames (post -> pre)...")
    X_rev, Y_rev = pair_frames_knn(features_post, features_pre)

    print(f"Solving reverse linear regression...")
    X_rev_np = X_rev.numpy().astype(np.float64)
    Y_rev_np = Y_rev.numpy().astype(np.float64)

    XtX_rev = X_rev_np.T @ X_rev_np
    XtY_rev = X_rev_np.T @ Y_rev_np
    W_rev = np.linalg.solve(XtX_rev + reg, XtY_rev)

    Y_rev_pred = X_rev_np @ W_rev
    mse_rev = np.mean((Y_rev_pred - Y_rev_np) ** 2)
    print(f"  Training MSE (reverse): {mse_rev:.6f}")

    # Save
    W_tensor = torch.from_numpy(W).float()       # (1024, 1024)
    W_rev_tensor = torch.from_numpy(W_rev).float()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'W_a2b': W_tensor,       # pre -> post
        'W_b2a': W_rev_tensor,   # post -> pre
        'mse_a2b': mse,
        'mse_b2a': mse_rev,
        'n_pairs_a2b': X.shape[0],
        'n_pairs_b2a': X_rev.shape[0],
        'regularization': args.regularization,
    }, args.output)
    print(f"\nSaved linear transforms to {args.output}")
    print(f"  W_a2b: {W_tensor.shape}, norm={W_tensor.norm():.4f}")
    print(f"  W_b2a: {W_rev_tensor.shape}, norm={W_rev_tensor.norm():.4f}")


if __name__ == '__main__':
    main()
