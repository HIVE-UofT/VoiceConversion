"""
UNet-VC v2 — Leave-One-Out Cross-Validation with Generalization Improvements

Key changes from v1:
1. Cross-patient frame pooling: NN pairing against ALL other patients' post-surgery
   frames, not just the same patient's. Learns general surgery effect.
2. Smaller model: hidden=64 (vs 128), stronger dropout=0.4 (vs 0.25)
3. Stronger augmentation: noise_std=0.05, mask_prob=0.2, random gain
4. Leave-one-out: train on N-1, test on 1, repeat N times
5. Reports both train and test metrics

Usage:
    python scripts/train_loo.py
    python scripts/train_loo.py --pre_dir /path/to/pre --post_dir /path/to/post
"""

import argparse
import os
import sys
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D


SAMPLE_RATE = 16000

# v2 config — smaller + more regularized
HIDDEN_DIM = 64
N_LEVELS = 2
DROPOUT = 0.4
BATCH_SIZE = 32
SEGMENT_LEN = 64
SEGMENT_HOP = 16
LR = 3e-4
WEIGHT_DECAY = 2e-3
EPOCHS = 200
PATIENCE = 30
COSINE_LOSS_WEIGHT = 0.5
AUGMENT_NOISE_STD = 0.05
AUGMENT_MASK_PROB = 0.2
AUGMENT_GAIN_RANGE = 0.1  # random gain ±10%


def extract_all_features(knn_vc, wav_dir):
    """Extract WavLM features. Returns list of (filename, features_tensor)."""
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")
    results = []
    for wf in wav_files:
        features = knn_vc.get_features(wf)
        results.append((Path(wf).stem, features.cpu(), wf))
        print(f"  {Path(wf).name}: {features.shape[0]} frames")
    total = sum(f.shape[0] for _, f, _ in results)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")
    return results


def pair_frames_knn(X, Y):
    """Pair source frames (X) to target frames (Y) via cosine NN."""
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (Y.norm(dim=1, keepdim=True) + 1e-8)
    chunk_size = 5000
    all_indices = []
    for i in range(0, X.shape[0], chunk_size):
        sim = X_norm[i:i + chunk_size] @ Y_norm.t()
        all_indices.append(sim.argmax(dim=1))
    indices = torch.cat(all_indices)
    return X, Y[indices]


def build_segments_cross_patient(pre_features_list, post_features_list,
                                  patient_indices, segment_len=SEGMENT_LEN,
                                  segment_hop=SEGMENT_HOP):
    """
    Build training segments with CROSS-PATIENT frame pooling.

    For each patient in patient_indices:
    - Take their pre-surgery frames
    - Pool ALL other training patients' post-surgery frames as matching targets
    - NN-pair against the pooled post-surgery set
    - This teaches the model "general post-surgery" not "this patient's post-surgery"
    """
    segments = []

    # Pool all post-surgery frames from training patients
    post_pool = torch.cat([post_features_list[i] for i in patient_indices], dim=0)
    print(f"    Post-surgery pool: {post_pool.shape[0]} frames from {len(patient_indices)} patients")

    for idx in patient_indices:
        pre_feat = pre_features_list[idx]
        # Pair against the POOLED post-surgery frames (cross-patient)
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_pool)

        n_frames = X_paired.shape[0]
        if n_frames < segment_len:
            continue

        for start in range(0, n_frames - segment_len + 1, segment_hop):
            end = start + segment_len
            segments.append((
                X_paired[start:end].t(),
                Y_paired[start:end].t(),
            ))

    print(f"    Created {len(segments)} segments")
    return segments


def build_segments_same_patient(pre_features_list, post_features_list,
                                 patient_indices, segment_len=SEGMENT_LEN,
                                 segment_hop=SEGMENT_HOP):
    """
    Build segments with SAME-PATIENT pairing (v1 style, for comparison).
    """
    segments = []
    for idx in patient_indices:
        pre_feat = pre_features_list[idx]
        post_feat = post_features_list[idx]
        X_paired, Y_paired = pair_frames_knn(pre_feat, post_feat)
        n_frames = X_paired.shape[0]
        if n_frames < segment_len:
            continue
        for start in range(0, n_frames - segment_len + 1, segment_hop):
            end = start + segment_len
            segments.append((
                X_paired[start:end].t(),
                Y_paired[start:end].t(),
            ))
    print(f"    Created {len(segments)} segments (same-patient pairing)")
    return segments


class FeatureSegmentDataset(Dataset):
    """Dataset with stronger augmentation."""

    def __init__(self, segments, augment=False, noise_std=0.05,
                 mask_prob=0.2, gain_range=0.1):
        self.segments = segments
        self.augment = augment
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.gain_range = gain_range

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y = self.segments[idx]
        if self.augment:
            # Gaussian noise
            x = x + torch.randn_like(x) * self.noise_std
            # Random frame masking
            mask = torch.rand(x.shape[-1]) > self.mask_prob
            x = x * mask.unsqueeze(0)
            # Random gain perturbation
            gain = 1.0 + (torch.rand(1).item() * 2 - 1) * self.gain_range
            x = x * gain
        return x, y


def combined_loss(y_pred, y_target, cosine_weight=0.5):
    mse = F.mse_loss(y_pred, y_target)
    cos_sim = F.cosine_similarity(y_pred, y_target, dim=1).mean()
    cosine_loss = 1.0 - cos_sim
    return mse + cosine_weight * cosine_loss, mse.item(), cosine_loss.item()


def train_model(train_indices, val_indices, pre_features, post_features,
                device, output_path, tag="", cross_patient=True):
    """Train a model. Returns best val loss."""

    if cross_patient:
        train_segs = build_segments_cross_patient(
            pre_features, post_features, train_indices)
    else:
        train_segs = build_segments_same_patient(
            pre_features, post_features, train_indices)

    # Val always uses same-patient (honest evaluation)
    val_segs = build_segments_same_patient(
        pre_features, post_features, val_indices)

    train_dataset = FeatureSegmentDataset(
        train_segs, augment=True, noise_std=AUGMENT_NOISE_STD,
        mask_prob=AUGMENT_MASK_PROB, gain_range=AUGMENT_GAIN_RANGE)
    val_dataset = FeatureSegmentDataset(val_segs, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"  {tag} Train: {len(train_dataset)} segs ({len(train_indices)} patients), "
          f"Val: {len(val_dataset)} segs ({len(val_indices)} patients)")

    model = ResUNet1D(feat_dim=1024, hidden_dim=HIDDEN_DIM, n_levels=N_LEVELS,
                      dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss, _, _ = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss, _, _ = combined_loss(y_pred, y_batch, COSINE_LOSS_WEIGHT)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses) if val_losses else float('inf')
        alpha_val = model.alpha.item()

        if epoch % 20 == 0 or epoch == 1:
            print(f"    {tag} Epoch {epoch:3d}  val={val_loss:.6f}  alpha={alpha_val:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'alpha': alpha_val,
                'config': {
                    'feat_dim': 1024,
                    'hidden_dim': HIDDEN_DIM,
                    'n_levels': N_LEVELS,
                    'dropout': DROPOUT,
                },
            }, output_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    {tag} Early stop at epoch {epoch}")
                break

    print(f"    {tag} Best val: {best_val_loss:.6f}")
    return best_val_loss


def convert_and_evaluate(knn_vc, model, pre_data, post_data, patient_idx,
                          output_dir, device, ecapa_encoder):
    """Convert one patient and compute SpkSim."""
    name, pre_feat, pre_wav = pre_data[patient_idx]
    _, _, post_wav = post_data[patient_idx]

    # Convert
    features = knn_vc.get_features(pre_wav)
    with torch.no_grad():
        x = features.t().unsqueeze(0).to(device)
        y = model(x)
        converted = y.squeeze(0).t()
    out_wav = knn_vc.vocode(converted[None]).cpu().squeeze()
    out_path = os.path.join(output_dir, name + '.wav')
    torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

    # SpkSim: converted vs post
    from speechbrain.inference.speaker import EncoderClassifier

    def get_emb(wav_path):
        sig, sr = torchaudio.load(wav_path)
        if sr != 16000:
            sig = torchaudio.functional.resample(sig, sr, 16000)
        if sig.shape[0] > 1:
            sig = sig.mean(dim=0, keepdim=True)
        return ecapa_encoder.encode_batch(sig).squeeze()

    emb_conv = get_emb(out_path)
    emb_post = get_emb(post_wav)
    emb_pre = get_emb(pre_wav)

    sim_post = F.cosine_similarity(emb_conv.unsqueeze(0), emb_post.unsqueeze(0)).item()
    sim_pre = F.cosine_similarity(emb_conv.unsqueeze(0), emb_pre.unsqueeze(0)).item()
    baseline = F.cosine_similarity(emb_pre.unsqueeze(0), emb_post.unsqueeze(0)).item()

    return sim_post, sim_pre, baseline


def main():
    parser = argparse.ArgumentParser(description="UNet-VC v2 — LOO with generalization improvements")
    parser.add_argument('--pre_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1")
    parser.add_argument('--post_dir', type=str,
                        default="/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2")
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results'))
    parser.add_argument('--cross_patient', action='store_true', default=True,
                        help='Use cross-patient frame pooling (default: True)')
    parser.add_argument('--same_patient', action='store_true', default=False,
                        help='Use same-patient pairing (v1 style, for ablation)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    use_cross = not args.same_patient

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frame pairing: {'CROSS-PATIENT' if use_cross else 'SAME-PATIENT'}")
    print(f"Model: hidden={HIDDEN_DIM}, dropout={DROPOUT}")
    os.makedirs(args.output, exist_ok=True)

    # Load WavLM
    print("Loading kNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    # Extract features
    print(f"\nExtracting pre-surgery features...")
    pre_data = extract_all_features(knn_vc, args.pre_dir)
    print(f"\nExtracting post-surgery features...")
    post_data = extract_all_features(knn_vc, args.post_dir)

    assert len(pre_data) == len(post_data)
    n_patients = len(pre_data)

    pre_features = [feat for _, feat, _ in pre_data]
    post_features = [feat for _, feat, _ in post_data]
    pre_names = [name for name, _, _ in pre_data]

    # Load ECAPA-TDNN for evaluation
    print("\nLoading ECAPA-TDNN...")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )

    # ═══ Leave-One-Out Cross-Validation ═══
    print(f"\n{'='*70}")
    print(f"  Leave-One-Out CV: {n_patients} patients")
    print(f"  Each round: train on {n_patients-1}, test on 1")
    print(f"{'='*70}")

    all_sim_post = []
    all_sim_pre = []
    all_baseline = []
    conv_dir = os.path.join(args.output, 'converted_loo')
    os.makedirs(conv_dir, exist_ok=True)

    for test_idx in range(n_patients):
        test_name = pre_names[test_idx]
        train_indices = [i for i in range(n_patients) if i != test_idx]

        # Use 3 random patients from train as val for early stopping
        np.random.seed(args.seed + test_idx)
        val_indices = list(np.random.choice(train_indices, size=min(3, len(train_indices) - 1), replace=False))
        pure_train = [i for i in train_indices if i not in val_indices]

        print(f"\n--- LOO {test_idx+1}/{n_patients}: test={test_name} ---")

        ckpt_path = os.path.join(args.output, f'loo_{test_idx}_model.pt')
        train_model(pure_train, val_indices, pre_features, post_features,
                    device, ckpt_path, tag=f"LOO-{test_idx+1}",
                    cross_patient=use_cross)

        # Load best model and evaluate
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

        sim_post, sim_pre, baseline = convert_and_evaluate(
            knn_vc, model, pre_data, post_data, test_idx,
            conv_dir, device, ecapa
        )

        all_sim_post.append(sim_post)
        all_sim_pre.append(sim_pre)
        all_baseline.append(baseline)

        print(f"    {test_name}: conv→post={sim_post:.3f}  conv→pre={sim_pre:.3f}  "
              f"baseline={baseline:.3f}  improvement={sim_post-baseline:+.3f}")

        # Clean up model checkpoint to save disk (keep converted audio)
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # ═══ Summary ═══
    print(f"\n{'='*70}")
    print(f"  LOO Results ({n_patients} patients)")
    print(f"  Pairing: {'CROSS-PATIENT' if use_cross else 'SAME-PATIENT'}")
    print(f"  Model: hidden={HIDDEN_DIM}, dropout={DROPOUT}")
    print(f"{'='*70}")
    print(f"  Baseline (no conversion):    {np.mean(all_baseline):.3f} +/- {np.std(all_baseline):.3f}")
    print(f"  UNet-VC v2 (conv vs post):   {np.mean(all_sim_post):.3f} +/- {np.std(all_sim_post):.3f}")
    print(f"  UNet-VC v2 (conv vs pre):    {np.mean(all_sim_pre):.3f} +/- {np.std(all_sim_pre):.3f}")
    print(f"  Improvement over baseline:   {np.mean(all_sim_post) - np.mean(all_baseline):+.3f}")
    print(f"{'='*70}")

    # Per-patient results
    print(f"\nPer-patient results:")
    print(f"{'Patient':<35} {'Baseline':>8} {'Conv→Post':>10} {'Conv→Pre':>9} {'Δ':>7}")
    print("-" * 75)
    for i in range(n_patients):
        delta = all_sim_post[i] - all_baseline[i]
        marker = "✓" if delta > 0 else "✗"
        print(f"  {pre_names[i]:<33} {all_baseline[i]:>8.3f} {all_sim_post[i]:>10.3f} "
              f"{all_sim_pre[i]:>9.3f} {delta:>+7.3f} {marker}")

    n_improved = sum(1 for i in range(n_patients) if all_sim_post[i] > all_baseline[i])
    print(f"\nImproved: {n_improved}/{n_patients} patients ({100*n_improved/n_patients:.0f}%)")

    # Save results
    results = {
        'method': 'UNet-VC v2',
        'pairing': 'cross-patient' if use_cross else 'same-patient',
        'hidden_dim': HIDDEN_DIM,
        'dropout': DROPOUT,
        'n_patients': n_patients,
        'baseline_mean': float(np.mean(all_baseline)),
        'baseline_std': float(np.std(all_baseline)),
        'conv_vs_post_mean': float(np.mean(all_sim_post)),
        'conv_vs_post_std': float(np.std(all_sim_post)),
        'conv_vs_pre_mean': float(np.mean(all_sim_pre)),
        'conv_vs_pre_std': float(np.std(all_sim_pre)),
        'per_patient': [
            {'name': pre_names[i], 'baseline': all_baseline[i],
             'conv_vs_post': all_sim_post[i], 'conv_vs_pre': all_sim_pre[i]}
            for i in range(n_patients)
        ]
    }
    results_path = os.path.join(args.output, 'loo_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
