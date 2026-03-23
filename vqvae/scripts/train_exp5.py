"""
VQVAE Experiment 5 — WavLM Feature Space

Operates on pre-extracted WavLM features instead of mel-spectrograms.
Uses the same disentanglement framework (VQ + adversarial + quality classification
+ cycle loss) but in a much better feature space.

Steps:
1. Extract WavLM features from all pre/post surgery WAVs (cached after first run)
2. Train VQVAE to disentangle content (VQ codes) from quality (continuous vector)
3. At inference: encode content from source, inject target quality, decode, vocode

Usage:
    python scripts/train_exp5.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.vqvae_wavlm import VQVAEWavLM, DomainClassifier1D, gradient_reversal


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PRE_DIR = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1"
POST_DIR = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'wavlm_cache')
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_exp5')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots_exp5')

# Model
FEAT_DIM = 1024
CODE_DIM = 64
NUM_CODES = 16
NUM_HEADS = 4
QUALITY_DIM = 32
COMMITMENT_WEIGHT = 0.25
EMA_DECAY = 0.95
ENTROPY_WEIGHT = 0.1

# Training
BATCH_SIZE = 8
EPOCHS = 300
LR = 2e-4
LR_ADV = 2e-4
SEGMENT_LEN = 128      # ~2.5s at 50fps WavLM
TARGET_LEN = SEGMENT_LEN

# Loss weights
LAMBDA_RECON = 1.0
LAMBDA_VQ = 1.0
LAMBDA_ADV = 1.0
LAMBDA_QUAL_CLS = 2.0
LAMBDA_CYCLE = 5.0


# ──────────────────────────────────────────────
# Feature extraction + caching
# ──────────────────────────────────────────────

def extract_and_cache_features(knn_vc, wav_dir, cache_path):
    """Extract WavLM features from all WAVs. Returns list of (features, filename) tuples."""
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        print(f"  {len(data)} files, {sum(f.shape[0] for f, _ in data)} total frames")
        return data

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        raise ValueError(f"No WAV files found in {wav_dir}")

    all_data = []
    for wf in wav_files:
        features = knn_vc.get_features(wf).cpu()  # (T, 1024)
        all_data.append((features, Path(wf).stem))
        print(f"  {Path(wf).name}: {features.shape[0]} frames")

    total = sum(f.shape[0] for f, _ in all_data)
    print(f"  Total: {total} frames ({total * 0.02 / 60:.1f} min)")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(all_data, cache_path)
    print(f"  Cached to {cache_path}")
    return all_data


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class WavLMSegmentDataset(Dataset):
    """
    Dataset of WavLM feature segments with domain labels.
    Segments are cut from per-file features with overlap.
    """
    def __init__(self, feature_list, label, segment_len=128, segment_hop=64, augment=False):
        """
        feature_list: list of (features_tensor, filename) tuples
        label: 0 (pre-surgery) or 1 (post-surgery)
        """
        self.segments = []
        self.label = label
        self.augment = augment

        for features, fname in feature_list:
            T = features.shape[0]
            if T < segment_len:
                # Pad short files
                pad = segment_len - T
                features = F.pad(features, (0, 0, 0, pad))
                self.segments.append(features[:segment_len])
            else:
                for start in range(0, T - segment_len + 1, segment_hop):
                    self.segments.append(features[start:start + segment_len])

        print(f"  Label {label}: {len(self.segments)} segments from {len(feature_list)} files")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        feat = self.segments[idx]  # (segment_len, 1024)

        if self.augment:
            # Random noise
            if torch.rand(1).item() > 0.5:
                feat = feat + torch.randn_like(feat) * 0.01

        return feat.t(), torch.tensor(self.label, dtype=torch.float32)  # (1024, T), label


class PairedDomainLoader:
    """Yields (pre_batch, post_batch) pairs, cycling the shorter domain."""
    def __init__(self, loader_a, loader_b):
        self.loader_a = loader_a
        self.loader_b = loader_b

    def __iter__(self):
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        for _ in range(max(len(self.loader_a), len(self.loader_b))):
            try:
                a_feat, a_lab = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a_feat, a_lab = next(iter_a)
            try:
                b_feat, b_lab = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b_feat, b_lab = next(iter_b)
            min_bs = min(a_feat.shape[0], b_feat.shape[0])
            yield a_feat[:min_bs], a_lab[:min_bs], b_feat[:min_bs], b_lab[:min_bs]

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Extract WavLM features ───
    print("Loading kNN-VC model (WavLM + HiFi-GAN)...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, device=device)

    print("\nExtracting pre-surgery features...")
    pre_data = extract_and_cache_features(knn_vc, PRE_DIR,
                                          os.path.join(CACHE_DIR, 'pre_features.pt'))
    print("\nExtracting post-surgery features...")
    post_data = extract_and_cache_features(knn_vc, POST_DIR,
                                           os.path.join(CACHE_DIR, 'post_features.pt'))

    # Free GPU memory from WavLM — we don't need it during training
    del knn_vc
    torch.cuda.empty_cache()

    # ─── Train/val split by file (80/20) ───
    n_pre_train = int(0.8 * len(pre_data))
    n_post_train = int(0.8 * len(post_data))

    pre_train, pre_val = pre_data[:n_pre_train], pre_data[n_pre_train:]
    post_train, post_val = post_data[:n_post_train], post_data[n_post_train:]

    print(f"\nSplit: {n_pre_train}/{len(pre_data)} pre, {n_post_train}/{len(post_data)} post for training")

    # ─── Datasets ───
    print("\nCreating datasets...")
    ds_pre_train = WavLMSegmentDataset(pre_train, label=0, segment_len=SEGMENT_LEN,
                                       segment_hop=64, augment=True)
    ds_post_train = WavLMSegmentDataset(post_train, label=1, segment_len=SEGMENT_LEN,
                                        segment_hop=64, augment=True)
    ds_pre_val = WavLMSegmentDataset(pre_val, label=0, segment_len=SEGMENT_LEN,
                                     segment_hop=SEGMENT_LEN, augment=False)
    ds_post_val = WavLMSegmentDataset(post_val, label=1, segment_len=SEGMENT_LEN,
                                      segment_hop=SEGMENT_LEN, augment=False)

    loader_pre = DataLoader(ds_pre_train, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=2)
    loader_post = DataLoader(ds_post_train, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2)
    paired_loader = PairedDomainLoader(loader_pre, loader_post)

    # Combined val loader
    from torch.utils.data import ConcatDataset
    val_dataset = ConcatDataset([ds_pre_val, ds_post_val])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ─── Also compute average quality vectors for inference ───
    # (will be saved with the checkpoint)

    # ─── Models ───
    model = VQVAEWavLM(
        feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
        num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
        commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
        entropy_weight=ENTROPY_WEIGHT,
    ).to(device)
    domain_cls = DomainClassifier1D(code_dim=CODE_DIM).to(device)
    quality_cls = nn.Linear(QUALITY_DIM, 1).to(device)

    print(f"VQVAE parameters: {model.count_parameters():,}")

    # ─── Optimizers ───
    opt_model = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    opt_adv = torch.optim.Adam(domain_cls.parameters(), lr=LR_ADV)
    opt_qual = torch.optim.Adam(quality_cls.parameters(), lr=LR_ADV)

    sched_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=1e-6)
    sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=EPOCHS, eta_min=1e-6)

    # ─── Logging ───
    history = {
        'recon_loss': [], 'vq_loss': [], 'adv_loss': [],
        'qual_cls_loss': [], 'cycle_loss': [], 'perplexity': [],
        'val_recon': [],
    }
    best_val_loss = float('inf')

    # ─── Training Loop ───
    for epoch in range(EPOCHS):
        model.train()
        domain_cls.train()
        quality_cls.train()

        ep = {k: 0.0 for k in ['recon', 'vq', 'adv', 'qual', 'cycle', 'perp']}
        n_batches = 0

        pbar = tqdm(paired_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for feat_pre, lab_pre, feat_post, lab_post in pbar:
            feat_pre = feat_pre.to(device)     # (B, 1024, T)
            feat_post = feat_post.to(device)
            lab_pre = lab_pre.to(device)
            lab_post = lab_post.to(device)

            feat = torch.cat([feat_pre, feat_post], dim=0)
            labels = torch.cat([lab_pre, lab_post], dim=0)

            # ═══ Step 1: Train adversarial classifier ═══
            with torch.no_grad():
                content_z_all = model.content_encoder(feat)

            opt_adv.zero_grad()
            adv_pred = domain_cls(content_z_all.detach())
            loss_adv_cls = F.binary_cross_entropy_with_logits(adv_pred.squeeze(1), labels)
            loss_adv_cls.backward()
            opt_adv.step()

            # ═══ Step 2: Train VQVAE + quality classifier ═══
            opt_model.zero_grad()
            opt_qual.zero_grad()

            # Self-reconstruction
            recon, vq_loss, perplexity, content_z = model(feat)
            loss_recon = F.mse_loss(recon, feat)

            # Adversarial disentanglement
            content_reversed = gradient_reversal(content_z, alpha=LAMBDA_ADV)
            adv_pred_gr = domain_cls(content_reversed)
            loss_adv_g = F.binary_cross_entropy_with_logits(adv_pred_gr.squeeze(1), labels)

            # Quality classification
            quality = model.quality_encoder(feat)
            qual_pred = quality_cls(quality)
            loss_qual = F.binary_cross_entropy_with_logits(qual_pred.squeeze(1), labels)

            # ═══ Step 3: Cycle loss ═══
            B = feat_pre.shape[0]

            content_z_pre = model.content_encoder(feat_pre)
            content_q_pre, _, _ = model.vq(content_z_pre)
            quality_pre = model.quality_encoder(feat_pre)

            content_z_post = model.content_encoder(feat_post)
            content_q_post, _, _ = model.vq(content_z_post)
            quality_post = model.quality_encoder(feat_post)

            # Cross-reconstruct
            cross_pre2post = model.decoder(content_q_pre, quality_post)
            cross_pre2post = model._match_time(cross_pre2post, feat_pre)

            cross_post2pre = model.decoder(content_q_post, quality_pre)
            cross_post2pre = model._match_time(cross_post2pre, feat_post)

            # Re-encode
            re_content_q_a2b, _, _ = model.vq(model.content_encoder(cross_pre2post))
            re_quality_a2b = model.quality_encoder(cross_pre2post)

            re_content_q_b2a, _, _ = model.vq(model.content_encoder(cross_post2pre))
            re_quality_b2a = model.quality_encoder(cross_post2pre)

            loss_cycle_content = (F.l1_loss(re_content_q_a2b, content_q_pre.detach())
                                + F.l1_loss(re_content_q_b2a, content_q_post.detach()))
            loss_cycle_quality = (F.l1_loss(re_quality_a2b, quality_post.detach())
                                + F.l1_loss(re_quality_b2a, quality_pre.detach()))
            loss_cycle = loss_cycle_content + loss_cycle_quality

            # ═══ Total loss ═══
            loss_total = (LAMBDA_RECON * loss_recon
                         + LAMBDA_VQ * vq_loss
                         + loss_adv_g
                         + LAMBDA_QUAL_CLS * loss_qual
                         + LAMBDA_CYCLE * loss_cycle)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step()
            opt_qual.step()

            ep['recon'] += loss_recon.item()
            ep['vq'] += vq_loss.item()
            ep['adv'] += loss_adv_cls.item()
            ep['qual'] += loss_qual.item()
            ep['cycle'] += loss_cycle.item()
            ep['perp'] += perplexity.item()
            n_batches += 1

            pbar.set_postfix({
                'recon': f'{loss_recon.item():.4f}',
                'cycle': f'{loss_cycle.item():.3f}',
                'perp': f'{perplexity.item():.0f}',
                'adv': f'{loss_adv_cls.item():.3f}',
            })

        # Average epoch metrics
        for key, ep_key in [('recon_loss', 'recon'), ('vq_loss', 'vq'), ('adv_loss', 'adv'),
                            ('qual_cls_loss', 'qual'), ('cycle_loss', 'cycle'), ('perplexity', 'perp')]:
            history[key].append(ep[ep_key] / max(n_batches, 1))

        sched_model.step()
        sched_adv.step()

        # ─── Validation ───
        model.eval()
        val_recon = 0
        n_val = 0
        with torch.no_grad():
            for feat_v, _ in val_loader:
                feat_v = feat_v.to(device)
                recon_v, _, _, _ = model(feat_v)
                val_recon += F.mse_loss(recon_v, feat_v).item()
                n_val += 1

        avg_val = val_recon / max(n_val, 1)
        history['val_recon'].append(avg_val)

        print(f"Epoch {epoch+1} | Recon: {history['recon_loss'][-1]:.4f} | "
              f"VQ: {history['vq_loss'][-1]:.4f} | Perp: {history['perplexity'][-1]:.0f} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | "
              f"Adv: {history['adv_loss'][-1]:.4f} | Qual: {history['qual_cls_loss'][-1]:.4f} | "
              f"Val Recon: {avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Compute average quality vectors for inference
            avg_quality_pre, avg_quality_post = compute_avg_quality(
                model, pre_data, post_data, device, SEGMENT_LEN)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'domain_cls': domain_cls.state_dict(),
                'quality_cls': quality_cls.state_dict(),
                'avg_quality_pre': avg_quality_pre,
                'avg_quality_post': avg_quality_post,
                'val_loss': avg_val,
                'config': {
                    'feat_dim': FEAT_DIM, 'code_dim': CODE_DIM,
                    'num_codes': NUM_CODES, 'num_heads': NUM_HEADS,
                    'quality_dim': QUALITY_DIM,
                },
            }, os.path.join(CHECKPOINT_DIR, 'best_vqvae_wavlm.pth'))
            print(f"  -> Saved best model (val={avg_val:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'vqvae_wavlm_epoch{epoch+1}.pth'))

        # ─── Plot every 20 epochs ───
        if (epoch + 1) % 20 == 0:
            plot_training_curves(history, epoch + 1)

    # Final plot
    plot_training_curves(history, EPOCHS)
    print(f"\nTraining complete. Best val recon loss: {best_val_loss:.4f}")


def compute_avg_quality(model, pre_data, post_data, device, segment_len):
    """Compute average quality vectors across all files for each domain."""
    model.eval()
    qualities_pre = []
    qualities_post = []

    with torch.no_grad():
        for features, _ in pre_data:
            T = features.shape[0]
            if T < segment_len:
                features = F.pad(features, (0, 0, 0, segment_len - T))
            # Process in chunks
            for start in range(0, features.shape[0] - segment_len + 1, segment_len):
                seg = features[start:start + segment_len].t().unsqueeze(0).to(device)
                q = model.quality_encoder(seg)
                qualities_pre.append(q.cpu())

        for features, _ in post_data:
            T = features.shape[0]
            if T < segment_len:
                features = F.pad(features, (0, 0, 0, segment_len - T))
            for start in range(0, features.shape[0] - segment_len + 1, segment_len):
                seg = features[start:start + segment_len].t().unsqueeze(0).to(device)
                q = model.quality_encoder(seg)
                qualities_post.append(q.cpu())

    avg_pre = torch.cat(qualities_pre, dim=0).mean(dim=0)   # (quality_dim,)
    avg_post = torch.cat(qualities_post, dim=0).mean(dim=0)
    print(f"  Avg quality vectors: pre={avg_pre.shape}, post={avg_post.shape}")
    return avg_pre, avg_post


def plot_training_curves(history, epoch):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(history['recon_loss'], label='Train')
    axes[0, 0].plot(history['val_recon'], label='Val')
    axes[0, 0].set_title('Reconstruction Loss (MSE on WavLM features)')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['vq_loss'], label='VQ (commit + entropy)')
    axes[0, 1].set_title('VQ Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[0, 2].plot(history['perplexity'], label='Avg Perplexity')
    axes[0, 2].axhline(y=NUM_CODES, color='r', linestyle='--', label=f'Max per head ({NUM_CODES})')
    axes[0, 2].set_title(f'Codebook Perplexity ({NUM_HEADS}×{NUM_CODES})')
    axes[0, 2].legend(); axes[0, 2].grid(True)

    axes[1, 0].plot(history['cycle_loss'], label='Cycle')
    axes[1, 0].set_title('Cycle Loss'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['adv_loss'], label='Adv (content)')
    axes[1, 1].axhline(y=0.693, color='r', linestyle='--', label='Random (0.693)')
    axes[1, 1].set_title('Adversarial Loss (want ~0.693)')
    axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].plot(history['qual_cls_loss'], label='Quality cls')
    axes[1, 2].set_title('Quality Classification (want low)')
    axes[1, 2].legend(); axes[1, 2].grid(True)

    plt.suptitle(f'VQVAE Exp 5 (WavLM) — Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'training_curves_epoch{epoch}.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    train()
