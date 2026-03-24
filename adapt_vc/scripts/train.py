"""
AdaptVC-Inspired — Training Script

Trains on raw audio with on-the-fly WavLM feature extraction.
WavLM-Large stays frozen on GPU; only adapters, encoders, VQ, and decoder are trained.

Key differences from VQVAE Exp5/6:
  - Raw audio input (not cached WavLM features)
  - Dual adapters learn which WavLM layers to use for content vs quality
  - Reconstruction target: WavLM layer 6 features (vocoder-compatible)
  - Same disentanglement losses (adversarial, quality cls, cycle, cross-recon)

Usage:
    python scripts/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.adapt_vc import AdaptVCModel, DomainClassifier1D, gradient_reversal


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PRE_DIR = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/1"
POST_DIR = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios/Tonsill/Speech/2"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')

# WavLM
WAVLM_LAYER_FOR_VOCODER = 6  # knn-vc HiFi-GAN was trained on layer 6
SAMPLE_RATE = 16000

# Model
FEAT_DIM = 1024
CODE_DIM = 64
NUM_CODES = 32
NUM_HEADS = 4
QUALITY_DIM = 64
COMMITMENT_WEIGHT = 0.25
EMA_DECAY = 0.99
ENTROPY_WEIGHT = 0.5
DROPOUT = 0.15
CONTENT_NOISE = 0.1

# Training
BATCH_SIZE = 8
EPOCHS = 400
LR = 1e-4
LR_ADV = 1e-4
SEGMENT_SAMPLES = 40000   # 2.5s at 16kHz → 125 WavLM frames
SEGMENT_HOP_SAMPLES = 20000  # 1.25s hop
WARMUP_EPOCHS = 30

# Loss weights
LAMBDA_RECON = 5.0
LAMBDA_VQ = 1.0
LAMBDA_ADV = 1.5
LAMBDA_QUAL_CLS = 5.0
LAMBDA_CYCLE = 2.0
LAMBDA_CROSS_RECON = 2.0


# ──────────────────────────────────────────────
# WavLM Feature Extractor
# ──────────────────────────────────────────────

class WavLMFeatureExtractor:
    """
    Extracts all hidden states from frozen WavLM-Large.
    Uses HuggingFace transformers for clean multi-layer access.
    """
    def __init__(self, device):
        from transformers import WavLMModel, AutoFeatureExtractor
        print("Loading WavLM-Large from HuggingFace...")
        self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers  # 24
        print(f"  WavLM-Large loaded: {self.num_layers} layers, {FEAT_DIM}-dim")

    @torch.no_grad()
    def extract(self, audio_batch):
        """
        audio_batch: (B, T_samples) float tensor at 16kHz
        Returns:
            hidden_states: (B, 24, 1024, T_frames) all layer outputs
            layer6: (B, 1024, T_frames) layer 6 output (vocoder target)
        """
        # WavLM expects normalized input
        outputs = self.model(
            audio_batch.to(self.device),
            output_hidden_states=True,
        )
        # hidden_states: tuple of 25 tensors (B, T_frames, 1024) [CNN + 24 layers]
        # Skip CNN output (index 0), use transformer layers 1-24
        all_layers = torch.stack(
            outputs.hidden_states[1:], dim=1
        )  # (B, 24, T_frames, 1024)
        all_layers = all_layers.permute(0, 1, 3, 2)  # (B, 24, 1024, T_frames)

        layer6 = all_layers[:, WAVLM_LAYER_FOR_VOCODER - 1]  # (B, 1024, T_frames)
        return all_layers, layer6


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class AudioSegmentDataset(Dataset):
    """Loads and segments raw audio for on-the-fly WavLM processing."""
    def __init__(self, wav_dir, label, segment_samples=40000,
                 hop_samples=20000, augment=False):
        self.segments = []
        self.label = label
        self.augment = augment

        wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        for wf in wav_files:
            audio, sr = torchaudio.load(wf)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = audio[0]  # mono

            T = audio.shape[0]
            if T < segment_samples:
                audio = F.pad(audio, (0, segment_samples - T))
                self.segments.append(audio[:segment_samples])
            else:
                for start in range(0, T - segment_samples + 1, hop_samples):
                    self.segments.append(audio[start:start + segment_samples])

        print(f"  Label {label}: {len(self.segments)} segments from {len(wav_files)} files")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        audio = self.segments[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            audio = audio + torch.randn_like(audio) * 0.002
        return audio, torch.tensor(self.label, dtype=torch.float32)


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
                a = next(iter_a)
            except StopIteration:
                iter_a = iter(self.loader_a)
                a = next(iter_a)
            try:
                b = next(iter_b)
            except StopIteration:
                iter_b = iter(self.loader_b)
                b = next(iter_b)
            min_bs = min(a[0].shape[0], b[0].shape[0])
            yield a[0][:min_bs], a[1][:min_bs], b[0][:min_bs], b[1][:min_bs]

    def __len__(self):
        return max(len(self.loader_a), len(self.loader_b))


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── WavLM feature extractor (frozen, on GPU) ───
    wavlm = WavLMFeatureExtractor(device)

    # ─── Datasets (raw audio) ───
    print("\nCreating audio datasets...")
    ds_pre_train = AudioSegmentDataset(PRE_DIR, label=0,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_HOP_SAMPLES, augment=True)
    ds_post_train = AudioSegmentDataset(POST_DIR, label=1,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_HOP_SAMPLES, augment=True)

    loader_pre = DataLoader(ds_pre_train, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, num_workers=2, pin_memory=True)
    loader_post = DataLoader(ds_post_train, batch_size=BATCH_SIZE, shuffle=True,
                             drop_last=True, num_workers=2, pin_memory=True)
    paired_loader = PairedDomainLoader(loader_pre, loader_post)

    # Validation: use non-overlapping segments
    ds_pre_val = AudioSegmentDataset(PRE_DIR, label=0,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES, augment=False)
    ds_post_val = AudioSegmentDataset(POST_DIR, label=1,
        segment_samples=SEGMENT_SAMPLES, hop_samples=SEGMENT_SAMPLES, augment=False)
    from torch.utils.data import ConcatDataset
    val_loader = DataLoader(ConcatDataset([ds_pre_val, ds_post_val]),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                            pin_memory=True)

    # ─── Models ───
    model = AdaptVCModel(
        feat_dim=FEAT_DIM, code_dim=CODE_DIM, num_codes=NUM_CODES,
        num_heads=NUM_HEADS, quality_dim=QUALITY_DIM,
        num_wavlm_layers=wavlm.num_layers,
        commitment_weight=COMMITMENT_WEIGHT, ema_decay=EMA_DECAY,
        entropy_weight=ENTROPY_WEIGHT, dropout=DROPOUT,
        content_noise_std=CONTENT_NOISE,
    ).to(device)
    domain_cls = DomainClassifier1D(code_dim=CODE_DIM).to(device)
    quality_cls = nn.Sequential(
        nn.Linear(QUALITY_DIM, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1),
    ).to(device)

    print(f"AdaptVC parameters: {model.count_parameters():,}")

    # ─── Optimizers ───
    opt_model = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    opt_adv = torch.optim.Adam(domain_cls.parameters(), lr=LR_ADV)
    opt_qual = torch.optim.Adam(quality_cls.parameters(), lr=LR_ADV)

    sched_model = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=EPOCHS, eta_min=1e-6)
    sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adv, T_max=EPOCHS, eta_min=1e-6)

    # ─── Logging ───
    history = {
        'recon_loss': [], 'vq_loss': [], 'adv_loss': [],
        'qual_cls_loss': [], 'cycle_loss': [], 'cross_recon_loss': [],
        'perplexity': [], 'val_recon': [],
    }
    best_val_loss = float('inf')

    # ─── Training loop ───
    for epoch in range(EPOCHS):
        model.train()
        domain_cls.train()
        quality_cls.train()

        ep = {k: 0.0 for k in ['recon', 'vq', 'adv', 'qual', 'cycle', 'cross', 'perp']}
        n_batches = 0

        pbar = tqdm(paired_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for audio_pre, lab_pre, audio_post, lab_post in pbar:
            # Extract WavLM features on-the-fly
            audio_all = torch.cat([audio_pre, audio_post], dim=0).to(device)
            labels = torch.cat([lab_pre, lab_post], dim=0).to(device)
            B_half = audio_pre.shape[0]

            hidden_all, target_all = wavlm.extract(audio_all)

            # ═══ Step 1: Adversarial classifier ═══
            loss_adv_cls = torch.tensor(0.693)
            loss_adv_g = torch.tensor(0.0, device=device)
            loss_qual = torch.tensor(0.0, device=device)
            loss_cross_recon = torch.tensor(0.0, device=device)

            if epoch >= WARMUP_EPOCHS:
                with torch.no_grad():
                    content_q_det, _ = model.encode_content(hidden_all)

                opt_adv.zero_grad()
                adv_pred = domain_cls(content_q_det.detach())
                loss_adv_cls = F.binary_cross_entropy_with_logits(adv_pred.squeeze(1), labels)
                loss_adv_cls.backward()
                opt_adv.step()

            # ═══ Step 2: Main model + quality classifier ═══
            opt_model.zero_grad()
            opt_qual.zero_grad()

            recon, vq_loss, perplexity, content_z, quality = model(hidden_all, target_all)
            loss_recon = F.mse_loss(recon, target_all)

            if epoch >= WARMUP_EPOCHS:
                content_reversed = gradient_reversal(content_z, alpha=LAMBDA_ADV)
                adv_pred_gr = domain_cls(content_reversed)
                loss_adv_g = F.binary_cross_entropy_with_logits(adv_pred_gr.squeeze(1), labels)

                qual_pred = quality_cls(quality)
                loss_qual = F.binary_cross_entropy_with_logits(qual_pred.squeeze(1), labels)

            # ═══ Step 3: Cycle + cross-reconstruction ═══
            loss_cycle = torch.tensor(0.0, device=device)

            if epoch >= WARMUP_EPOCHS:
                hidden_pre = hidden_all[:B_half]
                hidden_post = hidden_all[B_half:]
                target_pre = target_all[:B_half]
                target_post = target_all[B_half:]

                content_q_pre, skips_pre = model.encode_content(hidden_pre)
                quality_pre = model.encode_quality(hidden_pre)

                content_q_post, skips_post = model.encode_content(hidden_post)
                quality_post = model.encode_quality(hidden_post)

                # Cross-reconstruct
                cross_pre2post = model.unet_decoder(content_q_pre, quality_post, skips_pre)
                cross_pre2post = model._match_time(cross_pre2post, target_pre)

                cross_post2pre = model.unet_decoder(content_q_post, quality_pre, skips_post)
                cross_post2pre = model._match_time(cross_post2pre, target_post)

                # Re-encode cross-reconstructed (need to pass through WavLM again?
                # No — we re-encode the OUTPUT features through our encoders directly.
                # This checks cycle consistency in the learned feature space.)
                # We use the reconstructed features as if they were adapter outputs.
                re_enc_a2b, _ = model.unet_encoder(cross_pre2post)
                re_cq_a2b, _, _ = model.vq(model.content_proj(re_enc_a2b))
                # For quality re-encoding, pass through quality encoder
                re_q_a2b = model.quality_encoder(cross_pre2post)

                re_enc_b2a, _ = model.unet_encoder(cross_post2pre)
                re_cq_b2a, _, _ = model.vq(model.content_proj(re_enc_b2a))
                re_q_b2a = model.quality_encoder(cross_post2pre)

                loss_cycle_content = (F.l1_loss(re_cq_a2b, content_q_pre.detach())
                                    + F.l1_loss(re_cq_b2a, content_q_post.detach()))
                loss_cycle_quality = (F.l1_loss(re_q_a2b, quality_post.detach())
                                    + F.l1_loss(re_q_b2a, quality_pre.detach()))
                loss_cycle = loss_cycle_content + loss_cycle_quality

                # Cross-recon quality check
                cross_qp_a2b = quality_cls(re_q_a2b)
                cross_qp_b2a = quality_cls(re_q_b2a)
                loss_cross_recon = (
                    F.binary_cross_entropy_with_logits(
                        cross_qp_a2b.squeeze(1), torch.ones(B_half, device=device))
                    + F.binary_cross_entropy_with_logits(
                        cross_qp_b2a.squeeze(1), torch.zeros(B_half, device=device))
                )

            # ═══ Total loss ═══
            if epoch < WARMUP_EPOCHS:
                loss_total = LAMBDA_RECON * loss_recon + LAMBDA_VQ * vq_loss
            else:
                loss_total = (LAMBDA_RECON * loss_recon
                             + LAMBDA_VQ * vq_loss
                             + loss_adv_g
                             + LAMBDA_QUAL_CLS * loss_qual
                             + LAMBDA_CYCLE * loss_cycle
                             + LAMBDA_CROSS_RECON * loss_cross_recon)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_model.step()
            opt_qual.step()

            ep['recon'] += loss_recon.item()
            ep['vq'] += vq_loss.item()
            ep['adv'] += loss_adv_cls.item()
            ep['qual'] += loss_qual.item()
            ep['cycle'] += loss_cycle.item()
            ep['cross'] += loss_cross_recon.item()
            ep['perp'] += perplexity.item()
            n_batches += 1

            pbar.set_postfix({
                'recon': f'{loss_recon.item():.4f}',
                'perp': f'{perplexity.item():.0f}',
                'adv': f'{loss_adv_cls.item():.3f}',
            })

        for key, ep_key in [('recon_loss', 'recon'), ('vq_loss', 'vq'), ('adv_loss', 'adv'),
                            ('qual_cls_loss', 'qual'), ('cycle_loss', 'cycle'),
                            ('cross_recon_loss', 'cross'), ('perplexity', 'perp')]:
            history[key].append(ep[ep_key] / max(n_batches, 1))

        sched_model.step()
        sched_adv.step()

        # ─── Validation ───
        model.eval()
        val_recon = 0
        n_val = 0
        with torch.no_grad():
            for audio_v, _ in val_loader:
                audio_v = audio_v.to(device)
                hidden_v, target_v = wavlm.extract(audio_v)
                recon_v, _, _, _, _ = model(hidden_v, target_v)
                val_recon += F.mse_loss(recon_v, target_v).item()
                n_val += 1

        avg_val = val_recon / max(n_val, 1)
        history['val_recon'].append(avg_val)

        warmup_tag = " [WARMUP]" if epoch < WARMUP_EPOCHS else ""
        print(f"Epoch {epoch+1}{warmup_tag} | Recon: {history['recon_loss'][-1]:.4f} | "
              f"VQ: {history['vq_loss'][-1]:.4f} | Perp: {history['perplexity'][-1]:.0f}/{NUM_CODES} | "
              f"Cycle: {history['cycle_loss'][-1]:.4f} | Cross: {history['cross_recon_loss'][-1]:.4f} | "
              f"Adv: {history['adv_loss'][-1]:.4f} | Qual: {history['qual_cls_loss'][-1]:.4f} | "
              f"Val Recon: {avg_val:.4f}")

        # Save adapter weights
        adapter_w = model.get_adapter_weights()
        if (epoch + 1) % 50 == 0:
            print(f"  Adapter weights - Content: {adapter_w['content'][:6].round(3)}... "
                  f"Quality: {adapter_w['quality'][:6].round(3)}...")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            avg_quality_pre, avg_quality_post = compute_avg_quality(
                model, wavlm, PRE_DIR, POST_DIR, device)

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'domain_cls': domain_cls.state_dict(),
                'quality_cls': quality_cls.state_dict(),
                'avg_quality_pre': avg_quality_pre,
                'avg_quality_post': avg_quality_post,
                'adapter_weights': adapter_w,
                'val_loss': avg_val,
                'config': {
                    'feat_dim': FEAT_DIM, 'code_dim': CODE_DIM,
                    'num_codes': NUM_CODES, 'num_heads': NUM_HEADS,
                    'quality_dim': QUALITY_DIM,
                    'num_wavlm_layers': wavlm.num_layers,
                },
            }, os.path.join(CHECKPOINT_DIR, 'best_adapt_vc.pth'))
            print(f"  -> Saved best model (val={avg_val:.4f})")

        if (epoch + 1) % 50 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       os.path.join(CHECKPOINT_DIR, f'adapt_vc_epoch{epoch+1}.pth'))

        if (epoch + 1) % 20 == 0:
            plot_training_curves(history, epoch + 1, adapter_w)

    plot_training_curves(history, EPOCHS, model.get_adapter_weights())
    print(f"\nTraining complete. Best val recon loss: {best_val_loss:.4f}")


def compute_avg_quality(model, wavlm, pre_dir, post_dir, device):
    """Compute average quality vectors for each domain."""
    model.eval()

    def _avg_quality_for_dir(wav_dir):
        qualities = []
        for wf in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
            audio, sr = torchaudio.load(wf)
            if sr != SAMPLE_RATE:
                audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            audio = audio[0].unsqueeze(0).to(device)  # (1, T)
            with torch.no_grad():
                hidden, _ = wavlm.extract(audio)
                q = model.encode_quality(hidden)
            qualities.append(q.cpu())
        return torch.cat(qualities, dim=0).mean(dim=0)

    avg_pre = _avg_quality_for_dir(pre_dir)
    avg_post = _avg_quality_for_dir(post_dir)
    print(f"  Avg quality vectors: pre={avg_pre.shape}, post={avg_post.shape}")
    return avg_pre, avg_post


def plot_training_curves(history, epoch, adapter_weights=None):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))

    axes[0, 0].plot(history['recon_loss'], label='Train')
    axes[0, 0].plot(history['val_recon'], label='Val')
    axes[0, 0].set_title('Reconstruction Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['vq_loss'], label='VQ')
    axes[0, 1].set_title('VQ Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[0, 2].plot(history['perplexity'], label='Perplexity')
    axes[0, 2].axhline(y=NUM_CODES, color='r', linestyle='--', label=f'Max ({NUM_CODES})')
    axes[0, 2].set_title(f'Codebook Perplexity ({NUM_HEADS}x{NUM_CODES})')
    axes[0, 2].legend(); axes[0, 2].grid(True)

    axes[0, 3].plot(history['cross_recon_loss'], label='Cross-recon')
    axes[0, 3].set_title('Cross-Recon Quality Loss')
    axes[0, 3].legend(); axes[0, 3].grid(True)

    axes[1, 0].plot(history['cycle_loss'], label='Cycle')
    axes[1, 0].set_title('Cycle Loss'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['adv_loss'], label='Adv')
    axes[1, 1].axhline(y=0.693, color='r', linestyle='--', label='Random')
    axes[1, 1].set_title('Adversarial Loss')
    axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].plot(history['qual_cls_loss'], label='Quality cls')
    axes[1, 2].set_title('Quality Classification')
    axes[1, 2].legend(); axes[1, 2].grid(True)

    # Adapter weight visualization
    if adapter_weights is not None:
        ax = axes[1, 3]
        layers = np.arange(1, len(adapter_weights['content']) + 1)
        ax.bar(layers - 0.2, adapter_weights['content'], 0.4, label='Content', alpha=0.8)
        ax.bar(layers + 0.2, adapter_weights['quality'], 0.4, label='Quality', alpha=0.8)
        ax.set_xlabel('WavLM Layer')
        ax.set_ylabel('Adapter Weight')
        ax.set_title('Learned Layer Weights')
        ax.legend(); ax.grid(True, alpha=0.3)
    else:
        axes[1, 3].axis('off')

    plt.suptitle(f'AdaptVC — Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'training_curves_epoch{epoch}.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    train()
