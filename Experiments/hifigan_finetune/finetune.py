"""
Fine-tune HiFi-GAN vocoder on post-surgery speech using WavLM-Large features.

Loads pretrained HiFi-GAN from bshall/knn-vc, fine-tunes generator + MPD + MSD
on (WavLM layer-6 features → waveform) reconstruction pairs.

Usage:
    python finetune.py --data_dir /path/to/CUCO/data_final/Audios --out_dir ./output
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the cached knn-vc hub repo to path for hifigan imports
HUB_DIR = os.path.join(torch.hub.get_dir(), "bshall_knn-vc_master")
sys.path.insert(0, HUB_DIR)

from hifigan.models import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from hifigan.meldataset import mel_spectrogram
from hifigan.utils import AttrDict

from dataset import WavLMFeatureExtractor, HiFiGANDataset, collect_wav_paths


def load_pretrained_generator(device):
    """Load pretrained HiFi-GAN generator from bshall/knn-vc."""
    config_path = os.path.join(HUB_DIR, "hifigan", "config_v1_wavlm.json")
    with open(config_path) as f:
        h = AttrDict(json.loads(f.read()))

    generator = Generator(h).to(device)

    # Load pretrained weights (prematched version)
    url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
    generator.load_state_dict(state_dict["generator"])
    print(f"[Generator] Loaded with {sum(p.numel() for p in generator.parameters()):,d} parameters")
    return generator, h


def compute_mel_loss(y, y_hat, h):
    """L1 mel-spectrogram reconstruction loss."""
    if y.dim() == 3:
        y = y.squeeze(1)
    if y_hat.dim() == 3:
        y_hat = y_hat.squeeze(1)
    mel_y = mel_spectrogram(
        y, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
    )
    mel_y_hat = mel_spectrogram(
        y_hat, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss
    )
    return F.l1_loss(mel_y, mel_y_hat), mel_y, mel_y_hat


@torch.no_grad()
def validate(generator, val_loader, h, device):
    """Compute average mel reconstruction loss on validation set."""
    generator.eval()
    total_loss = 0.0
    n = 0
    for features, wav in val_loader:
        features = features.to(device)
        wav = wav.to(device)
        wav_hat = generator(features)
        # Trim to match lengths
        min_len = min(wav.size(-1), wav_hat.size(-1))
        mel_loss, _, _ = compute_mel_loss(wav[:, :, :min_len], wav_hat[:, :, :min_len], h)
        total_loss += mel_loss.item() * features.size(0)
        n += features.size(0)
    generator.train()
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HiFi-GAN on post-surgery speech")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CUCO Audios dir (contains Tonsill/, Fess/, etc.)")
    parser.add_argument("--surgery", type=str, default="Tonsill",
                        help="Surgery subdirectory to use for fine-tuning (default: Tonsill)")
    parser.add_argument("--exclude_patients", type=str,
                        default="0085,0110,0122,0132,0045",
                        help="Comma-separated patient IDs to exclude (held-out test set)")
    parser.add_argument("--out_dir", type=str, default="./output",
                        help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_interval", type=int, default=500,
                        help="Validate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exclude_patients = [p.strip() for p in args.exclude_patients.split(",") if p.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Collect data ---
    print(f"Surgery type: {args.surgery}")
    print(f"Excluding test patients: {exclude_patients}")
    wav_paths = collect_wav_paths(args.data_dir, surgery=args.surgery,
                                  exclude_patients=exclude_patients)
    print(f"Found {len(wav_paths)} post-surgery wav files after exclusions")
    assert len(wav_paths) > 0, (
        f"No wav files found under {args.data_dir}/{args.surgery}/Speech/2/ "
        f"(after excluding {exclude_patients})"
    )

    random.shuffle(wav_paths)
    split = int(0.9 * len(wav_paths))
    train_paths = wav_paths[:split]
    val_paths = wav_paths[split:]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # --- Extract/cache WavLM features ---
    print("Initializing WavLM-Large feature extractor...")
    wavlm = WavLMFeatureExtractor(device=device)

    # Pre-extract all features so we don't re-run WavLM during training
    print("Pre-extracting WavLM features (cached to disk)...")
    for p in tqdm(wav_paths, desc="Extracting features"):
        cache_path = p.with_suffix(".wavlm_l6.pt")
        if not cache_path.exists():
            wav, sr = __import__("torchaudio").load(p)
            if sr != 16000:
                wav = __import__("torchaudio").functional.resample(wav, sr, 16000)
            features = wavlm.extract(wav)
            torch.save(features, cache_path)

    # Free WavLM GPU memory — no longer needed during training
    del wavlm
    torch.cuda.empty_cache()

    # Create datasets with a dummy extractor (features already cached)
    wavlm_dummy = WavLMFeatureExtractor(device="cpu")
    train_ds = HiFiGANDataset(train_paths, wavlm_dummy, segment_size=7040, split=True)
    val_ds = HiFiGANDataset(val_paths, wavlm_dummy, segment_size=7040, split=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # --- Load models ---
    print("Loading pretrained HiFi-GAN generator...")
    generator, h = load_pretrained_generator(device)
    generator.train()

    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    # --- Optimizers ---
    optim_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()), lr=args.lr, betas=(0.8, 0.99)
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    step = 0
    epoch = 0

    print(f"\nStarting fine-tuning for {args.steps} steps...")
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        epoch += 1
        for features, wav in train_loader:
            if step >= args.steps:
                break

            features = features.to(device)  # (B, T', 1024)
            wav = wav.to(device)             # (B, 1, T)

            # --- Generator forward ---
            wav_hat = generator(features)    # (B, 1, T')

            # Align lengths
            min_len = min(wav.size(-1), wav_hat.size(-1))
            wav = wav[:, :, :min_len]
            wav_hat_det = wav_hat[:, :, :min_len]

            # --- Discriminator step ---
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(wav, wav_hat_det.detach())
            loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(wav, wav_hat_det.detach())
            loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_d = loss_disc_f + loss_disc_s
            loss_d.backward()
            optim_d.step()

            # --- Generator step ---
            optim_g.zero_grad()

            mel_loss, _, _ = compute_mel_loss(wav, wav_hat[:, :, :min_len], h)

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(wav, wav_hat[:, :, :min_len])
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(wav, wav_hat[:, :, :min_len])

            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _ = generator_loss(y_df_hat_g)
            loss_gen_s, _ = generator_loss(y_ds_hat_g)

            loss_g = loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + mel_loss * 45

            loss_g.backward()
            optim_g.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(
                mel=f"{mel_loss.item():.4f}",
                g=f"{loss_g.item():.2f}",
                d=f"{loss_d.item():.2f}",
                epoch=epoch,
            )

            # --- Validation ---
            if step % args.val_interval == 0:
                val_loss = validate(generator, val_loader, h, device)
                print(f"\n[Step {step}] Val mel loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = os.path.join(args.out_dir, "best_generator.pt")
                    torch.save(
                        {"generator": generator.state_dict(), "step": step, "val_mel_loss": val_loss},
                        ckpt_path,
                    )
                    print(f"  -> Saved best checkpoint to {ckpt_path}")

    pbar.close()

    # Save final checkpoint
    final_path = os.path.join(args.out_dir, "final_generator.pt")
    torch.save(
        {"generator": generator.state_dict(), "step": step, "val_mel_loss": best_val_loss},
        final_path,
    )
    print(f"\nDone! Final checkpoint: {final_path}")
    print(f"Best val mel loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
