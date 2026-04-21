"""
Simplified FreeVC fine-tuning for surgical voice conversion.

Starts from the pretrained FreeVC generator (VCTK) and fine-tunes the generator
only (no discriminator, no SR augmentation) on paired CUCO data:

  Input:  content = WavLM features of PRE-surgery audio (same patient)
          speaker emb = SpeakerEncoder(POST-surgery audio)
          target spectrogram = spec(POST-surgery audio)
  Target: generator should reconstruct POST-surgery audio

This teaches the generator to map (pre_content, post_style) -> post_audio,
which is exactly the conversion direction we want at inference time.

Loss: mel L1 + (optional) KL from posterior encoder (VITS-style).
No adversarial, no feature matching — keeps the loop simple and fast.

Output: checkpoints/freevc_cuco_finetuned.pth (generator weights).
"""
import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
from pathlib import Path

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEVC_DIR = os.path.join(EXP_DIR, 'FreeVC')
sys.path.insert(0, FREEVC_DIR)
os.chdir(FREEVC_DIR)

import utils
import commons
from models import SynthesizerTrn
from mel_processing import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
from losses import kl_loss
from speaker_encoder.voice_encoder import SpeakerEncoder

TEST_PATIENTS = {'0045', '0085', '0110', '0122', '0132'}
CUCO_BASE = '/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios'
SR = 16000


def collect_pairs(surgery='Tonsill'):
    def one(sess):
        pattern = os.path.join(CUCO_BASE, surgery, 'Speech', sess, '*.wav')
        return {Path(p).stem.split('_')[-1]: p
                for p in sorted(glob.glob(pattern))}
    pre, post = one('1'), one('2')
    return {pid: {'pre': pre[pid], 'post': post[pid]}
            for pid in pre if pid in post}


def load_wav(path, device):
    sig, sr = torchaudio.load(path)
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != SR:
        sig = torchaudio.functional.resample(sig, sr, SR)
    return sig.to(device)  # (1, T)


class PairedFinetuneDataset(torch.utils.data.Dataset):
    """For each (pre, post) pair, pre-extract content (WavLM) + spec + speaker emb.
    Returns random SEGMENT_LEN-frame crops of content + corresponding audio/spec."""

    def __init__(self, pairs_list, hps, cmodel, smodel, device,
                 segment_samples=32000):
        self.hps = hps
        self.segment_samples = segment_samples
        self.hop = hps.data.hop_length
        self.segment_frames = segment_samples // self.hop
        self.items = []

        print(f'Precomputing features for {len(pairs_list)} pairs...')
        for i, (pid, pre_path, post_path) in enumerate(pairs_list):
            # Pre content via WavLM
            pre_wav = load_wav(pre_path, device)
            with torch.no_grad():
                c_pre = cmodel.extract_features(pre_wav)[0].transpose(1, 2)  # (1, 1024, T')

            # Post audio + spec + speaker emb
            post_wav = load_wav(post_path, device)
            post_wav_np = post_wav.squeeze().cpu().numpy()
            with torch.no_grad():
                spk_post = smodel.embed_utterance(post_wav_np)
                c_post = cmodel.extract_features(post_wav)[0].transpose(1, 2)
            # post_wav is already float in [-1, 1] from torchaudio.load
            spec_post = spectrogram_torch(
                post_wav.cpu(),
                hps.data.filter_length, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length, center=False,
            ).squeeze(0)  # (n_freq, T)

            self.items.append({
                'pid': pid,
                'c_pre':   c_pre.squeeze(0).cpu(),   # (1024, T_pre)
                'c_post':  c_post.squeeze(0).cpu(),  # (1024, T_post)
                'spec':    spec_post,                # (n_freq, T)
                'wav':     post_wav.cpu(),           # (1, T)
                'spk':     torch.from_numpy(spk_post).float(),  # (256,)
            })
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(pairs_list)}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        # Pair content to spec length; random crop
        c = d['c_post']      # content for reconstruction target (matches wav)
        spec = d['spec']
        wav = d['wav']
        T_spec = spec.shape[1]
        T_c = c.shape[1]
        T = min(T_spec, T_c)
        spec, c = spec[:, :T], c[:, :T]
        wav = wav[:, :T * self.hop]

        if T <= self.segment_frames:
            # pad
            pad_T = self.segment_frames - T
            spec = F.pad(spec, (0, pad_T))
            c = F.pad(c, (0, pad_T))
            wav = F.pad(wav, (0, pad_T * self.hop))
        else:
            s = random.randint(0, T - self.segment_frames)
            spec = spec[:, s:s + self.segment_frames]
            c = c[:, s:s + self.segment_frames]
            wav = wav[:, s * self.hop:(s + self.segment_frames) * self.hop]

        return c, spec, wav.squeeze(0), d['spk']


def collate_fn(batch):
    c, spec, wav, spk = zip(*batch)
    c = torch.stack(c)
    spec = torch.stack(spec)
    wav = torch.stack(wav)
    spk = torch.stack(spk)
    # lengths (all same after cropping)
    c_len = torch.full((len(batch),), c.shape[-1], dtype=torch.long)
    spec_len = torch.full((len(batch),), spec.shape[-1], dtype=torch.long)
    return c, spec, spec_len, wav.unsqueeze(1), spk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--surgery', type=str, default='Tonsill')
    ap.add_argument('--config', type=str, default='configs/freevc.json')
    ap.add_argument('--pretrained', type=str,
                    default=os.path.join(EXP_DIR, 'checkpoints/freevc.pth'))
    ap.add_argument('--out',        type=str,
                    default=os.path.join(EXP_DIR, 'checkpoints/freevc_finetuned.pth'))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda')
    hps = utils.get_hparams_from_file(args.config)

    # Load model, restore from pretrained FreeVC
    print('Loading pretrained FreeVC...')
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    _ = utils.load_checkpoint(args.pretrained, net_g, None, True)
    net_g.train()

    # Feature extractors (frozen)
    print('Loading WavLM content encoder...')
    cmodel = utils.get_cmodel(0)
    print('Loading speaker encoder...')
    smodel = SpeakerEncoder('speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    # Data: (pre, post) pairs from training patients
    pairs = collect_pairs(args.surgery)
    train_pairs = sorted(
        (pid, pairs[pid]['pre'], pairs[pid]['post'])
        for pid in pairs if pid not in TEST_PATIENTS
    )
    print(f'Training on {len(train_pairs)} paired (pre, post) utterances.')
    ds = PairedFinetuneDataset(train_pairs, hps, cmodel, smodel, device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True, collate_fn=collate_fn,
    )

    opt = torch.optim.AdamW(net_g.parameters(), lr=args.lr, weight_decay=1e-3)

    print(f'\nFine-tuning for {args.epochs} epochs...')
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        epoch_mel, epoch_kl, n_batches = 0.0, 0.0, 0
        for c, spec, spec_len, wav, spk in loader:
            c, spec, spec_len, wav, spk = c.to(device), spec.to(device), spec_len.to(device), wav.to(device), spk.to(device)

            # VITS forward: content + posterior (from spec) + spk → reconstructed segment
            # Returns: (y_hat, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q))
            y_hat, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = \
                net_g(c, spec, g=spk)

            # Target mel: slice of target audio matching generator output
            y_slice = commons.slice_segments(wav, ids_slice * hps.data.hop_length,
                                              hps.train.segment_size)
            y_mel = mel_spectrogram_torch(
                y_slice.squeeze(1), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax,
            )
            loss_mel = F.l1_loss(y_hat_mel, y_mel) * hps.train.c_mel

            # KL posterior→prior (FreeVC's kl_loss)
            kl = kl_loss(z_p, logs_q, m_p, logs_p, spec_mask) * hps.train.c_kl

            loss = loss_mel + kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 5.0)
            opt.step()

            epoch_mel += loss_mel.item(); epoch_kl += kl.item(); n_batches += 1
            step += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f'Epoch {epoch+1:3d}/{args.epochs} | '
                  f'mel={epoch_mel/max(n_batches,1):.4f} | '
                  f'kl={epoch_kl/max(n_batches,1):.4f} | '
                  f'step={step} | {elapsed/60:.1f} min')

    # Save fine-tuned generator (FreeVC checkpoint format so utils.load_checkpoint works)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        'model': net_g.state_dict(),
        'iteration': step,
        'optimizer': opt.state_dict(),
        'learning_rate': args.lr,
    }, args.out)
    print(f'\nSaved fine-tuned checkpoint: {args.out}')

    # Also save an avg_post_spk_emb for inference convenience
    print('Computing avg post speaker embedding...')
    post_embs = []
    for item in ds.items:
        post_embs.append(item['spk'].numpy())
    avg_spk = np.mean(np.stack(post_embs), axis=0)
    avg_spk = avg_spk / (np.linalg.norm(avg_spk) + 1e-8)
    np.save(args.out.replace('.pth', '_avg_spk.npy'), avg_spk)
    print(f'Saved avg speaker embedding: {args.out.replace(".pth", "_avg_spk.npy")}')


if __name__ == '__main__':
    main()
