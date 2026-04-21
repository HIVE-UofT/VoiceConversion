"""
DLA-VC — Test Set Evaluation with Fine-Tuned HiFi-GAN

Proper inference: content from test pre-surgery, quality from the MEAN
post-surgery quality computed over TRAINING patients only (stored in checkpoint).
No post-surgery audio of the test patient is used.

1. Loads best_model.pt from results_tonsill_split/.
2. Converts each test patient using avg_quality_post from the checkpoint.
3. Synthesises with the fine-tuned HiFi-GAN.
4. Evaluates ECAPA-TDNN: converted→post vs baseline pre→post.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --checkpoint ../results_tonsill_split/best_model.pt
"""

import os
import sys
import torch
import torchaudio

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.dla_vc import DLAVCModel

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')
CKPT    = os.path.join(os.path.dirname(__file__), '..', 'results_tonsill_split', 'best_model.pth')

WAVLM_LAYER_FOR_VOCODER = 6


class WavLMFeatureExtractor:
    def __init__(self, device):
        from transformers import WavLMModel
        print("Loading WavLM-Large...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.num_layers = self.model.config.num_hidden_layers

    @torch.no_grad()
    def extract(self, audio_batch):
        outputs = self.model(audio_batch.to(self.device), output_hidden_states=True)
        all_layers = torch.stack(outputs.hidden_states[1:], dim=1).permute(0, 1, 3, 2)
        layer6 = all_layers[:, WAVLM_LAYER_FOR_VOCODER - 1]
        return all_layers, layer6


def load_audio(path, device):
    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav[0].unsqueeze(0).to(device)   # (1, T)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CKPT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load fine-tuned kNN-VC (WavLM encoder + fine-tuned HiFi-GAN)
    knn_vc = load_finetuned_knnvc(device)

    # Load DLA checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt['config']

    model = DLAVCModel(
        feat_dim=cfg['feat_dim'],
        code_dim=cfg['code_dim'],
        num_codes=cfg['num_codes'],
        num_heads=cfg['num_heads'],
        quality_dim=cfg['quality_dim'],
        num_wavlm_layers=cfg['num_wavlm_layers'],
        commitment_weight=cfg.get('commitment_weight', 0.25),
        ema_decay=cfg.get('ema_decay', 0.99),
        entropy_weight=cfg.get('entropy_weight', 0.5),
        dropout=0.0,
        content_noise_std=0.0,
        use_vq=cfg.get('use_vq', False),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # avg_q_post kept as a fallback — per-patient q_shift-predicted style is preferred
    avg_q_post = ckpt['avg_quality_post'].to(device)
    if avg_q_post.dim() == 1:
        avg_q_post = avg_q_post.unsqueeze(0)
    has_q_shift = hasattr(model, 'q_shift') and any(
        p.requires_grad or p.abs().sum() > 0 for p in model.q_shift.parameters())
    print(f'[DLA-VC] Loaded: epoch={ckpt.get("epoch","?")}  '
          f'avg_quality_post shape={avg_q_post.shape}  '
          f'q_shift available={has_q_shift}')

    # DLA uses its own WavLM extractor (all-layer hidden states)
    wavlm = WavLMFeatureExtractor(device)

    test_pre  = {pid: p for pid, p in
                 get_wav_files(surgery='Tonsill', session='1').items()
                 if pid in TEST_PATIENTS}
    test_post = {pid: p for pid, p in
                 get_wav_files(surgery='Tonsill', session='2').items()
                 if pid in TEST_PATIENTS}

    print(f'\nEvaluating on {len(test_pre)} test patients: {sorted(test_pre)}')

    print('\nLoading ECAPA-TDNN...')
    ecapa = load_ecapa(device)

    pids, sims_conv, sims_base = [], [], []

    for pid in sorted(test_pre):
        pre_path  = test_pre[pid]
        post_path = test_post[pid]
        out_path  = os.path.join(OUT_DIR, f'{pid}_dlavc.wav')

        # Extract all-layer WavLM features from pre-surgery audio
        audio = load_audio(pre_path, device)                      # (1, T)
        hidden, _ = wavlm.extract(audio)                          # (1, L, 1024, T')

        # Convert: content from pre, quality = per-patient predicted post style
        # via q_shift(encode_quality(pre)). Falls back to avg_q_post if q_shift
        # hasn't been trained (older checkpoints).
        with torch.no_grad():
            if has_q_shift:
                target_q = model.predict_post_quality(hidden)     # (1, Q)
            else:
                target_q = avg_q_post                             # (1, Q)
            converted = model.convert(hidden, target_q)           # (1, 1024, T')

        # Synthesise with fine-tuned HiFi-GAN
        out_wav = knn_vc.vocode(converted.squeeze(0).t()[None]).cpu().squeeze()
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary('DLA-VC', pids, sims_conv, sims_base)

    # ── Training patients evaluation ──────────────────────────────────────────────
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        audio = load_audio(pre_path, device)
        hidden, _ = wavlm.extract(audio)
        with torch.no_grad():
            converted = model.convert(hidden, avg_q_post)
        out_wav = knn_vc.vocode(converted.squeeze(0).t()[None]).cpu().squeeze()

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("DLA-VC [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f'\nConverted files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
