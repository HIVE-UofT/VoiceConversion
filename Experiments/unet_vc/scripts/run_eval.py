"""
UNet-VC — Test Set Evaluation with Fine-Tuned HiFi-GAN

1. Loads best_model.pt from checkpoints_kfold/.
2. Converts each test patient's pre-surgery recording.
3. Synthesises with the fine-tuned HiFi-GAN.
4. Evaluates ECAPA-TDNN: converted→post vs baseline pre→post.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --checkpoint ../checkpoints_kfold/best_model.pt
"""

import os
import sys
import torch
import torchaudio
import numpy as np

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    TEST_PATIENTS, get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')
CKPT    = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_kfold', 'best_model.pt')

# Match training config
HIDDEN_DIM = 128
N_LEVELS   = 2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CKPT)
    parser.add_argument('--stock_vocoder', action='store_true',
                        help='Skip the CUCO-fine-tuned HiFi-GAN and use the stock '
                             'bshall/knn-vc HiFi-GAN instead. Useful when the '
                             'fine-tuned vocoder over-fits the small training set.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load kNN-VC vocoder: either the CUCO-fine-tuned HiFi-GAN or the stock one.
    if args.stock_vocoder:
        print('[HiFi-GAN] Using STOCK bshall/knn-vc HiFi-GAN (--stock_vocoder).')
        knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, device=device)
    else:
        knn_vc = load_finetuned_knnvc(device)

    # Load trained UNet-VC model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get('config', {})
    model = ResUNet1D(
        feat_dim=cfg.get('feat_dim', 1024),
        hidden_dim=cfg.get('hidden_dim', HIDDEN_DIM),
        n_levels=cfg.get('n_levels', N_LEVELS),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'[UNet-VC] Loaded checkpoint: epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.6f}  '
          f'alpha={ckpt.get("alpha", float("nan")):.4f}')

    # Test patient files
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
        out_path  = os.path.join(OUT_DIR, f'{pid}_unetvc.wav')

        # Extract WavLM features and convert with trained model
        features = knn_vc.get_features(str(pre_path))   # (T, 1024)
        with torch.no_grad():
            converted = model(features.t().unsqueeze(0)).squeeze(0).t()  # (T, 1024)
        out_wav = knn_vc.vocode(converted[None]).cpu().squeeze()         # (T_audio,)
        torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        pids.append(pid)
        sims_conv.append(cosine_sim(emb_conv, emb_post))
        sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary('UNet-VC', pids, sims_conv, sims_base)

    # ── Training patients evaluation ──────────────────────────────────────────────
    train_pre  = get_wav_files(surgery="Tonsill", session="1", exclude=TEST_PATIENTS)
    train_post = get_wav_files(surgery="Tonsill", session="2", exclude=TEST_PATIENTS)

    print(f"\nEvaluating on {len(train_pre)} training patients...")
    tr_pids, tr_sims_conv, tr_sims_base = [], [], []
    for pid in sorted(train_pre):
        pre_path  = train_pre[pid]
        post_path = train_post[pid]

        features = knn_vc.get_features(str(pre_path))
        with torch.no_grad():
            converted = model(features.t().unsqueeze(0)).squeeze(0).t()
        out_wav = knn_vc.vocode(converted[None]).cpu().squeeze()

        emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
        emb_post = get_ecapa_embedding(ecapa, post_path, device)
        emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)

        tr_pids.append(pid)
        tr_sims_conv.append(cosine_sim(emb_conv, emb_post))
        tr_sims_base.append(cosine_sim(emb_pre,  emb_post))

    print_ecapa_summary("UNet-VC [TRAIN SET]", tr_pids, tr_sims_conv, tr_sims_base)
    print(f'\nConverted files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
