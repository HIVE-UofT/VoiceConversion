"""
UNet-VC-ECAPA Multi-Surgery — Test Set Evaluation

Loads results_multi_v2/best_model.pt (one model trained on Tonsill+Fess+Sept)
and evaluates each surgery's held-out test patients separately, then prints
a combined summary across all 15 test patients. Also reports per-surgery
train-set similarity as a secondary diagnostic.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --checkpoint results_multi_v2/best_model.pt
"""

import os
import sys
import torch
import torchaudio
import numpy as np

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
    ECAPA_SAVEDIR,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.unet import ResUNet1D

# Hardcoded per-surgery test patient IDs (must match train_split_v2.py).
TEST_PATIENTS_BY_SURGERY = {
    "Tonsill": ["0045", "0085", "0110", "0122", "0132"],
    "Sept":    ["0023", "0033", "0044", "0076", "0077"],
    "Fess":    ["0030", "0046", "0086", "0117", "0123"],
}

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')
CKPT    = os.path.join(os.path.dirname(__file__), '..', 'results_multi_v2', 'best_model.pt')

HIDDEN_DIM = 64
N_LEVELS   = 2


def ecapa_embed(ecapa, wav_or_path, device):
    """Return (1, 192) ECAPA embedding for use as FiLM conditioning."""
    if isinstance(wav_or_path, (str, os.PathLike)):
        wav, sr = torchaudio.load(str(wav_or_path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    else:
        wav = wav_or_path
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav.to(device)).squeeze()
    return emb.unsqueeze(0)


def convert_and_score(model, knn_vc, ecapa, pid, pre_path, post_path, device,
                      out_dir, tag):
    """Run UNet conversion and compute (sim_conv->post, sim_pre->post)."""
    features = knn_vc.get_features(str(pre_path))
    spk_emb  = ecapa_embed(ecapa, pre_path, device)
    with torch.no_grad():
        converted = model(features.t().unsqueeze(0), spk_emb).squeeze(0).t()
    out_wav = knn_vc.vocode(converted[None]).cpu().squeeze()
    out_path = os.path.join(out_dir, f'{pid}_{tag}.wav')
    torchaudio.save(out_path, out_wav.unsqueeze(0), SAMPLE_RATE)

    emb_conv = get_ecapa_embedding(ecapa, out_wav.unsqueeze(0), device)
    emb_post = get_ecapa_embedding(ecapa, post_path, device)
    emb_pre  = get_ecapa_embedding(ecapa, pre_path,  device)
    return cosine_sim(emb_conv, emb_post), cosine_sim(emb_pre, emb_post)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CKPT)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUT_DIR, exist_ok=True)

    knn_vc = load_finetuned_knnvc(device)

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
    print(f'[UNet-VC-ECAPA-Multi] Loaded: epoch={ckpt.get("epoch","?")}  '
          f'val_ecapa={ckpt.get("val_ecapa", float("nan")):.4f}  '
          f'alpha={ckpt.get("alpha", float("nan")):.4f}  '
          f'surgeries={ckpt.get("surgeries", "?")}')

    print('\nLoading ECAPA-TDNN...')
    ecapa = load_ecapa(device)

    surgeries = sorted(TEST_PATIENTS_BY_SURGERY.keys())

    # ───────── TEST EVAL (per-surgery + combined) ─────────
    test_dir = os.path.join(OUT_DIR, 'test')
    os.makedirs(test_dir, exist_ok=True)
    all_pids, all_conv, all_base = [], [], []

    print(f"\n{'#'*70}\n#   TEST EVALUATION (per surgery)\n{'#'*70}")
    for surg in surgeries:
        test_ids = set(TEST_PATIENTS_BY_SURGERY[surg])
        pre_map  = {pid: p for pid, p in
                    get_wav_files(surgery=surg, session='1').items()
                    if pid in test_ids}
        post_map = {pid: p for pid, p in
                    get_wav_files(surgery=surg, session='2').items()
                    if pid in test_ids}

        missing = test_ids - (set(pre_map) & set(post_map))
        if missing:
            print(f"  [WARN] {surg}: missing pre/post for {sorted(missing)}")

        common_pids = sorted(set(pre_map) & set(post_map))
        print(f"\n--- {surg}: {len(common_pids)} test patients ---")
        pids, sims_conv, sims_base = [], [], []
        for pid in common_pids:
            sc, sb = convert_and_score(
                model, knn_vc, ecapa, pid, pre_map[pid], post_map[pid],
                device, test_dir, tag=f'{surg.lower()}_unetvcecapa_multi')
            pids.append(pid)
            sims_conv.append(sc)
            sims_base.append(sb)
            print(f"  [TEST/{surg}] {pid}: baseline={sb:.4f}  conv={sc:.4f}  delta={sc-sb:+.4f}")

        print_ecapa_summary(f'UNet-VC-ECAPA-Multi — TEST / {surg}',
                            pids, sims_conv, sims_base)
        all_pids.extend([f'{surg}:{p}' for p in pids])
        all_conv.extend(sims_conv)
        all_base.extend(sims_base)

    print_ecapa_summary('UNet-VC-ECAPA-Multi — TEST / COMBINED (all surgeries)',
                        all_pids, all_conv, all_base)

    # ───────── TRAIN EVAL (per-surgery, secondary) ─────────
    print(f"\n{'#'*70}\n#   TRAIN EVALUATION (per surgery, secondary)\n{'#'*70}")
    train_dir = os.path.join(OUT_DIR, 'train')
    os.makedirs(train_dir, exist_ok=True)

    for surg in surgeries:
        test_ids = set(TEST_PATIENTS_BY_SURGERY[surg])
        train_pre  = get_wav_files(surgery=surg, session='1', exclude=test_ids)
        train_post = get_wav_files(surgery=surg, session='2', exclude=test_ids)

        common_pids = sorted(set(train_pre) & set(train_post))
        print(f"\n--- {surg}: {len(common_pids)} train patients ---")
        tr_pids, tr_sims_conv, tr_sims_base = [], [], []
        for pid in common_pids:
            sc, sb = convert_and_score(
                model, knn_vc, ecapa, pid, train_pre[pid], train_post[pid],
                device, train_dir, tag=f'{surg.lower()}_train_unetvcecapa_multi')
            tr_pids.append(pid)
            tr_sims_conv.append(sc)
            tr_sims_base.append(sb)

        print_ecapa_summary(f'UNet-VC-ECAPA-Multi — TRAIN / {surg}',
                            tr_pids, tr_sims_conv, tr_sims_base)

    print(f'\nConverted files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
