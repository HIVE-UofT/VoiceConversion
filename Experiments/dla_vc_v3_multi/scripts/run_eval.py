"""
DLA-VC v3 Multi-Surgery — Test Set Evaluation

Loads results_multi_v3/best_model.pth (one DLA-VC model trained on
Tonsill + Fess + Sept) and evaluates each surgery's held-out test patients
separately, then prints a combined summary across all 15 test patients.
Also reports per-surgery train-set similarity as a secondary diagnostic.

Per-patient post-quality is predicted from the test patient's pre audio via
model.predict_post_quality(hidden); no post-surgery audio of the test patient
is used at inference.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --checkpoint results_multi_v3/best_model.pth
"""

import os
import sys
import torch
import torchaudio

SHARED = os.path.join(os.path.dirname(__file__), '..', '..', 'shared')
sys.path.insert(0, SHARED)
from utils import (
    get_wav_files, load_finetuned_knnvc,
    load_ecapa, get_ecapa_embedding, cosine_sim, print_ecapa_summary, SAMPLE_RATE,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.dla_vc import DLAVCModel

# Hardcoded per-surgery test patient IDs (must match train_split.py).
TEST_PATIENTS_BY_SURGERY = {
    "Tonsill": ["0045", "0085", "0110", "0122", "0132"],
    "Sept":    ["0023", "0033", "0044", "0076", "0077"],
    "Fess":    ["0030", "0046", "0086", "0117", "0123"],
}

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'converted_test')
CKPT    = os.path.join(os.path.dirname(__file__), '..', 'results_multi_v3', 'best_model.pth')

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


def convert_and_score(model, wavlm, knn_vc, ecapa, pid, pre_path, post_path,
                      device, out_dir, tag, has_q_shift, avg_q_post):
    """Run DLA-VC conversion (per-patient predicted post quality via q_shift,
    falls back to avg_q_post if q_shift not trained) and compute
    (sim_conv->post, sim_pre->post)."""
    audio = load_audio(pre_path, device)                  # (1, T)
    hidden, _ = wavlm.extract(audio)                      # (1, L, 1024, T')

    with torch.no_grad():
        if has_q_shift:
            target_q = model.predict_post_quality(hidden) # (1, Q)
        else:
            target_q = avg_q_post                         # (1, Q)
        converted = model.convert(hidden, target_q)       # (1, 1024, T')

    out_wav = knn_vc.vocode(converted.squeeze(0).t()[None]).cpu().squeeze()
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

    # Load fine-tuned kNN-VC (WavLM encoder + fine-tuned/stock HiFi-GAN)
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
        use_residual_output=cfg.get('use_residual_output', False),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # avg_q_post kept as a fallback — per-patient q_shift-predicted style is preferred
    avg_q_post = ckpt['avg_quality_post'].to(device)
    if avg_q_post.dim() == 1:
        avg_q_post = avg_q_post.unsqueeze(0)
    has_q_shift = hasattr(model, 'q_shift') and any(
        p.requires_grad or p.abs().sum() > 0 for p in model.q_shift.parameters())
    print(f'[DLA-VC-Multi] Loaded: epoch={ckpt.get("epoch","?")}  '
          f'avg_quality_post shape={avg_q_post.shape}  '
          f'q_shift available={has_q_shift}')

    # DLA uses its own WavLM extractor (all-layer hidden states)
    wavlm = WavLMFeatureExtractor(device)

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
                model, wavlm, knn_vc, ecapa, pid, pre_map[pid], post_map[pid],
                device, test_dir, tag=f'{surg.lower()}_dlavc_multi',
                has_q_shift=has_q_shift, avg_q_post=avg_q_post)
            pids.append(pid)
            sims_conv.append(sc)
            sims_base.append(sb)
            print(f"  [TEST/{surg}] {pid}: baseline={sb:.4f}  conv={sc:.4f}  delta={sc-sb:+.4f}")

        print_ecapa_summary(f'DLA-VC-Multi — TEST / {surg}',
                            pids, sims_conv, sims_base)
        all_pids.extend([f'{surg}:{p}' for p in pids])
        all_conv.extend(sims_conv)
        all_base.extend(sims_base)

    print_ecapa_summary('DLA-VC-Multi — TEST / COMBINED (all surgeries)',
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
                model, wavlm, knn_vc, ecapa, pid, train_pre[pid], train_post[pid],
                device, train_dir, tag=f'{surg.lower()}_train_dlavc_multi',
                has_q_shift=has_q_shift, avg_q_post=avg_q_post)
            tr_pids.append(pid)
            tr_sims_conv.append(sc)
            tr_sims_base.append(sb)

        print_ecapa_summary(f'DLA-VC-Multi — TRAIN / {surg}',
                            tr_pids, tr_sims_conv, tr_sims_base)

    print(f'\nConverted files saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
