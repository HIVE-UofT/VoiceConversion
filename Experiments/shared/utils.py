"""
shared/utils.py — Common utilities for Tonsill VC evaluation.

Single source of truth for:
  - TEST_PATIENTS: the 5 held-out patients, excluded from ALL training and
    statistics computation for every method
  - Fine-tuned HiFi-GAN loader: swaps fine-tuned weights into knn_vc.hifigan
    so all existing knn_vc.vocode() / knn_vc.match() calls use the new vocoder
  - ECAPA evaluation helpers
  - Wav file collection with patient filtering
"""

import os
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────

CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepharfi/CUCO/data_final/Audios"
SAMPLE_RATE = 16000
ECAPA_SAVEDIR = "/lustre06/project/6086959/sepharfi/pretrained_models/ecapa-voxceleb"
HIFIGAN_CKPT = (
    "/home/sepharfi/projects/def-zshakeri/sepharfi/VoiceConversion"
    "/Experiments/hifigan_finetune/output/best_generator.pt"
)

# Fixed test patients — same for every method, every experiment.
# Selected to cover the full distribution of surgery effects (one per quintile):
#   0132: SpkSim=0.390 (strong effect, Q1)
#   0110: SpkSim=0.487 (moderate-strong, Q2)
#   0085: SpkSim=0.535 (near median, Q3)
#   0045: SpkSim=0.594 (mild-moderate, Q4)
#   0122: SpkSim=0.644 (mild, Q5)
# Overall mean of test set = 0.530, matching the 28-patient mean of 0.529.
TEST_PATIENTS = ["0085", "0110", "0122", "0132", "0045"]


# ── Patient / file helpers ─────────────────────────────────────────────────────

def get_patient_id(wav_path):
    """Extract patient ID from filename, e.g. 'Speech_0085.wav' → '0085'."""
    return Path(wav_path).stem.split("_")[-1]


# All audio subtypes in CUCO — (category, subcategory or None)
ALL_AUDIO_SUBDIRS = [
    ("Speech",           None),
    ("TDU",              "Agua"),
    ("TDU",              "Brasero"),
    ("TDU",              "Dia"),
    ("TDU",              "Mesa"),
    ("Vowels",           "A"),
    ("Vowels",           "E"),
    ("Vowels",           "I"),
    ("Vowels",           "O"),
    ("Vowels",           "U"),
    ("Sustained vowels", "A1"),
    ("Sustained vowels", "A2"),
    ("Sustained vowels", "A3"),
]


def get_all_audio_pairs(surgery="Tonsill", session_pre="1", session_post="2",
                        exclude=None):
    """
    Collect all (pre_path, post_path) pairs across ALL audio types
    (Speech, TDU, Vowels, Sustained vowels) for `surgery`, using session
    `session_pre` as source and `session_post` as target.

    Returns {patient_id: [(pre_path, post_path), ...]} — only includes pairs
    where both sessions exist.  Patients in `exclude` are omitted entirely.
    """
    exclude = set(exclude) if exclude else set()
    base = Path(CUCO_BASE) / surgery
    patient_pairs: dict = {}

    for cat, sub in ALL_AUDIO_SUBDIRS:
        type_dir = base / cat
        if sub:
            type_dir = type_dir / sub
        pre_dir  = type_dir / session_pre
        post_dir = type_dir / session_post
        if not pre_dir.exists() or not post_dir.exists():
            continue
        pre_wavs  = {Path(f).stem.split("_")[-1]: Path(f)
                     for f in sorted(pre_dir.glob("*.wav"))}
        post_wavs = {Path(f).stem.split("_")[-1]: Path(f)
                     for f in sorted(post_dir.glob("*.wav"))}
        for pid, pre_path in pre_wavs.items():
            if pid in exclude or pid not in post_wavs:
                continue
            patient_pairs.setdefault(pid, []).append(
                (str(pre_path), str(post_wavs[pid]))
            )

    return patient_pairs


def get_wav_files(surgery="Tonsill", session="2", exclude=None):
    """
    Return {patient_id: Path} for all wav files matching surgery/session,
    excluding any patient IDs in `exclude`.

    session: '1' = pre-surgery, '2' = post-surgery
    """
    exclude = set(exclude) if exclude else set()
    speech_dir = Path(CUCO_BASE) / surgery / "Speech" / session
    if not speech_dir.exists():
        raise FileNotFoundError(f"Directory not found: {speech_dir}")
    result = {}
    for wav_file in sorted(speech_dir.glob("*.wav")):
        pid = get_patient_id(wav_file)
        if pid not in exclude:
            result[pid] = wav_file
    return result


# ── Fine-tuned HiFi-GAN ────────────────────────────────────────────────────────

def _merge_weight_norm(state_dict):
    """
    Convert a weight-norm parameterised state dict (keys: name_g, name_v)
    into a plain state dict (key: name) by computing:

        weight = weight_v * (weight_g / ||weight_v||_per_output_channel)

    This is needed because knn_vc.hifigan has weight norm *removed* at load
    time (single merged .weight keys), but the fine-tuned checkpoint is saved
    while weight norm is still active (split _g / _v keys).
    """
    # Find all weight-norm bases (have both _g and _v counterparts)
    wn_bases = {k[:-2] for k in state_dict if k.endswith('_g')
                and k[:-2] + '_v' in state_dict}

    merged = {}
    for k, val in state_dict.items():
        if k.endswith('_g') and k[:-2] in wn_bases:
            base   = k[:-2]
            g      = state_dict[base + '_g']   # (C_out, 1, 1, ...)
            v      = state_dict[base + '_v']   # (C_out, C_in, K, ...)
            # L2 norm per output channel, broadcast back to v's shape
            norm   = v.view(v.shape[0], -1).norm(dim=1)
            norm   = norm.view(v.shape[0], *([1] * (v.dim() - 1)))
            merged[base] = v * (g / (norm + 1e-8))
        elif k.endswith('_v') and k[:-2] in wn_bases:
            pass  # already handled above
        else:
            merged[k] = val
    return merged


def load_finetuned_knnvc(device, hifigan_ckpt=HIFIGAN_CKPT):
    """
    Load kNN-VC (WavLM encoder + HiFi-GAN) and, if available, swap in the
    CUCO-fine-tuned generator weights. If the fine-tuned checkpoint is
    missing on disk OR the environment variable FORCE_STOCK_VOCODER is set,
    fall back to the stock bshall/knn-vc HiFi-GAN. The override is useful
    for re-evaluating all methods with the same stock vocoder so test
    numbers are comparable across the table.
    """
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc", prematched=True, device=device
    )
    force_stock = os.environ.get("FORCE_STOCK_VOCODER", "").lower() in ("1", "true", "yes")
    if force_stock or not os.path.exists(hifigan_ckpt):
        reason = "FORCE_STOCK_VOCODER env var set" if force_stock \
                 else f"fine-tuned checkpoint not found at {hifigan_ckpt}"
        print(
            f"[HiFi-GAN] Using STOCK bshall/knn-vc HiFi-GAN ({reason}). "
            f"Numbers in this run are not directly comparable to runs that "
            f"used the CUCO-fine-tuned vocoder."
        )
        knn_vc.hifigan.eval()
        return knn_vc
    ckpt       = torch.load(hifigan_ckpt, map_location=device, weights_only=False)
    gen_state  = _merge_weight_norm(ckpt["generator"])
    knn_vc.hifigan.load_state_dict(gen_state)
    knn_vc.hifigan.eval()
    step = ckpt.get("step", "?")
    mel  = ckpt.get("val_mel_loss", float("nan"))
    print(
        f"[HiFi-GAN] Fine-tuned weights loaded  "
        f"step={step}  val_mel_loss={mel:.4f}"
    )
    return knn_vc


# ── ECAPA helpers ──────────────────────────────────────────────────────────────

def load_ecapa(device):
    """Load ECAPA-TDNN speaker encoder from SpeechBrain (cached locally)."""
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=ECAPA_SAVEDIR,
        run_opts={"device": str(device)},
    )


@torch.no_grad()
def get_ecapa_embedding(ecapa, wav_or_path, device):
    """
    Compute ECAPA-TDNN speaker embedding.

    wav_or_path: str/Path to a wav file, or a (T,) / (1, T) waveform tensor
                 at SAMPLE_RATE.
    Returns: (192,) embedding tensor on CPU.
    """
    if isinstance(wav_or_path, (str, Path)):
        wav, sr = torchaudio.load(str(wav_or_path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    else:
        wav = wav_or_path
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
    return ecapa.encode_batch(wav.to(device)).squeeze().cpu()


def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ── Reporting helper ───────────────────────────────────────────────────────────

def print_ecapa_summary(method_name, pids, sims_conv, sims_base):
    """Print per-patient and mean ECAPA results."""
    import numpy as np
    sims_conv = list(sims_conv)
    sims_base = list(sims_base)
    print(f"\n{'='*60}")
    print(f"  {method_name} — ECAPA Evaluation on Test Patients")
    print(f"{'='*60}")
    print(f"  {'Patient':>10}  {'Baseline':>10}  {'Converted':>10}  {'Delta':>8}")
    print(f"  {'-'*46}")
    deltas = []
    for pid, sc, sb in zip(pids, sims_conv, sims_base):
        d = sc - sb
        deltas.append(d)
        print(f"  {pid:>10}  {sb:>10.4f}  {sc:>10.4f}  {d:>+8.4f}")
    print(f"  {'-'*46}")
    print(f"  {'Mean':>10}  {np.mean(sims_base):>10.4f}  {np.mean(sims_conv):>10.4f}"
          f"  {np.mean(deltas):>+8.4f}")
    print(f"  {'Std':>10}  {np.std(sims_base):>10.4f}  {np.std(sims_conv):>10.4f}")
