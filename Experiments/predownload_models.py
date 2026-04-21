"""
Run this ONCE on the login node (which has internet access) to cache
all model weights before submitting SLURM jobs.

Usage:
    python predownload_models.py
"""

import torch
import os

print("=" * 50)
print("Pre-caching model weights for offline compute nodes")
print("=" * 50)

# 1. kNN-VC (downloads WavLM-Large.pt ~1.26 GB + HiFi-GAN weights)
print("\n[1/3] Downloading kNN-VC + WavLM-Large weights (torch.hub)...")
knn_vc = torch.hub.load(
    'bshall/knn-vc', 'knn_vc',
    prematched=True,
    device='cpu',
    progress=True,
)
ckpt = os.path.join(torch.hub.get_dir(), 'checkpoints', 'WavLM-Large.pt')
print(f"  WavLM-Large.pt: {os.path.getsize(ckpt)/1e9:.2f} GB  OK")

# 2. HuggingFace WavLM-Large (used by DLA-VC — all hidden layers)
print("\n[2/3] Downloading microsoft/wavlm-large (HuggingFace)...")
from transformers import WavLMModel
hf_wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
print(f"  Cached to: {hf_cache}  OK")
del hf_wavlm

# 3. ECAPA — already at fixed savedir, just verify
print("\n[3/3] Checking ECAPA-TDNN...")
savedir = "/lustre06/project/6086959/sepharfi/pretrained_models/ecapa-voxceleb"
if os.path.exists(os.path.join(savedir, "embedding_model.ckpt")):
    print(f"  Already cached at: {savedir}  OK")
else:
    print(f"  Not found — downloading...")
    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": "cpu"},
    )
    print("  Done.")

print("\nAll weights cached. You can now submit SLURM jobs.")
