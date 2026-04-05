"""Quick comparison of speaker similarity (ECAPA-TDNN) across all methods."""
import os, glob, numpy as np, librosa
from pathlib import Path
from shared_evaluate import compute_speaker_similarity

DATA_ROOT = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"
SR = 16000

# All 4 conditions with their pre/post directory names
CONDITIONS = {
    'Contr':   {'pre': 'Contr/Speech/1',   'post': 'Contr/Speech/2',   'ses_rename': True},
    'Fess':    {'pre': 'Fess/Speech/1',    'post': 'Fess/Speech/2',    'ses_rename': True},
    'Sept':    {'pre': 'Sept/Speech/1',    'post': 'Sept/Speech/2',    'ses_rename': False},
    'Tonsill': {'pre': 'Tonsill/Speech/1', 'post': 'Tonsill/Speech/2', 'ses_rename': True},
}

METHODS = {
    'kNN-VC': 'knn_vc/knn_vc_converted',
    # 'Mean-Shift': 'mean_shift/converted',
    # 'MKL-VC': 'mkl_vc/converted',
    # 'VQVAE-WavLM (Exp5)': 'vqvae/converted_exp5',
    # 'VQVAE-WavLM (Exp6)': 'vqvae/converted_exp6',
    'UNet-VC': 'unet_vc/converted',
    'UNet-ADV-VC': 'unet_adv_vc/converted',
    # 'DLA-VC': 'dla_vc/converted',
}

BASE = os.path.dirname(os.path.abspath(__file__))

# Compute baseline (pre vs post) for each condition
baseline_results = {}
for cond_name, cond_info in CONDITIONS.items():
    pre_dir = os.path.join(DATA_ROOT, cond_info['pre'])
    post_dir = os.path.join(DATA_ROOT, cond_info['post'])

    if not os.path.isdir(pre_dir) or not os.path.isdir(post_dir):
        print(f"SKIP {cond_name}: directories not found")
        continue

    print(f"Computing baseline for {cond_name} (pre vs post)...")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    sims = []
    for pf in pre_files:
        name = Path(pf).stem
        if cond_info['ses_rename']:
            post_name = name.replace('ses1', 'ses2')
        else:
            post_name = name  # Sept: same filename in both folders
        post_path = os.path.join(post_dir, post_name + '.wav')
        if not os.path.exists(post_path):
            continue
        y_pre, _ = librosa.load(pf, sr=SR)
        y_post, _ = librosa.load(post_path, sr=SR)
        sim = compute_speaker_similarity(y_pre, y_post, SR)
        sims.append(sim)
        print(f"  {name}: {sim:.3f}")

    if sims:
        baseline_results[cond_name] = sims
        print(f"  Baseline {cond_name}: {np.mean(sims):.3f} +/- {np.std(sims):.3f}\n")
    else:
        print(f"  No matching pairs found for {cond_name}\n")



CONDITIONS = {
    'Tonsill': {'pre': 'Tonsill/Speech/1', 'post': 'Tonsill/Speech/2', 'ses_rename': True}}
# Compute post vs post baseline (post vs converted) for each method & condition
post_vs_conv_results = {}  # {method_name: {cond_name: [sims]}}
for method_name, method_rel in METHODS.items():
    method_dir = os.path.join(BASE, method_rel)
    if not os.path.isdir(method_dir):
        print(f"SKIP {method_name}: {method_dir} not found")
        continue

    post_vs_conv_results[method_name] = {}
    for cond_name, cond_info in CONDITIONS.items():
        post_dir = os.path.join(DATA_ROOT, cond_info['post'])
        if not os.path.isdir(post_dir):
            continue

        # Converted files use ses1 naming; find matching post (ses2) files
        conv_files = sorted(glob.glob(os.path.join(method_dir, f"{cond_name}_*.wav")))
        if not conv_files:
            continue

        print(f"Computing post vs converted for {method_name} / {cond_name}...")
        sims = []
        for cf in conv_files:
            name = Path(cf).stem
            if cond_info['ses_rename']:
                post_name = name.replace('ses1', 'ses2')
            else:
                post_name = name  # Sept: same filename
            post_path = os.path.join(post_dir, post_name + '.wav')
            if not os.path.exists(post_path):
                continue
            y_conv, _ = librosa.load(cf, sr=SR)
            y_post, _ = librosa.load(post_path, sr=SR)
            sim = compute_speaker_similarity(y_conv, y_post, SR)
            sims.append(sim)
            print(f"  {name}: {sim:.3f}")

        if sims:
            post_vs_conv_results[method_name][cond_name] = sims
            print(f"  {method_name} {cond_name} post-vs-conv: {np.mean(sims):.3f} +/- {np.std(sims):.3f}\n")

# Final baseline comparison table
print(f"\n{'='*60}")
print(f"  Baseline Speaker Similarity (ECAPA-TDNN) — Pre vs Post")
print(f"  Higher = more similar voice identity across sessions")
print(f"{'='*60}")
for cond_name, sims in baseline_results.items():
    print(f"  {cond_name:<30s}  {np.mean(sims):.3f} +/- {np.std(sims):.3f}  (n={len(sims)})")
print(f"{'='*60}")

# Post vs Converted comparison table
print(f"\n{'='*70}")
print(f"  Post vs Converted Speaker Similarity (ECAPA-TDNN)")
print(f"  Higher = converted voice still sounds like post-surgery voice")
print(f"{'='*70}")
cond_names = list(CONDITIONS.keys())
header = f"  {'Method':<25s}" + "".join(f"  {c:<12s}" for c in cond_names)
print(header)
print(f"  {'-'*65}")
for method_name in METHODS:
    row = f"  {method_name:<25s}"
    for cond_name in cond_names:
        sims = post_vs_conv_results.get(method_name, {}).get(cond_name)
        if sims:
            row += f"  {np.mean(sims):.3f}+/-{np.std(sims):.3f}"
        else:
            row += f"  {'n/a':<12s}"
    print(row)
print(f"{'='*70}")
