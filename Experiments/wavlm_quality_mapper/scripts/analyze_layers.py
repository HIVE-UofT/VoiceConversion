"""
WavLM Layer 12-16 Analysis: Pre vs Post Surgery

Extracts WavLM-Large hidden states from layers 12-16 for pre- and post-surgery
audio, then compares them via:
  1. Per-layer mean/std shift (global statistics)
  2. Per-dimension delta magnitude (which features change most)
  3. Cosine similarity distributions (per-frame, per-patient)
  4. PCA / t-SNE visualization of pre vs post in each layer
  5. CKA similarity between layers (how redundant are layers 12-16?)
  6. Content invariance: same speaker, different content (Speech, Vowels, TDU)
     — measures whether layers 12-16 are stable across content for a given speaker

Outputs plots to ../plots/analysis/ and statistics to ../cache/layer_stats.pt

Usage:
    python scripts/analyze_layers.py
    python scripts/analyze_layers.py --surgery Tonsill --max_patients 10
"""

import argparse
import os
import sys
import glob
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CUCO_BASE = "/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/Audios"
SAMPLE_RATE = 16000
LAYERS = list(range(1, 25))  # All 24 WavLM transformer layers
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots', 'analysis')


# ──────────────────────────────────────────────
# WavLM Multi-Layer Extractor
# ──────────────────────────────────────────────
class WavLMMultiLayerExtractor:
    def __init__(self, device, layers=LAYERS):
        from transformers import WavLMModel
        print("Loading WavLM-Large...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        # Convert 1-indexed layers to 0-indexed into hidden_states[1:]
        self.layer_indices = [l - 1 for l in layers]
        self.layer_names = [f"layer_{l}" for l in layers]
        print(f"  Extracting layers: {layers}")

    @torch.no_grad()
    def extract(self, wav_path):
        """
        Returns dict: {layer_idx: (T, 1024)} for each requested layer.
        """
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).to(self.device)

        outputs = self.model(wav.unsqueeze(0), output_hidden_states=True)
        # hidden_states: tuple of 25 tensors (1, T, 1024) [CNN + 24 transformer]
        result = {}
        for li in self.layer_indices:
            result[li] = outputs.hidden_states[li + 1].squeeze(0).cpu()  # (T, 1024)
        return result


def extract_and_cache(extractor, wav_files, cache_path):
    """Extract features for all files, cache to disk."""
    if os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    all_feats = []
    for wf in wav_files:
        feats = extractor.extract(wf)
        name = Path(wf).stem
        all_feats.append((name, feats))
        print(f"    {name}: {list(feats.values())[0].shape[0]} frames")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(all_feats, cache_path)
    return all_feats


# ──────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────

def compute_global_stats(pre_data, post_data, layer_indices):
    """Per-layer mean, std, and delta for pre vs post."""
    stats = {}
    for li in layer_indices:
        pre_all = torch.cat([feats[li] for _, feats in pre_data], dim=0)
        post_all = torch.cat([feats[li] for _, feats in post_data], dim=0)

        stats[li] = {
            'pre_mean': pre_all.mean(dim=0),
            'pre_std': pre_all.std(dim=0),
            'post_mean': post_all.mean(dim=0),
            'post_std': post_all.std(dim=0),
            'delta_mean': post_all.mean(dim=0) - pre_all.mean(dim=0),
            'delta_std': post_all.std(dim=0) - pre_all.std(dim=0),
            'n_pre': pre_all.shape[0],
            'n_post': post_all.shape[0],
        }
        print(f"  Layer {li+1}: pre={pre_all.shape[0]} frames, post={post_all.shape[0]} frames")
        print(f"    Mean shift L2: {stats[li]['delta_mean'].norm():.4f}")
        print(f"    Mean shift cos: {F.cosine_similarity(stats[li]['pre_mean'].unsqueeze(0), stats[li]['post_mean'].unsqueeze(0)).item():.4f}")
    return stats


def compute_per_patient_similarity(pre_data, post_data, layer_indices):
    """Per-patient cosine similarity between pre and post (frame-averaged)."""
    results = {}
    for li in layer_indices:
        patient_sims = []
        for (name_pre, feats_pre), (name_post, feats_post) in zip(pre_data, post_data):
            pre_f = feats_pre[li]  # (T1, 1024)
            post_f = feats_post[li]  # (T2, 1024)
            # Average over frames
            pre_avg = pre_f.mean(dim=0)
            post_avg = post_f.mean(dim=0)
            sim = F.cosine_similarity(pre_avg.unsqueeze(0), post_avg.unsqueeze(0)).item()
            patient_sims.append({'name': name_pre, 'sim': sim})
        results[li] = patient_sims
        sims = [p['sim'] for p in patient_sims]
        print(f"  Layer {li+1}: mean cos sim = {np.mean(sims):.4f} +/- {np.std(sims):.4f}")
    return results


def compute_frame_similarity_distribution(pre_data, post_data, layer_indices, n_samples=5000):
    """Sample random frame pairs from same patient, compute cosine sim distribution."""
    results = {}
    for li in layer_indices:
        all_sims = []
        for (_, feats_pre), (_, feats_post) in zip(pre_data, post_data):
            pre_f = feats_pre[li]
            post_f = feats_post[li]
            n = min(pre_f.shape[0], post_f.shape[0], n_samples // len(pre_data))
            idx = torch.randperm(min(pre_f.shape[0], post_f.shape[0]))[:n]
            sims = F.cosine_similarity(pre_f[idx], post_f[idx], dim=1)
            all_sims.append(sims)
        results[li] = torch.cat(all_sims).numpy()
    return results


def compute_cka(pre_data, post_data, layer_indices, n_samples=2000):
    """Linear CKA between layers to measure redundancy."""
    # Pool frames across patients
    layer_feats = {}
    for li in layer_indices:
        pre_all = torch.cat([feats[li] for _, feats in pre_data], dim=0)
        post_all = torch.cat([feats[li] for _, feats in post_data], dim=0)
        combined = torch.cat([pre_all, post_all], dim=0)
        idx = torch.randperm(combined.shape[0])[:n_samples]
        layer_feats[li] = combined[idx].numpy()

    def linear_cka(X, Y):
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
        hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
        return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)

    n = len(layer_indices)
    cka_matrix = np.zeros((n, n))
    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            cka_matrix[i, j] = linear_cka(layer_feats[li], layer_feats[lj])

    return cka_matrix


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_delta_magnitude(stats, layer_indices, save_dir):
    """Bar chart of per-layer mean shift magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # L2 norm of delta
    l2s = [stats[li]['delta_mean'].norm().item() for li in layer_indices]
    axes[0].bar([f"L{li+1}" for li in layer_indices], l2s, color='steelblue')
    axes[0].set_ylabel("L2 norm of mean shift")
    axes[0].set_title("Pre→Post Mean Shift Magnitude (per layer)")

    # Top-20 dimensions with largest delta (averaged across layers)
    avg_delta = torch.stack([stats[li]['delta_mean'].abs() for li in layer_indices]).mean(dim=0)
    top_dims = avg_delta.topk(20)
    axes[1].barh(range(20), top_dims.values.numpy(), color='coral')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f"dim {d}" for d in top_dims.indices.numpy()])
    axes[1].set_xlabel("|delta|")
    axes[1].set_title("Top 20 Most Changed Dimensions (avg across L12-16)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'delta_magnitude.png'), dpi=150)
    plt.close()
    print(f"  Saved delta_magnitude.png")


def plot_similarity_distributions(frame_sims, layer_indices, save_dir):
    """Violin/histogram of frame-level cosine similarities per layer."""
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [frame_sims[li] for li in layer_indices]
    labels = [f"L{li+1}" for li in layer_indices]
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine similarity (pre vs post, same patient)")
    ax.set_title("Frame-Level Pre/Post Similarity Distribution per Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'frame_similarity.png'), dpi=150)
    plt.close()
    print(f"  Saved frame_similarity.png")


def plot_patient_similarity(patient_sims, layer_indices, save_dir):
    """Per-patient similarity heatmap across layers."""
    names = [p['name'] for p in patient_sims[layer_indices[0]]]
    n_patients = len(names)
    n_layers = len(layer_indices)
    matrix = np.zeros((n_patients, n_layers))
    for j, li in enumerate(layer_indices):
        for i, p in enumerate(patient_sims[li]):
            matrix[i, j] = p['sim']

    fig, ax = plt.subplots(figsize=(8, max(6, n_patients * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.8, vmax=1.0)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{li+1}" for li in layer_indices])
    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Per-Patient Pre/Post Cosine Similarity (utterance-level)")
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'patient_similarity.png'), dpi=150)
    plt.close()
    print(f"  Saved patient_similarity.png")


def plot_cka_matrix(cka_matrix, layer_indices, save_dir):
    """CKA similarity heatmap between layers."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_matrix, cmap='viridis', vmin=0.5, vmax=1.0)
    labels = [f"L{li+1}" for li in layer_indices]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Linear CKA Between Layers")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cka_matrix[i,j]:.2f}", ha='center', va='center',
                    color='white' if cka_matrix[i,j] < 0.75 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cka_layers.png'), dpi=150)
    plt.close()
    print(f"  Saved cka_layers.png")


def compute_content_invariance(extractor, surgery, layer_indices, max_patients=None):
    """
    For each pre-surgery speaker, extract layers 12-16 from multiple content types
    (Speech, Vowels A/E/I/O/U, TDU words). Compute within-speaker cross-content
    cosine similarity vs between-speaker similarity to measure content invariance.

    Returns:
        within_sims: {layer_idx: [sim, ...]} — same speaker, different content
        between_sims: {layer_idx: [sim, ...]} — different speakers, same content
        content_labels: list of content type names
    """
    # Gather all content sources for pre-surgery (session 1)
    base = os.path.join(CUCO_BASE, surgery)
    content_sources = {}

    # Speech
    speech_dir = os.path.join(base, "Speech", "1")
    speech_files = sorted(glob.glob(os.path.join(speech_dir, "*.wav")))
    if speech_files:
        content_sources['Speech'] = speech_files

    # Vowels (A, E, I, O, U)
    for vowel in ['A', 'E', 'I', 'O', 'U']:
        vdir = os.path.join(base, "Vowels", vowel, "1")
        vfiles = sorted(glob.glob(os.path.join(vdir, "*.wav")))
        if vfiles:
            content_sources[f'Vowel_{vowel}'] = vfiles

    # TDU words
    for word in ['Agua', 'Brasero', 'Dia', 'Mesa']:
        wdir = os.path.join(base, "TDU", word, "1")
        wfiles = sorted(glob.glob(os.path.join(wdir, "*.wav")))
        if wfiles:
            content_sources[f'TDU_{word}'] = wfiles

    content_labels = list(content_sources.keys())
    print(f"  Content types found: {content_labels}")

    # Extract patient IDs from filenames (last 4 digits)
    def get_patient_id(path):
        stem = Path(path).stem
        return stem.split('_')[-1]

    # Build patient → {content_type: wav_path} mapping
    all_patients = set()
    for ctype, files in content_sources.items():
        for f in files:
            all_patients.add(get_patient_id(f))
    all_patients = sorted(all_patients)
    if max_patients:
        all_patients = all_patients[:max_patients]

    # Extract features: patient_feats[patient_id][content_type] = {layer_idx: (T, 1024)}
    print(f"  Extracting features for {len(all_patients)} patients x {len(content_labels)} content types...")
    patient_feats = {}
    for pid in all_patients:
        patient_feats[pid] = {}
        for ctype, files in content_sources.items():
            matching = [f for f in files if get_patient_id(f) == pid]
            if matching:
                feats = extractor.extract(matching[0])
                patient_feats[pid][ctype] = feats
        n_types = len(patient_feats[pid])
        print(f"    Patient {pid}: {n_types} content types")

    # Compute within-speaker cross-content similarity (utterance-level averages)
    within_sims = {li: [] for li in layer_indices}
    between_sims = {li: [] for li in layer_indices}

    for li in layer_indices:
        # Within-speaker: all pairs of content types for same speaker
        for pid in all_patients:
            ctypes_available = list(patient_feats[pid].keys())
            for i in range(len(ctypes_available)):
                for j in range(i + 1, len(ctypes_available)):
                    ct_i = ctypes_available[i]
                    ct_j = ctypes_available[j]
                    avg_i = patient_feats[pid][ct_i][li].mean(dim=0)
                    avg_j = patient_feats[pid][ct_j][li].mean(dim=0)
                    sim = F.cosine_similarity(avg_i.unsqueeze(0), avg_j.unsqueeze(0)).item()
                    within_sims[li].append(sim)

        # Between-speaker: same content type, different speakers
        for ctype in content_labels:
            pids_with = [pid for pid in all_patients if ctype in patient_feats[pid]]
            for i in range(len(pids_with)):
                for j in range(i + 1, min(i + 5, len(pids_with))):  # limit pairs
                    avg_i = patient_feats[pids_with[i]][ctype][li].mean(dim=0)
                    avg_j = patient_feats[pids_with[j]][ctype][li].mean(dim=0)
                    sim = F.cosine_similarity(avg_i.unsqueeze(0), avg_j.unsqueeze(0)).item()
                    between_sims[li].append(sim)

        w_mean = np.mean(within_sims[li])
        b_mean = np.mean(between_sims[li])
        print(f"  Layer {li+1}: within-speaker (cross-content) = {w_mean:.4f}, "
              f"between-speaker (same-content) = {b_mean:.4f}, "
              f"gap = {w_mean - b_mean:+.4f}")

    return within_sims, between_sims, content_labels, patient_feats


def plot_content_invariance(within_sims, between_sims, layer_indices, save_dir):
    """Box plot comparing within-speaker vs between-speaker similarity per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin plot of within vs between per layer
    positions = []
    data_within = []
    data_between = []
    labels = []
    for i, li in enumerate(layer_indices):
        data_within.append(within_sims[li])
        data_between.append(between_sims[li])
        labels.append(f"L{li+1}")

    x = np.arange(len(layer_indices))
    width = 0.35
    ax = axes[0]
    bp1 = ax.boxplot(data_within, positions=x - width/2, widths=width * 0.8,
                     patch_artist=True, showfliers=False)
    bp2 = ax.boxplot(data_between, positions=x + width/2, widths=width * 0.8,
                     patch_artist=True, showfliers=False)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Within-Speaker (cross-content) vs Between-Speaker (same-content)")
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              ['Same speaker, diff content', 'Diff speaker, same content'],
              loc='lower left')

    # Right: gap (within - between) per layer
    ax = axes[1]
    gaps = [np.mean(within_sims[li]) - np.mean(between_sims[li]) for li in layer_indices]
    within_means = [np.mean(within_sims[li]) for li in layer_indices]
    between_means = [np.mean(between_sims[li]) for li in layer_indices]
    ax.bar(labels, gaps, color='seagreen')
    ax.set_ylabel("Gap (within - between)")
    ax.set_title("Content Invariance Gap per Layer\n(higher = more speaker-specific, less content-dependent)")
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add value labels
    for i, (g, w, b) in enumerate(zip(gaps, within_means, between_means)):
        ax.text(i, g + 0.002, f"{g:.3f}\n(w={w:.3f}, b={b:.3f})",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'content_invariance.png'), dpi=150)
    plt.close()
    print(f"  Saved content_invariance.png")


def plot_content_invariance_per_patient(patient_feats, layer_indices, all_patients, save_dir):
    """Heatmap: for each patient, avg within-speaker cross-content sim per layer."""
    n_patients = len(all_patients)
    n_layers = len(layer_indices)
    matrix = np.zeros((n_patients, n_layers))

    for pi, pid in enumerate(all_patients):
        for lj, li in enumerate(layer_indices):
            ctypes = list(patient_feats[pid].keys())
            sims = []
            for a in range(len(ctypes)):
                for b in range(a + 1, len(ctypes)):
                    avg_a = patient_feats[pid][ctypes[a]][li].mean(dim=0)
                    avg_b = patient_feats[pid][ctypes[b]][li].mean(dim=0)
                    sims.append(F.cosine_similarity(
                        avg_a.unsqueeze(0), avg_b.unsqueeze(0)).item())
            matrix[pi, lj] = np.mean(sims) if sims else 0.0

    fig, ax = plt.subplots(figsize=(8, max(6, n_patients * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0.7, vmax=1.0)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{li+1}" for li in layer_indices])
    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(all_patients, fontsize=7)
    ax.set_title("Per-Patient Content Invariance\n(same speaker, different content — higher = more invariant)")
    plt.colorbar(im, ax=ax, label="Avg cosine similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'content_invariance_per_patient.png'), dpi=150)
    plt.close()
    print(f"  Saved content_invariance_per_patient.png")


def plot_pca_tsne(pre_data, post_data, layer_indices, save_dir, n_samples=1000):
    """PCA and t-SNE of pre vs post for each layer."""
    fig, axes = plt.subplots(len(layer_indices), 2, figsize=(12, 4 * len(layer_indices)))
    if len(layer_indices) == 1:
        axes = axes[np.newaxis, :]

    for row, li in enumerate(layer_indices):
        pre_all = torch.cat([feats[li] for _, feats in pre_data], dim=0)
        post_all = torch.cat([feats[li] for _, feats in post_data], dim=0)

        # Subsample
        n_pre = min(n_samples, pre_all.shape[0])
        n_post = min(n_samples, post_all.shape[0])
        pre_sub = pre_all[torch.randperm(pre_all.shape[0])[:n_pre]].numpy()
        post_sub = post_all[torch.randperm(post_all.shape[0])[:n_post]].numpy()

        combined = np.concatenate([pre_sub, post_sub], axis=0)
        labels = np.array([0] * n_pre + [1] * n_post)

        # PCA
        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(combined)
        ax = axes[row, 0]
        ax.scatter(pca_out[:n_pre, 0], pca_out[:n_pre, 1], s=3, alpha=0.3, c='blue', label='Pre')
        ax.scatter(pca_out[n_pre:, 0], pca_out[n_pre:, 1], s=3, alpha=0.3, c='red', label='Post')
        ax.set_title(f"Layer {li+1} — PCA")
        ax.legend(markerscale=4)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
        tsne_out = tsne.fit_transform(combined)
        ax = axes[row, 1]
        ax.scatter(tsne_out[:n_pre, 0], tsne_out[:n_pre, 1], s=3, alpha=0.3, c='blue', label='Pre')
        ax.scatter(tsne_out[n_pre:, 0], tsne_out[n_pre:, 1], s=3, alpha=0.3, c='red', label='Post')
        ax.set_title(f"Layer {li+1} — t-SNE")
        ax.legend(markerscale=4)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_tsne.png'), dpi=150)
    plt.close()
    print(f"  Saved pca_tsne.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surgery', type=str, default='Tonsill')
    parser.add_argument('--max_patients', type=int, default=None,
                        help="Limit patients for faster analysis")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    pre_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "1")
    post_dir = os.path.join(CUCO_BASE, args.surgery, "Speech", "2")
    pre_files = sorted(glob.glob(os.path.join(pre_dir, "*.wav")))
    post_files = sorted(glob.glob(os.path.join(post_dir, "*.wav")))
    assert len(pre_files) == len(post_files), f"Mismatch: {len(pre_files)} vs {len(post_files)}"

    if args.max_patients:
        pre_files = pre_files[:args.max_patients]
        post_files = post_files[:args.max_patients]
    print(f"\n{args.surgery}: {len(pre_files)} patients")

    # Extract
    layer_indices = [l - 1 for l in LAYERS]
    extractor = WavLMMultiLayerExtractor(device, LAYERS)

    print("\nExtracting pre-surgery features...")
    pre_data = extract_and_cache(
        extractor, pre_files,
        os.path.join(CACHE_DIR, f'pre_layers_{args.surgery.lower()}.pt'))

    print("\nExtracting post-surgery features...")
    post_data = extract_and_cache(
        extractor, post_files,
        os.path.join(CACHE_DIR, f'post_layers_{args.surgery.lower()}.pt'))

    # Analysis
    print("\n" + "=" * 60)
    print("  Global Statistics")
    print("=" * 60)
    stats = compute_global_stats(pre_data, post_data, layer_indices)

    print("\n" + "=" * 60)
    print("  Per-Patient Similarity")
    print("=" * 60)
    patient_sims = compute_per_patient_similarity(pre_data, post_data, layer_indices)

    print("\n" + "=" * 60)
    print("  Frame-Level Similarity Distribution")
    print("=" * 60)
    frame_sims = compute_frame_similarity_distribution(pre_data, post_data, layer_indices)

    print("\n" + "=" * 60)
    print("  CKA Between Layers")
    print("=" * 60)
    cka_matrix = compute_cka(pre_data, post_data, layer_indices)
    for i, li in enumerate(layer_indices):
        for j, lj in enumerate(layer_indices):
            if j > i:
                print(f"  CKA(L{li+1}, L{lj+1}) = {cka_matrix[i,j]:.4f}")

    # Content invariance
    print("\n" + "=" * 60)
    print("  Content Invariance (same speaker, different content)")
    print("=" * 60)
    within_sims, between_sims, content_labels, patient_feats_multi = \
        compute_content_invariance(extractor, args.surgery, layer_indices, args.max_patients)

    # Plots
    print("\n" + "=" * 60)
    print("  Generating Plots")
    print("=" * 60)
    plot_delta_magnitude(stats, layer_indices, PLOT_DIR)
    plot_similarity_distributions(frame_sims, layer_indices, PLOT_DIR)
    plot_patient_similarity(patient_sims, layer_indices, PLOT_DIR)
    plot_cka_matrix(cka_matrix, layer_indices, PLOT_DIR)
    plot_pca_tsne(pre_data, post_data, layer_indices, PLOT_DIR)
    plot_content_invariance(within_sims, between_sims, layer_indices, PLOT_DIR)

    # Per-patient content invariance heatmap
    all_pids = sorted(patient_feats_multi.keys())
    plot_content_invariance_per_patient(patient_feats_multi, layer_indices, all_pids, PLOT_DIR)

    # Save stats
    torch.save({
        'stats': stats,
        'patient_sims': patient_sims,
        'cka_matrix': cka_matrix,
        'layers': LAYERS,
        'content_invariance': {
            'within_sims': {li: within_sims[li] for li in layer_indices},
            'between_sims': {li: between_sims[li] for li in layer_indices},
            'content_labels': content_labels,
        },
    }, os.path.join(CACHE_DIR, f'layer_stats_{args.surgery.lower()}.pt'))
    print(f"\nSaved stats to cache/layer_stats_{args.surgery.lower()}.pt")
    print("Done!")


if __name__ == '__main__':
    main()
