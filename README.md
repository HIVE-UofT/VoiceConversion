# Voice Conversion for Pre/Post Tonsillectomy Speech

Non-parallel voice conversion between pre-surgery and post-surgery (tonsillectomy) speech. The goal is to convert a pre-surgery voice to sound like the same speaker post-surgery, preserving linguistic content while transforming voice quality (resonance, nasality, vocal tract characteristics).

## Dataset

**CUCO Dataset** — Paired recordings from tonsillectomy patients:
- **28 patients**, each recorded before (session 1) and after (session 2) surgery
- Speech tasks: reading passages, sustained vowels, etc.
- Raw audio: 44.1 kHz stereo WAV files
- Processed: 80-band mel-spectrograms (SR=16kHz, n_fft=2048, hop_length=512), normalized to [0, 1]
- Segmented into 5-second chunks: ~230 pre-surgery + ~235 post-surgery segments
- Patient-level train/val/test split (70/15/15) to prevent data leakage

**Data location:**
```
/home/sepharfi/projects/def-zshakeri/sepehr/CUCO/data_final/
├── Audios/Tonsill/Speech/
│   ├── 1/    # Pre-surgery WAV files (28 files)
│   └── 2/    # Post-surgery WAV files (28 files)
└── processed_data/
    ├── train_dataset.pkl
    ├── val_dataset.pkl
    └── test_dataset.pkl
```

## Methods

Four approaches were explored, progressing from simple baselines to more sophisticated architectures:

### 1. VAE (Baseline)

**Directory:** `VAE/`

Variational autoencoder with explicit content/surgery disentanglement. The first attempt at separating *what* is said from *how* it sounds.

- **Architecture:** Conv2D encoder → LSTM → two latent heads: content (512-dim, continuous) and surgery status (8-dim, continuous). Decoder reconstructs mel from both. Gradient reversal on content to prevent it from encoding surgery status.
- **Losses:** Reconstruction (L1), KL divergence (content + surgery), adversarial (gradient reversal on content), surgery truth classification
- **Status:** Initial baseline. Established the disentanglement framework but limited by continuous latent space — no discrete bottleneck to force true separation.

### 2. MaskCycleGAN-VC

**Directory:** `mask_cyclegan/`

CycleGAN-based approach with Filling-in-Frames (FIF) masking, based on [Kaneko et al., 2021](https://arxiv.org/abs/2102.12841). Unpaired training — no need for aligned samples.

- **Architecture:** 2-1-2D CNN generator (2D downsample → 1D residual blocks → 2D upsample) with GLU activations, PatchGAN discriminator
- **Losses:** Adversarial (LSGAN), cycle-consistency (A→B→A), identity (low weight=0.5 for subtle domains), multi-resolution STFT
- **FIF mechanism:** Random temporal masking forces generator to learn speech structure through cycle + adversarial loss on masked input
- **Key fixes applied:** Removed broken FIF L1 loss (was comparing unpaired samples), added multi-res STFT, balanced D/G learning rates, reduced identity weight (5.0→0.5), added N_D_STEPS=2

**Issues encountered:**
- Discriminator collapse — pre/post surgery domains are too subtle for adversarial training
- Near-identity conversions — generator learned to pass input through unchanged
- CycleGAN fundamentally struggles with subtle domain differences where the two domains share most acoustic properties

### 3. kNN-VC (Baseline Comparison)

**Directory:** `knn_vc/`

k-Nearest Neighbors Voice Conversion based on [Baas et al., Interspeech 2023](https://arxiv.org/abs/2305.18975). No training required — uses pre-trained self-supervised features.

- **How it works:**
  1. Extract WavLM-Large features from all post-surgery recordings → matching set (83,437 frames)
  2. For each source (pre-surgery) frame, find k=4 nearest neighbors in the matching set
  3. Replace source frame with the mean of its neighbors
  4. Reconstruct audio with HiFi-GAN vocoder
- **Advantages:** No training on small dataset, leverages WavLM pre-trained on thousands of hours of speech, simple and fast
- **Status:** Inference complete (28 files converted), evaluation pending

### 4. VQVAE with Feature Disentanglement (Main Method)

**Directory:** `vqvae/`

Vector-quantized VAE that explicitly disentangles voice into content (discrete VQ codes) and voice quality (continuous vector). The discrete bottleneck forces content through a fixed-size codebook, stripping quality information.

- **Architecture:**
  - ContentEncoder: mel → continuous features, T/8 temporal downsampling
  - ProductVectorQuantizer: 4 heads × 16 codes × 16-dim each (65K effective combinations)
  - VoiceQualityEncoder: mel → 32-dim quality vector (captures resonance/nasality)
  - Decoder: quantized content + quality → mel (8x upsample with ResBlock2d)
  - DomainClassifier: Bidirectional GRU adversarial classifier on content codes

- **Losses:**
  | Loss | Purpose |
  |------|---------|
  | Reconstruction (L1) | Decoded output must match input mel |
  | VQ commitment + entropy | Codebook usage: commitment keeps encoder near codes, entropy encourages uniform usage |
  | Multi-resolution STFT | Preserves harmonic/spectral detail |
  | Adversarial (gradient reversal) | Content must NOT predict pre/post surgery |
  | Quality classification (BCE) | Quality vector MUST predict pre/post surgery |
  | Cycle (cross-reconstruction) | Swap quality between domains, re-encode, verify content + quality preserved |

- **Conversion at inference:** Encode content from source → quantize → combine with average post-surgery quality vector from training set → decode

**Experiments run (detailed in `vqvae/PROGRESS.md`):**

| Experiment | Key Changes | Perplexity | Val Recon | Main Issue |
|------------|-------------|------------|-----------|------------|
| 1 | Baseline (256 codes, no cycle) | 10/256 | 0.0390 | Codebook collapse, near-identity conversion |
| 2 | + Cycle loss, dead code reset, quality dropout | 15/256 | 0.0337 | Still collapsed, steganography in code sequences |
| 3 | + T/8 bottleneck, 64 codes, GRU classifier, content noise | 9/64 | 0.0433 | Collapse worse, reconstruction degraded |
| 4 | + Product VQ (4×16), entropy regularization, lower commitment | *pending* | *pending* | — |

**Key findings:**
- Disentanglement works well across all experiments (adversarial loss stays at random chance ~0.693)
- Quality encoder successfully separates pre/post surgery (BCE < 0.02)
- Codebook collapse is the persistent blocker — only ~10 codes used regardless of codebook size or reset strategies
- Product quantization + entropy regularization (Exp 4) targets this directly

### 5. Mean Shift VC (Domain-Level Feature Transform)

**Directory:** `mean_shift/`

The simplest possible domain transform: shift all WavLM features by the difference in domain means.

- **How it works:** `converted = source_features + (mean_post - mean_pre)` in WavLM space, then vocode with HiFi-GAN
- **Training:** Compute mean WavLM embedding for pre and post surgery domains (no neural network)
- **Inference:** Apply fixed shift — no reference audio needed
- **Rationale:** If pre/post surgery differences are captured as a consistent direction in WavLM embedding space, a simple translation should work

### 6. LinearVC (Linear Domain Transform)

**Directory:** `linear_vc/`

Learns a linear projection matrix W in WavLM space, based on [LinearVC (2025)](https://arxiv.org/html/2506.01510).

- **How it works:** `converted = source_features @ W` where W is a 1024x1024 matrix
- **Training:** Pair pre/post surgery frames via nearest neighbors, solve ridge regression: `W = (X^T X + λI)^{-1} X^T Y`
- **Inference:** Matrix multiply — no reference audio needed
- **Key insight:** Content and speaker/quality information occupy orthogonal subspaces in WavLM. A linear map can rotate from one domain to another while preserving content.

### 7. MKL-VC (Factorized Optimal Transport)

**Directory:** `mkl_vc/`

Optimal transport map between domain distributions in WavLM space, based on [MKL-VC (Interspeech 2025)](https://arxiv.org/html/2506.09709).

- **How it works:** Models pre and post surgery WavLM features as multivariate Gaussians, computes the Monge-Kantorovich Linear transport map (closed-form solution)
- **Factorization:** Splits 1024-dim features into K=2 dimensional subgroups sorted by variance, solves OT independently per subgroup. Prevents information loss from low-variance dimensions.
- **Training:** Compute domain-level Gaussian statistics (mean + covariance per subgroup)
- **Inference:** Apply analytical transform — no reference audio needed
- **Advantage over kNN-VC:** Generates new feature combinations rather than retrieving existing frames

### 8. UNet-VC (Nonlinear Feature Transform)

**Directory:** `unet_vc/`

Residual 1D U-Net that learns a nonlinear transform in WavLM feature space. Extends LinearVC by allowing nonlinear mappings while preserving content through skip connections and residual learning.

- **Architecture:** Residual 1D U-Net: project 1024→256, 3-level encoder/decoder with skip connections, project back to 1024. Global residual: `output = input + α * network(input)` where α is learned (initialized at 0.1)
- **Training:** NN-paired frames (same as LinearVC), segmented into overlapping 64-frame windows. MSE loss, AdamW with cosine schedule, early stopping
- **Inference:** Forward pass through U-Net + HiFi-GAN vocoding — no reference audio needed
- **Key design:** Residual learning means network only learns the small delta between domains. Skip connections preserve content structure. Lightweight (~2M params) to avoid overfitting on small dataset

## Project Structure

```
VoiceConversion/
├── README.md
├── requirements.txt
├── VAE/
│   ├── model/model.py              # SurgeryVAE architecture
│   ├── scripts/
│   │   ├── dataset_processing.py   # Raw audio → mel-spectrogram pkl
│   │   ├── train.py                # Training with disentanglement losses
│   │   └── test.py                 # Testing/inference
│   └── submit.sh
├── mask_cyclegan/
│   ├── model/mask_cyclegan.py      # Generator + Discriminator + FIF masking
│   ├── scripts/
│   │   ├── train.py                # CycleGAN training loop
│   │   ├── inference.py            # Single/batch conversion
│   │   └── evaluate.py             # MCD, F0, cycle, identity metrics
│   └── submit.sh
├── knn_vc/
│   ├── scripts/
│   │   ├── build_matching_set.py   # Extract WavLM features from post-surgery
│   │   ├── inference.py            # kNN lookup + HiFi-GAN vocoding
│   │   └── evaluate.py             # MCD, F0, content preservation metrics
│   ├── matching_sets/              # Pre-computed WavLM feature tensors
│   ├── knn_vc_converted/           # 28 converted WAV files
│   ├── submit_build.sh
│   ├── submit_inference.sh
│   └── submit_evaluate.sh
├── vqvae/
│   ├── model/
│   │   ├── vqvae.py                # VQVAE, ProductVQ, encoders, decoder, GRU classifier
│   │   └── __init__.py
│   ├── scripts/
│   │   ├── train.py                # Training with 6 losses
│   │   ├── inference.py            # Convert using avg quality vector
│   │   └── evaluate.py             # MCD, F0, disentanglement probes
│   ├── checkpoints/                # Saved model weights
│   ├── plots/                      # Training visualizations per epoch
│   ├── PROGRESS.md                 # Detailed experiment log with analysis
│   └── submit.sh
├── mean_shift/
│   ├── scripts/
│   │   ├── train.py                # Compute domain mean vectors
│   │   ├── inference.py            # Apply mean shift + vocode
│   │   └── evaluate.py             # MCD, F0, content preservation
│   ├── domain_stats.pt             # Saved mean vectors (after training)
│   └── submit.sh
├── linear_vc/
│   ├── scripts/
│   │   ├── train.py                # Learn linear projection W via ridge regression
│   │   ├── inference.py            # Apply W @ features + vocode
│   │   └── evaluate.py             # MCD, F0, content preservation
│   ├── linear_transform.pt         # Saved W matrices (after training)
│   └── submit.sh
├── mkl_vc/
│   ├── scripts/
│   │   ├── train.py                # Compute factorized OT maps
│   │   ├── inference.py            # Apply MKL transform + vocode
│   │   └── evaluate.py             # MCD, F0, content preservation
│   ├── mkl_transform.pt            # Saved OT parameters (after training)
│   └── submit.sh
└── unet_vc/
    ├── model/
    │   ├── unet.py                 # Residual 1D U-Net architecture
    │   └── __init__.py
    ├── scripts/
    │   ├── train.py                # Train on NN-paired WavLM features
    │   ├── inference.py            # Apply U-Net transform + vocode
    │   └── evaluate.py             # MCD, F0, content preservation
    ├── checkpoints/                # Saved model weights (after training)
    └── submit.sh
```

## Environment

- **Compute:** Compute Canada (def-zshakeri), H100 GPU
- **Python:** 3.10
- **Key dependencies:** PyTorch, librosa, torchaudio, soundfile, scikit-learn
- **SLURM:** All jobs submitted via `sbatch submit.sh` in each method directory

## Evaluation Metrics

All methods are evaluated with comparable metrics:

| Metric | What it measures | Ideal |
|--------|-----------------|-------|
| MCD to target | Spectral distance between converted and real post-surgery | Lower |
| Content preservation MCD | Spectral distance between source and converted | Lower |
| F0 Correlation | Pitch tracking: source vs converted | Higher |
| Disentanglement (VQVAE only) | Can a linear probe predict surgery status from content codes? | ~50% accuracy |
| Quality classification (VQVAE only) | Can a linear probe predict surgery status from quality vector? | ~100% accuracy |

## References

- [AutoVC (Qian et al., ICML 2019)](https://arxiv.org/abs/1905.05879) — Information bottleneck for VC
- [VQVC+ (Wu et al., Interspeech 2020)](https://www.isca-archive.org/interspeech_2020/wu20p_interspeech.html) — VQ disentanglement for VC
- [MaskCycleGAN-VC (Kaneko et al., 2021)](https://arxiv.org/abs/2102.12841) — CycleGAN with FIF masking
- [kNN-VC (Baas et al., Interspeech 2023)](https://arxiv.org/abs/2305.18975) — Nearest neighbor VC with WavLM
- [PQ-VAE (2024)](https://arxiv.org/html/2406.02940v1) — Product quantization for speech tokenization
- [LinearVC (2025)](https://arxiv.org/html/2506.01510) — Linear transforms of SSL features for VC
- [MKL-VC (Interspeech 2025)](https://arxiv.org/html/2506.09709) — Training-free VC with factorized optimal transport
- [Deep Learning for Pathological Speech (2025)](https://arxiv.org/html/2501.03536v1) — Survey of DL methods for pathological speech
