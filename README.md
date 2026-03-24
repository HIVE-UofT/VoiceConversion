# Voice Conversion for Pre/Post Tonsillectomy Speech

Non-parallel voice conversion between pre-surgery and post-surgery (tonsillectomy) speech using the CUCO dataset. The goal is to convert a pre-surgery voice to sound like the same speaker post-surgery, preserving linguistic content while transforming voice quality (resonance, nasality, vocal tract characteristics).

## Results

All methods evaluated on the **Tonsill** subset (28 patients) using ECAPA-TDNN speaker cosine similarity — the primary metric measuring content-independent voice identity match between converted and real post-surgery audio.

| Method | SpkSim (conv→post) | vs Baseline | Type | Training Required |
|--------|:------------------:|:-----------:|------|:-----------------:|
| **UNet-VC** | **0.792 ± 0.055** | **+19.8%** | Nonlinear feature transform | Yes (~2M params) |
| kNN-VC | 0.710 ± 0.084 | +7.4% | Nearest-neighbor retrieval | No |
| MKL-VC | 0.642 ± 0.097 | -2.9% | Optimal transport | No |
| Mean-Shift | 0.638 ± 0.098 | -3.5% | Mean translation | No |
| VQVAE Exp5 | 0.394 ± 0.119 | -40.4% | VQ disentanglement | Yes |
| VQ-UNet (Exp6) | *pending* | — | VQ + U-Net + FiLM | Yes |
| **AdaptVC** | *pending* | — | **Adapter + VQ + FiLM U-Net** | Yes |
| LinearVC | *failed* | — | Linear projection | No |

**Baseline** (pre vs post, same patient, no conversion): **0.661 ± 0.118**

Higher SpkSim = converted voice sounds more like the post-surgery voice. The baseline is the floor — a good conversion should beat it. UNet-VC currently leads by a wide margin.

### Baselines Across Surgery Types

| Condition | Baseline SpkSim | n | Interpretation |
|-----------|:---------------:|:-:|---------------|
| Contr (control) | 0.886 ± 0.056 | 28 | No surgery — natural session variability |
| Fess | 0.828 ± 0.076 | 27 | Mild surgery effect |
| Sept | 0.828 ± 0.065 | 32 | Mild surgery effect |
| **Tonsill** | **0.661 ± 0.118** | **28** | **Largest surgery effect — hardest to convert** |

## Dataset

**CUCO Dataset** — Paired recordings from tonsillectomy patients:
- **28 patients**, each recorded before (session 1) and after (session 2) surgery
- Speech tasks: reading passages, sustained vowels
- 4 conditions: Tonsill, Fess, Sept, Contr (control)
- Audio: resampled to 16kHz mono for processing

```
CUCO/data_final/Audios/
├── Tonsill/Speech/{1,2}/   # 28 pre + 28 post files
├── Fess/Speech/{1,2}/      # 27 pre + 27 post files
├── Sept/Speech/{1,2}/      # 32 pre + 32 post files
└── Contr/Speech/{1,2}/     # 28 pre + 28 post files
```

## Methods

### Baselines (No/Minimal Training)

#### kNN-VC — Nearest-Neighbor Retrieval
**Directory:** `knn_vc/` | [Baas et al., Interspeech 2023](https://arxiv.org/abs/2305.18975)

Replaces each source WavLM frame with the mean of its k=4 nearest neighbors from the post-surgery matching set. No training — leverages frozen WavLM-Large + HiFi-GAN vocoder. Strong baseline (0.710).

#### Mean-Shift — Domain Mean Translation
**Directory:** `mean_shift/`

`converted = source + (mean_post - mean_pre)` in WavLM space. Assumes surgery effect is a consistent direction in embedding space. Simple but limited (0.638).

#### MKL-VC — Factorized Optimal Transport
**Directory:** `mkl_vc/` | [MKL-VC, Interspeech 2025](https://arxiv.org/html/2506.09709)

Models domains as Gaussians, computes Monge-Kantorovich transport map. Factorized into K=2 subgroups by variance. Generates new feature combinations (unlike kNN retrieval). Similar to Mean-Shift (0.642).

#### LinearVC — Linear Projection
**Directory:** `linear_vc/` | [LinearVC, 2025](https://arxiv.org/html/2506.01510)

Learns `W @ features` via ridge regression on NN-paired frames. Failed due to torch.load compatibility issue — not yet evaluated.

### Learned Methods

#### UNet-VC — Residual U-Net Feature Transform (Current Best)
**Directory:** `unet_vc/`

Residual 1D U-Net learning the small delta between pre and post surgery domains in WavLM space. `output = input + α * network(input)` where α is learned (init 0.1). Skip connections preserve content; residual learning captures the subtle domain shift. ~2M params. **Best result: 0.792.**

#### MaskCycleGAN-VC
**Directory:** `mask_cyclegan/` | [Kaneko et al., 2021](https://arxiv.org/abs/2102.12841)

CycleGAN with Filling-in-Frames masking on mel spectrograms. Failed — discriminator collapse due to subtle domain differences. Pre/post surgery domains are too similar for adversarial training to distinguish.

#### VAE (Initial Baseline)
**Directory:** `VAE/`

Variational autoencoder with content/surgery disentanglement on mel spectrograms. Established the disentanglement framework but limited by continuous latent space.

#### VQVAE — VQ Disentanglement on WavLM Features
**Directory:** `vqvae/`

Vector-quantized VAE operating on WavLM features with explicit content (VQ codes) + quality (continuous vector) disentanglement. Uses adversarial, quality classification, cycle consistency, and cross-reconstruction losses.

| Exp | Architecture | SpkSim | Issue |
|-----|-------------|:------:|-------|
| 1-4 | VQ on mel spectrograms | — | Codebook collapse, near-identity conversion |
| 5 | VQ on WavLM features (1D conv) | 0.394 | Bottleneck destroys too much info |
| 6 | VQ-UNet + FiLM skip connections | *pending* | Hybrid: U-Net skips + VQ bottleneck |

Exp5 showed disentanglement works (adversarial loss ~0.693, quality classification converges) but the reconstruction quality is too poor for useful conversion.

### AdaptVC — Adapter + VQ + FiLM U-Net (Proposed Method)
**Directory:** `adapt_vc/`

Novel architecture inspired by [AdaptVC (ICASSP 2025)](https://arxiv.org/abs/2501.01347), combining learned WavLM layer selection with VQ disentanglement and FiLM-conditioned U-Net decoding. Designed for paired surgical voice conversion with minimal data.

**Architecture:**
```
Raw Audio → WavLM-Large (frozen) → 24 hidden states
                │
    ┌───────────┴───────────┐
    │                       │
Content Adapter         Quality Adapter
(learned layer weights) (learned layer weights)
    │                       │
U-Net Encoder           Quality Encoder
    │── skip1 → FiLM ──┐   │
    │── skip2 → FiLM ──┤   └→ quality_vec (64-dim)
    ↓                   │         │
Content Proj → VQ ──────┴→ U-Net Decoder → WavLM layer 6 features
(4×32 codes)                               → HiFi-GAN → Audio
```

**Key innovations:**
1. **Dual adapters** — learned softmax-weighted sum over WavLM's 24 layers. Content adapter learns which layers encode linguistics (expected: mid-to-late layers). Quality adapter learns which layers encode surgery-related voice quality (expected: early acoustic layers).
2. **VQ + FiLM U-Net** — Product VQ at the bottleneck for content abstraction. Skip connections preserve detail. FiLM layers modulate skips based on quality vector.
3. **Paired training** — unlike AdaptVC's self-reconstruction, directly supervises pre→post transformation with cross-reconstruction, cycle consistency, and adversarial losses.
4. **On-the-fly WavLM extraction** — raw audio input, WavLM processes each batch live (no feature caching).

**Losses:** Reconstruction (MSE vs layer 6 features), VQ commitment + entropy, adversarial (gradient reversal on content), quality classification (BCE), cycle consistency, cross-reconstruction quality check.

**What makes this novel** (based on literature review of 30+ papers):

| Component | Closest Prior Work | How We Differ |
|---|---|---|
| Learned layer adapters for VC | AdaptVC: adapters on HuBERT-Base | We use dual adapters (content + quality) on WavLM-Large; paired training, not self-reconstruction |
| VQ + FiLM U-Net decoder in SSL space | None | No prior work combines VQ bottleneck + FiLM-conditioned U-Net operating in SSL feature space |
| WavLM→WavLM U-Net mapping | None | All prior U-Net VC methods output mel-spectrograms, not SSL features |
| Medical/pathological VC with adapters | None | No prior work applies SSL adapter-based VC to pathological voice |

## Evaluation

All methods use `shared_evaluate.py` with ECAPA-TDNN speaker similarity as the primary metric:

| Metric | What it Measures | Ideal |
|--------|-----------------|-------|
| **SpkSim (conv vs target)** | Does converted voice sound like post-surgery? | Higher (> baseline) |
| SpkSim (conv vs source) | How much source identity remains? | Reference |
| Baseline SpkSim | Pre vs post same patient (no conversion) | Floor to beat |
| LSD | Log-spectral distance to target | Lower |
| SED | Spectral envelope distance to target | Lower |

## Project Structure

```
VoiceConversion/
├── README.md
├── shared_evaluate.py          # Shared ECAPA-TDNN + spectral metrics
├── compare_all_spksim.py       # Cross-method comparison script
├── adapt_vc/                   # AdaptVC-inspired (proposed method)
│   ├── model/adapt_vc.py       # Adapters + VQ + FiLM U-Net decoder
│   ├── scripts/{train,inference,evaluate}.py
│   └── submit.sh
├── unet_vc/                    # Residual U-Net (current best)
│   ├── model/unet.py
│   ├── scripts/{train,inference,evaluate}.py
│   └── submit.sh
├── knn_vc/                     # kNN-VC baseline
│   ├── scripts/{build_matching_set,inference,evaluate}.py
│   └── submit_*.sh
├── vqvae/                      # VQVAE experiments 1-6
│   ├── model/{vqvae,vqvae_wavlm,vqvae_unet}.py
│   ├── scripts/{train,train_exp5,train_exp6,inference_exp5,inference_exp6,...}.py
│   └── submit_exp{5,6}.sh
├── mkl_vc/                     # Factorized OT
├── mean_shift/                 # Domain mean translation
├── linear_vc/                  # Linear projection
├── mask_cyclegan/              # CycleGAN (failed)
└── VAE/                        # Initial VAE baseline
```

## Environment

- **Compute:** Compute Canada (def-zshakeri), NVIDIA H100 GPU
- **Python:** 3.10
- **Key dependencies:** PyTorch, torchaudio, librosa, speechbrain, transformers (for WavLM)
- **SLURM:** `sbatch submit.sh` in each method directory

## References

- [kNN-VC (Baas et al., Interspeech 2023)](https://arxiv.org/abs/2305.18975) — Nearest neighbor VC with WavLM
- [AdaptVC (Kim et al., ICASSP 2025)](https://arxiv.org/abs/2501.01347) — Adaptive learning with VQ on HuBERT
- [MKL-VC (Interspeech 2025)](https://arxiv.org/html/2506.09709) — Training-free VC with factorized optimal transport
- [LinearVC (2025)](https://arxiv.org/html/2506.01510) — Linear transforms of SSL features for VC
- [Vevo (ICLR 2025)](https://arxiv.org/abs/2502.07243) — VQ-VAE tokenizer on HuBERT for controllable voice imitation
- [MaskCycleGAN-VC (Kaneko et al., 2021)](https://arxiv.org/abs/2102.12841) — CycleGAN with FIF masking
- [FreeVC (Li et al., ICASSP 2023)](https://arxiv.org/abs/2210.15418) — Text-free one-shot VC with WavLM bottleneck
- [VQMIVC (Wang et al., Interspeech 2021)](https://arxiv.org/abs/2106.10132) — VQ + mutual information disentanglement
- [RepCodec (ACL 2024)](https://arxiv.org/abs/2309.00169) — VQ codec for SSL speech representation
- [QR-VC (2024)](https://arxiv.org/abs/2411.16147) — Quantization residuals from WavLM for VC
- [AutoVC (Qian et al., ICML 2019)](https://arxiv.org/abs/1905.05879) — Information bottleneck for VC
- [W2VC (EURASIP 2023)](https://link.springer.com/article/10.1186/s13636-023-00312-8) — WavLM representation-based VC
- [Deep Learning for Pathological Speech (2025)](https://arxiv.org/html/2501.03536v1) — Survey
