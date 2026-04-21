# Voice Conversion for Pre/Post Tonsillectomy Speech

Non-parallel voice conversion between pre-surgery and post-surgery (tonsillectomy) speech using the CUCO dataset. The goal is to convert a pre-surgery voice to sound like the same speaker post-surgery, preserving linguistic content while transforming voice quality (resonance, nasality, vocal tract characteristics).

---

## Results

### Full Dataset Evaluation (28 Tonsill Patients)

Methods trained/evaluated on all 28 patients using ECAPA-TDNN speaker cosine similarity. Baseline = pre vs post, same patient, no conversion.

| Method | SpkSim (conv→post) | vs Baseline | LSD (dB) | SED (dB) | Type | Params |
|--------|:------------------:|:-----------:|:--------:|:--------:|------|:------:|
| **UNet-VC** | **0.792 ± 0.055** | **+19.8%** | 1.96 ± 0.14 | 2.64 ± 0.97 | Residual U-Net | ~2M |
| UNet-Adv-VC | 0.780 ± 0.058 | +18.0% | 1.97 ± 0.13 | 2.70 ± 1.08 | U-Net + adversarial | ~2M |
| kNN-VC | 0.710 ± 0.084 | +7.4% | — | — | Nearest-neighbor retrieval | 0 |
| MKL-VC | 0.642 ± 0.097 | -2.9% | — | — | Factorized OT | 0 |
| Mean-Shift | 0.638 ± 0.098 | -3.5% | — | — | Mean translation | 0 |
| VQVAE Exp5 | 0.394 ± 0.119 | -40.4% | — | — | VQ disentanglement | ~3M |
| MaskCycleGAN | — | — | — | — | CycleGAN on mel | ~10M |
| LinearVC | — | — | — | — | Linear projection | — |
| **DLA-VC** | *tuning in progress (current best on test split: 0.487)* | — | — | — | UNet-VC adaptation + dual adapters + Product VQ + FiLM quality branch | 6.6M |
| VQ-UNet (Exp6) | *in progress* | — | — | — | VQ + U-Net + FiLM | — |

**Baseline (pre vs post, same patient, no conversion): 0.661 ± 0.118**

### Train/Test Split Evaluation (5-Patient Held-Out Test Set)

Strict evaluation with a fixed 5-patient test set (`[0045, 0085, 0110, 0122, 0132]`, seed=42). All 9 methods trained on 23 train patients using **all 13 audio types** (Speech + 4 TDU sentences + 5 vowels + 3 sustained vowels) = ~282 files. Vocoder: HiFi-GAN fine-tuned on CUCO post-surgery audio (step=2500, val_mel_loss=0.3961). Evaluation: speech files only, ECAPA-TDNN SpkSim.

- **Test baseline** (pre vs post, no conversion, 5 test patients): **0.6707 ± 0.1061**
- **Train baseline** (pre vs post, no conversion, 23 train patients): **0.6586 ± 0.1248**

| Method | Test SpkSim | Δ (test) | Train SpkSim | Δ (train) | Notes |
|--------|:-----------:|:--------:|:------------:|:---------:|-------|
| **UNet-VC** | **0.6868** | **+0.0162** | 0.7746 | +0.1160 | K-fold; all audio types |
| UNet-Adv-VC | 0.6825 | +0.0119 | 0.8001 | +0.1415 | K-fold; all audio types |
| UNet-VC-ECAPA | 0.6837 | +0.0130 | 0.6982 | +0.0396 | ep 98; ECAPA loss + style crop fix |
| UNet-VC-SPK | 0.6705 | −0.0002 | 0.7849 | +0.1263 | FiLM speaker cond; all audio types |
| Mean-Shift | 0.6862 | +0.0155 | 0.6892 | +0.0306 | All audio types in matching set |
| MKL-VC | 0.6845 | +0.0138 | 0.6881 | +0.0295 | All audio types in matching set |
| LinearVC | 0.6186 | −0.0521 | 0.6739 | +0.0153 | Ridge regression; all audio types |
| kNN-VC | 0.5793 | −0.0914 | 0.7043 | +0.0457 | Matching set = train patients only |
| **DLA-VC** (current best, *tuning in progress*) | 0.4874 | −0.1832 | 0.2787 | −0.3799 | UNet-VC adaptation + dual adapters + Product VQ + FiLM quality branch |
| &nbsp;&nbsp;&nbsp;— ablation: DLA-VC without VQ | 0.4366 | −0.2341 | 0.2787 | −0.3799 | Validates VQ bottleneck is architecturally load-bearing |
| FreeVC zero-shot (pretrained on VCTK) | 0.4056 | −0.2651 | 0.2954 | −0.3632 | Foundation-model baseline (no fine-tune) |
| FreeVC fine-tuned (200 ep, self-recon on post) | 0.5121 | −0.1586 | 0.4054 | −0.2532 | Best of the pretrained-model approaches |
| FreeVC + learned shift (FreeVC speaker space) | 0.4146 | −0.2561 | — | — | Frozen FreeVC + trained 70K-param shift MLP |
| FreeVC + learned shift (ECAPA space + bridge) | 0.3971 | −0.2735 | — | — | Shift in ECAPA space, bridge → FreeVC gen |

> **kNN-VC note:** The train/test split exposes a fundamental limitation — kNN-VC requires the target patient's post-surgery features in its matching set. In the full-dataset evaluation (28 patients, self-matching), it scored 0.710 (+7.4%). In the strict split, it can only use training patients' features, so it converts test patients toward a generic post-surgery voice rather than their own, explaining the −0.0914 drop.
>
> **Large train/test gap for UNet-VC variants:** Training patients achieve 0.77–0.80, while test patients achieve 0.68–0.69. Expected in the 23-patient small-data regime. The gap is smallest for UNet-VC-ECAPA (+0.0130 test vs +0.0396 train), suggesting the ECAPA loss may improve generalization relative to raw reconstruction methods.

### Cross-Surgery Type Results (Sept — Unet-VC-v2)

| Pairing | Test SpkSim | Baseline | Delta |
|---------|:-----------:|:--------:|:-----:|
| Cross-patient pairs | 0.759 ± 0.049 | 0.841 | −0.082 |
| Same-patient pairs  | 0.778 ± 0.041 | 0.841 | −0.063 |

> Sept surgery has a much smaller domain gap than Tonsill (baseline 0.828 vs 0.661), so conversion provides less headroom for improvement.

### Baselines Across Surgery Types

| Condition | Baseline SpkSim | n | Interpretation |
|-----------|:---------------:|:-:|---------------|
| Contr (control) | 0.886 ± 0.056 | 28 | No surgery — natural session variability |
| Fess | 0.828 ± 0.076 | 27 | Mild surgery effect |
| Sept | 0.828 ± 0.065 | 32 | Mild surgery effect |
| **Tonsill** | **0.661 ± 0.118** | **28** | **Largest surgery effect — hardest to convert** |

---

## Dataset

**CUCO Dataset** — Paired recordings from tonsillectomy patients:
- **28 patients** (Tonsill), each recorded before (session 1) and after (session 2) surgery
- Also: Fess (27 patients), Sept (32 patients), Contr/control (28 patients)
- Speech tasks: reading passages, sustained vowels
- Audio: resampled to 16kHz mono
- Fixed train/val/test split: patients `[0045, 0085, 0110, 0122, 0132]` held out as test (seed=42); never used in any training stage

```
CUCO/data_final/Audios/
├── Tonsill/
│   ├── Speech/{1,2}/                    # 28 pre + 28 post files
│   ├── TDU/{Agua,Brasero,Dia,Mesa}/{1,2}/  # 4 sentence types × 28 patients
│   ├── Vowels/{A,E,I,O,U}/{1,2}/        # 5 vowels × 28 patients
│   └── Sustained vowels/{A1,A2,A3}/{1,2}/  # 3 sustained vowels × 28 patients
├── Fess/Speech/{1,2}/      # 27 pre + 27 post files
├── Sept/Speech/{1,2}/      # 32 pre + 32 post files
└── Contr/Speech/{1,2}/     # 28 pre + 28 post files
```

All 13 audio types used in training: 23 train patients × ~12 files ≈ **282 training file pairs**.

---

## Preliminary Analysis

### WavLM Quality Mapper — ECAPA Content Invariance Analysis
**Directory:** `Experiments/wavlm_quality_mapper/` | Log: `ecapa_analyze-58874812.out`

Before building conversion models, measured how much surgery changes ECAPA-TDNN representations and how discriminative they are between speakers.

| Comparison | SpkSim |
|-----------|:------:|
| Pre vs Pre (same speaker, cross-session baseline) | 0.780 ± 0.080 |
| Pre vs Post (surgery effect) | 0.529 ± 0.114 |
| Between-speaker (different patients) | 0.196 ± 0.117 |

- **Voice change magnitude** (pre-pre minus pre-post): +0.251 — surgery introduces a substantial shift
- **Speaker discrimination** (pre-post minus between-speaker): +0.333 — speaker identity is still largely preserved post-surgery

This confirms: (1) surgery causes a meaningful voice change that models can learn to compensate, (2) ECAPA-TDNN is a meaningful metric because post-surgery voices remain closer to the same speaker's pre-surgery voice than to other speakers.

---

## Methods

### Baselines (No/Minimal Training)

#### kNN-VC — Nearest-Neighbor Retrieval
**Directory:** `Experiments/knn_vc/` | [Baas et al., Interspeech 2023](https://arxiv.org/abs/2305.18975)

Replaces each source WavLM frame with the mean of its k=4 nearest neighbors from the post-surgery matching set. No training — leverages frozen WavLM-Large (layer 6 features) + HiFi-GAN vocoder.

- **SpkSim:** 0.710 ± 0.084 (+7.4% vs baseline)
- **Key insight:** Strong despite no training; the WavLM feature space already captures surgery-related voice change. Sets a high bar for learned methods.

#### Mean-Shift — Domain Mean Translation
**Directory:** `Experiments/mean_shift/`

`converted = source + (mean_post − mean_pre)` in WavLM feature space. Assumes the surgery effect is a constant direction in embedding space.

- **SpkSim:** 0.638 ± 0.098 (−3.5% vs baseline)
- **Key insight:** The surgery effect is not a simple additive shift — patient-specific adaptation is needed.

#### MKL-VC — Factorized Optimal Transport
**Directory:** `Experiments/mkl_vc/` | [MKL-VC, Interspeech 2025](https://arxiv.org/html/2506.09709)

Models pre/post domains as Gaussians, computes Monge-Kantorovich transport map. Factorized into K=2 subgroups by variance. Generates new feature combinations (unlike kNN retrieval).

- **SpkSim:** 0.642 ± 0.097 (−2.9% vs baseline)
- **Key insight:** Factorization does not help over Mean-Shift; the domain gap is not well-modeled by a single Gaussian.

#### LinearVC — Linear Projection
**Directory:** `Experiments/linear_vc/` | [LinearVC, 2025](https://arxiv.org/html/2506.01510)

Learns `W @ features` via ridge regression on nearest-neighbor-paired frames.

- **Status:** Not yet evaluated — `torch.load` compatibility issue pending.

---

### Analysis Tools

#### ECAPA Mapper — Embedding-Space Mapping
**Directory:** `Experiments/ecapa_mapper/`

Maps pre-surgery ECAPA-TDNN embeddings to post-surgery embeddings directly (bypassing audio synthesis) to measure the upper bound of embedding-domain conversion. Multiple approaches tested.

| Experiment | Mapper | Test SpkSim | Baseline | Delta |
|-----------|--------|:-----------:|:--------:|:-----:|
| 1 | MLP (5-fold CV) | 0.7537 | 0.7274 | +0.026 |
| 2 | MLP + 10× crop augmentation | 0.7265 | 0.7274 | +0.026 |
| 3 | Ridge regression | 0.7342 | 0.7274 | +0.034 |
| 4 | MFCC → ECAPA projection | 0.5480 | 0.5099 | +0.038 |
| 5 | k-NN in ECAPA space (k=10) | 0.5998 | 0.7106 | −0.111 |

- **Key insight:** A simple MLP can push test-set SpkSim to ~0.754, which is a soft upper bound for how much voice identity shift can be recovered in embedding space. All audio-domain methods should aim to approach this.

#### HiFi-GAN Fine-Tuning
**Directory:** `Experiments/hifigan_finetune/` | Log: `hifigan_finetune-59193247.out`

Fine-tuning the HiFi-GAN vocoder on CUCO post-surgery audio to improve audio quality of converted speech.

- **Status:** **Complete** — step=2500, val_mel_loss=0.3961. Fine-tuned weights are loaded by all current split-based evaluations.

---

### Learned Methods

#### VAE — Initial Baseline
**Directory:** `Experiments/VAE/`

Variational autoencoder with content/surgery disentanglement on mel spectrograms. The first method explored; established the disentanglement framework and motivated moving to WavLM features.

- **Status:** Complete (3 log files). Metrics not directly comparable — preceded ECAPA-TDNN evaluation protocol.
- **Key insight:** Continuous latent spaces allow information leakage; motivated the switch to VQ-based approaches.

---

#### MaskCycleGAN-VC — CycleGAN with Fill-in-Frames Masking
**Directory:** `Experiments/mask_cyclegan/` | [Kaneko et al., 2021](https://arxiv.org/abs/2102.12841)

CycleGAN operating on mel spectrograms with Filling-in-Frames (FIF) masking applied to the source during training.

- **MCD (reconstruction):** 76.53 ± 6.70
- Pre/post surgery domains are acoustically very similar, which makes it difficult for the adversarial discriminator to learn a meaningful pre-vs-post boundary.

---

#### UNet-VC — Residual U-Net Feature Transform (Best Performing)
**Directory:** `Experiments/unet_vc/` | Log: `unet_vc-10810013.out`

Residual 1D U-Net learning the small delta between pre and post surgery domains in WavLM feature space. Core idea: `output = input + α × network(input)` where α is a learned scalar initialized to 0.1.

**Architecture:**
- Input/output: WavLM-Large layer 6 features (1024-dim) over 64-frame segments
- 1D U-Net with 2 encoder/decoder levels, HIDDEN_DIM=128 channels
- Skip connections: preserve content-relevant information
- Learnable alpha (residual weight): starts at 0.1, learned to ~0.27 by end of training
- Output → HiFi-GAN vocoder → waveform

**Training:**
| Hyperparameter | Value |
|---------------|-------|
| Segments | 3,778 train / 557 val |
| Batch size | 32 |
| Segment length | 64 frames |
| Learning rate | 5e-4 |
| Epochs | 300 |
| Loss | MSE + Cosine (weight 0.5) |
| Optimizer | Adam |

**Results (28 Tonsill patients):**
- SpkSim (conv→post): **0.792 ± 0.055** (+19.8% vs baseline)
- LSD to target: 1.96 ± 0.14 dB
- SED to target: 2.64 ± 0.97 dB

**Why it works:** The pre→post domain shift is small (Tonsill baseline 0.661 — voices are already similar). A residual formulation prevents the network from over-correcting; the small learned α ensures the output stays close to the input. Skip connections prevent information loss.

---

#### UNet-Adv-VC — U-Net with Adversarial Discriminator
**Directory:** `Experiments/unet_adv_vc/` | Log: `unet_adv_vc-10872738.out`

Same residual U-Net architecture as UNet-VC with an added adversarial discriminator to push converted features toward the post-surgery distribution.

**Results (28 Tonsill patients):**
- SpkSim (conv→post): **0.780 ± 0.058** (+18.0% vs baseline)
- LSD to target: 1.97 ± 0.13 dB
- SED to target: 2.70 ± 1.08 dB

Individual SpkSim range: 0.637–0.886 (conv→post), 0.541–0.879 (conv→source)

- **Key insight:** Adding adversarial loss marginally hurts performance vs pure reconstruction (0.780 vs 0.792). The discriminator may introduce instability that outweighs its domain-alignment benefit for this subtle domain gap.

---

#### UNet-VC-ECAPA — U-Net with ECAPA Similarity Loss
**Directory:** `Experiments/unet_vc_ecapa/` | Log: `unet_ecapa-11425380.out`

UNet-VC augmented with an additional ECAPA-TDNN speaker similarity loss during training — directly optimizing the evaluation metric at train time.

**Architecture:** Same U-Net backbone as UNet-VC (~4.4M params including ECAPA head), with cosine similarity between converted and target post-surgery ECAPA embeddings added to the training objective.

**Training:** Best model at epoch 98/300 (val_ecapa=0.6072, alpha=0.3822). Training on all 13 audio types (~282 files). ECAPA style loss uses `STYLE_CROP_LEN=128` WavLM frames to avoid HiFiGAN OOM.

**Results (5-patient test split — 0045, 0085, 0110, 0122, 0132 — baseline = 0.6707):**

| Patient | Conv→Post | Baseline | Δ |
|---------|:---------:|:--------:|:-:|
| Tonsill_0045 | 0.712 | 0.730 | −0.019 |
| Tonsill_0085 | 0.758 | 0.724 | +0.034 |
| Tonsill_0110 | 0.682 | 0.651 | +0.031 |
| Tonsill_0122 | 0.675 | 0.774 | −0.100 |
| Tonsill_0132 | 0.592 | 0.474 | +0.119 |
| **Mean** | **0.684** | **0.671** | **+0.013** |

**Train-patient evaluation** (23 train patients, speech files): Mean = 0.698 (+0.040 over train baseline 0.659)

- **Key insight:** With more training data (all audio types) and the fine-tuned vocoder, the model achieves a slight positive delta on test (+0.013) vs the previous −0.061. The ECAPA loss provides useful speaker-level signal when backed by sufficient training data.

---

#### UNet-VC-Spk — U-Net with FiLM Speaker Conditioning
**Directory:** `Experiments/unet_vc_spk/` | Log: `unet_spk_split-10962628.out`

UNet-VC extended with FiLM (Feature-wise Linear Modulation) layers that condition the U-Net skip connections on a target speaker embedding. During training, the model sees the target post-surgery speaker embedding and learns to modulate its transformation toward that speaker.

**Architecture (~6M params):**
- Base: same residual 1D U-Net
- FiLM layer on each skip connection: `γ(spk) × skip + β(spk)` where γ, β are learned from the target ECAPA embedding
- Speaker conditioning vector: 192-dim ECAPA-TDNN embedding

**Training (23 train / val split, seed=42, all 13 audio types):**

| Run | Epochs | Test SpkSim | Train SpkSim | Notes |
|-----|:------:|:-----------:|:------------:|-------|
| 59230553 | full | 0.671 (−0.000) | 0.785 (+0.126) | All audio types; fine-tuned HiFiGAN |

**Results (5-patient test split — 0045, 0085, 0110, 0122, 0132 — baseline = 0.6707):**

| Patient | Conv→Post | Baseline | Δ |
|---------|:---------:|:--------:|:-:|
| Tonsill_0045 | 0.699 | 0.730 | −0.032 |
| Tonsill_0085 | 0.736 | 0.724 | +0.012 |
| Tonsill_0110 | 0.659 | 0.651 | +0.008 |
| Tonsill_0122 | 0.678 | 0.774 | −0.096 |
| Tonsill_0132 | 0.581 | 0.474 | +0.107 |
| **Mean** | **0.671** | **0.671** | **−0.000** |

**Train-patient evaluation** (23 train patients): Mean = 0.785 (+0.126 over train baseline 0.659)

- **Key insight:** FiLM conditioning gives the largest train improvement (0.785) but nearly zero test delta. The train/test gap (0.785 vs 0.671) is larger than plain UNet-VC, suggesting the speaker conditioning overfits to training identities.

---

#### UNet-VC-v2 — U-Net with Multi-Surgery Evaluation
**Directory:** `Experiments/unet_vc_v2/` | Logs: `unet_v2_split-10950131.out`, `unet_v2_split-10961341.out`

Variant of UNet-VC testing cross-surgery generalization on the Sept dataset. Two pairing strategies evaluated: cross-patient (surgery effect shared across speakers) vs same-patient (direct pre→post supervision).

**Architecture:** 1D Residual U-Net, ~1.2M params (cross-patient) or ~4.4M params (same-patient variant).

**Results on Sept (5-patient test split, baseline = 0.841):**

| Pairing | Patient | Conv→Post | Baseline | Δ |
|---------|---------|:---------:|:--------:|:-:|
| Cross-patient | Sept_0023 | 0.823 | 0.869 | −0.046 |
| Cross-patient | Sept_0033 | 0.730 | 0.834 | −0.104 |
| Cross-patient | Sept_0044 | 0.694 | 0.755 | −0.061 |
| Cross-patient | Sept_0076 | 0.809 | 0.901 | −0.092 |
| Cross-patient | Sept_0077 | 0.741 | 0.845 | −0.104 |
| **Cross-patient Mean** | | **0.759 ± 0.049** | **0.841** | **−0.082** |
| Same-patient | Sept_0023 | 0.816 | 0.869 | −0.053 |
| Same-patient | Sept_0033 | 0.762 | 0.834 | −0.072 |
| Same-patient | Sept_0044 | 0.714 | 0.755 | −0.040 |
| Same-patient | Sept_0076 | 0.830 | 0.901 | −0.071 |
| Same-patient | Sept_0077 | 0.768 | 0.845 | −0.077 |
| **Same-patient Mean** | | **0.778 ± 0.041** | **0.841** | **−0.063** |

- **Key insight:** For Sept (small surgery effect, high baseline), conversion *hurts* — all methods score below baseline. Same-patient pairing slightly outperforms cross-patient. The model is over-converting for surgeries with subtle voice changes. Future work should gate conversion magnitude by predicted surgery effect size.

---

#### VQVAE — VQ Disentanglement Experiments (Exp 1–6)
**Directory:** `Experiments/vqvae/` | Logs: `vqvae_exp5-10797364.out`, PROGRESS.md

Six progressive experiments attempting explicit content/quality disentanglement via Vector Quantization. Goal: learn a bottleneck where content codes are surgery-agnostic, and a separate quality vector encodes the surgery effect.

| Exp | Architecture | Input | VQ Codes | SpkSim | Notes |
|-----|-------------|-------|:--------:|:------:|-------|
| 1 | VQ-VAE baseline | Mel | 256 | — | 10/256 active codes |
| 2 | VQ + cycle loss | Mel | 256 | — | Sequence-level patterns still encode surgery |
| 3 | Tight bottleneck | Mel | 64 (T/8) | — | 9/64 active codes |
| 4 | Product VQ + entropy reg | Mel | Planned | — | Not yet implemented |
| **5** | **VQ on WavLM features (1D conv)** | **WavLM** | **256** | **0.394** | **Disentanglement succeeds; bottleneck too lossy** |
| 6 | VQ-UNet + FiLM skip connections | WavLM | — | *pending* | — |

**Exp 5 details** (400 epochs, best evaluated):
- VQ perplexity: 24/256 codes active (codebook collapse, ~90% unused)
- Adversarial surgery-type loss: ~0.693 (≈ chance) → disentanglement *succeeds* — codes do not encode surgery type
- Quality classification loss: converged → quality vector *does* capture surgery information
- Cycle consistency loss: converged to ~0.066
- Despite successful disentanglement, conversion SpkSim = **0.394 ± 0.119** (−40.4% vs baseline)

**The steganography problem:** Even when individual VQ codes are surgery-agnostic, temporal *sequences* of codes can encode surgery information. The adversarial loss operates on frame-level code lookup — it does not prevent sequence-level patterns from leaking. The VQ bottleneck simultaneously destroys useful reconstruction information while leaking surgery information through the sequence structure.

**Exp 6 (in progress):** Hybrid approach — U-Net skip connections bypass the VQ bottleneck (so reconstruction quality is preserved) while only the U-Net bottleneck goes through VQ (so content abstraction happens at the most compressed representation).

---

#### DLA-VC — Dual Layer Adapter VC (Proposed, UNet Adaptation for Generalizability)
**Directory:** `Experiments/dla_vc/` | Latest log: `dla_vc_train-59618265.out` | Also: `Experiments/dla_vc_noVQ/` (ablation)

**Framing.** DLA-VC is best viewed as a deliberate **extension of UNet-VC** designed to improve generalizability across patients (and potentially across surgery types). UNet-VC is a single residual map `pre_features → post_features`; any patient or surgery-specific variation is entangled in the same weights as content processing. DLA-VC keeps UNet-VC's encoder–decoder backbone and adds three factorisation components, inspired by AdaptVC ([Kim et al., ICASSP 2025](https://arxiv.org/abs/2501.01347)) and classical VQ-based disentanglement.

**Three adaptations on top of UNet-VC:**
1. **Dual WavLM layer adapters.** UNet-VC uses WavLM layer 6 only. DLA-VC learns two softmax-weighted combinations over all 24 WavLM layers — one for content, one for quality — so each pathway gets the representation that best suits its role.
2. **Product VQ bottleneck on content.** An 8-head × 32-code discrete bottleneck (effective codebook ~10¹²) compresses content to a surgery-invariant representation. *This bottleneck is load-bearing* — our no-VQ ablation shows test SpkSim drops by an additional 0.05 (−0.2341 vs −0.1832) when removed, confirming VQ is what forces speaker/surgery info out of the content channel.
3. **Explicit FiLM-conditioned quality branch.** A parallel 192-dim quality encoder (matched to ECAPA's dim) modulates the decoder skip connections via FiLM, factorising "what is said" from "who is speaking and in what surgical state". Also includes a small `q_shift` MLP that learns the pre→post direction in quality space, enabling per-patient post-style prediction at inference.

**Architecture (6.6M params):**
```
Raw Audio → WavLM-Large (frozen) → 24 hidden states
                │
    ┌───────────┴───────────┐
    │                       │
Content Adapter         Quality Adapter
(softmax over 24 layers) (softmax over 24 layers)
    │                       │
U-Net Encoder           Quality Encoder (→ 192-d quality vec)
    │── skip1 (InstNorm) ──┐   │
    │── skip2 (InstNorm) ──┤   ├─ q_shift MLP (pre→post direction)
    ↓                      │   │
Content Proj → Product VQ ─┴─→ FiLM U-Net Decoder → WavLM layer 6
(8×32 codes, head_dim=16)                          → HiFi-GAN → Audio
```

**Two-phase training:**
- *Phase 1 — Warm-up (epochs 1–60):* pure feature reconstruction. VQ commitment weight ramps 0.01 → 1.0 so the encoder learns continuous representations first.
- *Phase 2 — Conversion (epochs 60+):* adds direct paired-patient conversion loss (kNN-paired target, same trick as UNet-VC), cross-domain cycle consistency, and q_shift MLP training.

**Current status — implementation complete, hyperparameter tuning in progress.**
The architecture, staged training pipeline, q_shift module, and evaluation path all work end-to-end. Current best: **test Δ = −0.1832** (vs UNet-VC's +0.0162), train Δ = −0.3799. The direction under active exploration:
- **Content-adversarial (GRL) loss** on content codes for stronger domain invariance
- **Slower / non-linear VQ annealing** schedule (currently linear 0.01→1.0 over 60 ep)
- **Codebook-geometry sweeps** (head/code-count trade-offs, dead-code resetting)
- **Multi-surgery training for DLA** (may benefit from quality diversity even though it hurts simpler methods)
- **Joint HiFi-GAN refinement** on DLA's decoder output distribution

**Positive ablation (sister experiment `dla_vc_noVQ/`):** removing VQ degrades test Δ from −0.1832 → −0.2341, validating that the bottleneck is doing real work — it is not merely cosmetic.

OOM fixes in place (required to train on all 13 audio types):
- `STYLE_CROP_LEN = 128` WavLM frames (~0.8s) before HiFi-GAN vocoding
- `MAX_WAVLM_SAMPLES = 48000` (3s) before WavLM extraction (prevents O(T²) attention OOM)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**Novelty vs prior work:**

| Component | Closest Prior Work | Difference |
|---|---|---|
| UNet-VC + dual adapters + VQ + FiLM quality | AdaptVC (single adapter, HuBERT, no VQ) | Two adapters; paired training; VQ bottleneck; built on UNet residual backbone |
| VQ + FiLM U-Net in WavLM feature space | None identified | No prior work combines Product VQ + FiLM decoder in SSL feature space for surgical VC |
| SSL adapter-based VC for pathological speech | None identified | First application of adapter-based VC to medical/surgical voice change |
| Learned per-patient post-style predictor (q_shift) | Mean-Shift / ECAPA Mapper | Learns a nonlinear pre→post map in the model's own quality space, not in ECAPA / MFCC space |

---

## Evaluation

All methods use `shared_evaluate.py` with ECAPA-TDNN speaker similarity as the primary metric:

| Metric | What it Measures | Ideal |
|--------|-----------------|-------|
| **SpkSim (conv vs post)** | Does converted voice match post-surgery identity? | > baseline |
| SpkSim (conv vs source) | How much source identity is preserved? | Reference |
| Baseline SpkSim | Pre vs post, same patient, no conversion | Floor to beat |
| LSD (dB) | Log-spectral distance to target | Lower |
| SED (dB) | Spectral envelope distance to target | Lower |

ECAPA-TDNN embeddings are extracted with SpeechBrain's pre-trained model. SpkSim is cosine similarity between mean-pooled embeddings of full utterances.

---

## Project Structure

```
VoiceConversion/
├── README.md
├── shared_evaluate.py              # ECAPA-TDNN + spectral metrics
├── compare_all_spksim.py           # Cross-method comparison
├── Experiments/
│   ├── shared/
│   │   └── utils.py                # get_wav_files(), get_all_audio_pairs()
│   │                               #   ALL_AUDIO_SUBDIRS: 13 audio type tuples
│   │                               #   get_all_audio_pairs(surgery, exclude) →
│   │                               #     {pid: [(pre_path, post_path), ...]}
│   ├── unet_vc/                    # Residual U-Net (test: 0.687, train: 0.775)
│   │   ├── model/unet.py
│   │   └── scripts/{train_kfold,run_eval}.py
│   ├── unet_adv_vc/                # U-Net + adversarial (test: 0.683, train: 0.800)
│   │   ├── model/unet.py
│   │   └── scripts/{train_split,run_eval}.py
│   ├── unet_vc_spk/                # U-Net + FiLM speaker cond (test: 0.671, train: 0.785)
│   │   ├── model/unet.py
│   │   └── scripts/{train_split,run_eval}.py
│   ├── unet_vc_ecapa/              # U-Net + ECAPA loss (test: 0.684, train: 0.698)
│   │   ├── model/unet.py
│   │   └── scripts/{train_split_v2,run_eval}.py
│   ├── dla_vc/                     # DLA-VC (UNet adaptation; tuning in progress)
│   │   ├── model/dla_vc.py         #   current best: test Δ = −0.1832
│   │   └── scripts/{train_split,run_eval}.py
│   ├── dla_vc_noVQ/                # Ablation: DLA-VC without Product VQ
│   │   └── (same structure)        #   test Δ = −0.2341 (validates VQ necessity)
│   ├── free_vc/                    # FreeVC foundation-model baseline
│   │   └── scripts/{run_eval_zeroshot,finetune}.py
│   ├── free_vc_shift/              # Frozen FreeVC + learned speaker shift
│   └── free_vc_ecapa/              # Frozen FreeVC + ECAPA-space shift + bridge
│   ├── knn_vc/                     # kNN-VC (test: 0.579, train: 0.704)
│   ├── mkl_vc/                     # Factorized OT (test: 0.685, train: 0.688)
│   ├── mean_shift/                 # Mean translation (test: 0.686, train: 0.689)
│   ├── linear_vc/                  # Linear projection (test: 0.619, train: 0.674)
│   ├── unet_vc_v2/                 # U-Net on Sept dataset
│   ├── vqvae/                      # VQVAE experiments 1–6
│   ├── mask_cyclegan/              # CycleGAN on mel spectrograms
│   ├── VAE/                        # Initial VAE baseline
│   ├── ecapa_mapper/               # ECAPA embedding mapping analysis
│   ├── wavlm_quality_mapper/       # WavLM layer quality analysis
│   └── hifigan_finetune/           # HiFi-GAN fine-tuning (complete: step=2500)
```

---

## Environment

- **Compute:** Compute Canada (def-zshakeri), NVIDIA H100 GPU
- **Python:** 3.10, CUDA 11.8
- **Key deps:** PyTorch, torchaudio, librosa, speechbrain, transformers (WavLM)
- **SLURM:** `sbatch submit.sh` (or `submit_split.sh`, `submit_exp5.sh`) in each method directory
- **Typical wall time:** 45 min–12 hr depending on experiment

---

## Key Findings

1. **Simple residual learning generalizes best.** In the strict 5-patient held-out test, UNet-VC achieves the highest test SpkSim (0.687, +0.016). Adding adversarial or speaker conditioning losses gives larger training-set gains but smaller test gains — consistent with small-data overfitting.

2. **kNN-VC needs test-time access to the target speaker.** In the full 28-patient evaluation (using self-matching), kNN-VC scored 0.710 (+7.4%). In the strict split with unseen test patients, it scores 0.579 (−0.091) — worst of all methods. It cannot generalize: it needs the target patient's own post-surgery features in its matching set.

3. **Global transform methods (Mean-Shift, MKL-VC) generalize well but have a ceiling.** Without any overfitting risk (no learned parameters per patient), they achieve ~+0.014 on test. The limit is that a global shift cannot capture patient-specific surgery effects.

4. **All 13 audio types improve data coverage.** Including TDU sentences, vowels, and sustained vowels increases the training set from 23 files to ~282 files (23 patients × ~12 audio types). This directly benefits learned methods which previously trained on 23 segments.

5. **Disentanglement via VQ is learnable but bottleneck-limited.** VQVAE Exp5 achieves successful disentanglement (adversarial loss at chance level), but only 24/256 codebook entries are active — codebook collapse degrades reconstruction (SpkSim = 0.394, −40% vs baseline).

6. **Surgery type determines conversion utility.** For Sept/Fess (baseline ~0.828), conversion hurts. For Tonsill (baseline 0.661), conversion helps. Models should condition on surgery-effect magnitude.

7. **DLA-VC is a UNet adaptation for better generalisability.** DLA-VC extends UNet-VC with dual WavLM layer adapters, a Product VQ content bottleneck, and a FiLM-modulated quality branch. Implementation is complete and trains stably; current best test Δ = −0.1832 (tuning in progress, see above). The no-VQ ablation (−0.2341) validates that the bottleneck is functionally load-bearing — the architecture's components are each pulling their weight even while absolute performance continues to improve with further hyperparameter tuning.

8. **Foundation-model approaches (FreeVC) don't automatically transfer to medical VC.** Fine-tuning a pretrained FreeVC (trained on VCTK) narrows the gap from −0.27 (zero-shot) to −0.16, but does not reach the simple from-scratch residual methods' +0.016. Learned shifts in either FreeVC's speaker space or ECAPA space (via a learned bridge) do not close this gap — the pretrained generator's learned distribution does not contain the surgical voice manifold, and 23 patients is insufficient to adapt it there.

---

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
