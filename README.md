# Voice Conversion for Post-Tonsillectomy Speech

Reference-free voice conversion from pre-surgery to post-surgery speech on the CUCO dataset. Given only a patient's pre-surgery audio, predict and synthesise their post-surgery voice — no post-surgery audio of the target patient is used at inference. Pre-surgery audio is mapped to a predicted post-surgery feature representation, then vocoded back to waveform.

This is **not** standard zero-shot any-to-any voice conversion. Zero-shot VC assumes a reference utterance from the target speaker at inference; here we have none. The model must learn the surgical pre→post transformation from a small set of paired training patients and generalise to a new patient at inference, using only that patient's pre-surgery audio.

---

## Final Results

### Setup

- **Dataset:** CUCO Tonsillectomy. 28 patients with paired pre/post surgery recordings.
- **Split:** 5 held-out test patients (`0045, 0085, 0110, 0122, 0132`, seed=42); 23 train/val patients; never overlapping.
- **Vocoder:** stock `bshall/knn-vc` HiFi-GAN (not fine-tuned on CUCO — see "Vocoder finding" below).
- **Metric:** ECAPA-TDNN cosine similarity between converted and post-surgery audio. Δ = converted similarity − pre/post baseline similarity, averaged across the 5 test patients.
- **Test baseline** (pre vs post, no conversion, 5 test patients): **0.6707 ± 0.1061**
- **Train baseline** (pre vs post, no conversion, 23 train patients): **0.6586 ± 0.1248**

### Headline table

| Method | Test Δ | Test Std | Type | Notes |
|---|:---:|:---:|---|---|
| **UNet-VC-ECAPA** | **+0.0635** | 0.043 | Residual U-Net + ECAPA loss | **Best overall; deployment recommendation.** |
| UNet-VC (vanilla) | +0.0525 | 0.031 | Residual U-Net | Strong simple baseline. |
| UNet-VC (+ audio aug) | +0.0493 | 0.026 | Residual U-Net | Augmentation neutral / mildly harmful. |
| **DLA-VC v3 (KD)** | **+0.0450** | 0.032 | Disentangled VC + KD from UNet-VC-ECAPA | **Best DLA-VC variant.** |
| UNet-Adv-VC | +0.0418 | 0.022 | U-Net + adversarial discriminator | Adversarial loss hurts vs vanilla. |
| UNet-VC-Spk | +0.0357 | 0.027 | U-Net + FiLM speaker conditioning | Conditioning helps train, hurts test (overfit). |
| DLA-VC v3b (KD + low λ_recon) | +0.0240 | 0.035 | KD variant with weaker recon anchor | Better val, worse test — small-N val noise. |
| MKL-VC | +0.0120 | 0.062 | Training-free factorized OT | Best of the training-free methods. |
| Mean-Shift | +0.0103 | 0.068 | Training-free global shift | Tied with MKL-VC. |
| LinearVC | +0.0091 | 0.039 | Linear projection (ridge) | Marginal but positive. |
| DLA-VC v2 | −0.0668 | 0.082 | DLA-VC without KD | Pre-KD architectural baseline — fails. |
| kNN-VC | −0.0483 | 0.055 | Nearest-neighbor retrieval | Needs target post audio; degrades on unseen patients. |

### Per-patient results for the top three methods

Test set (pre↔post baseline shown; "Δ" = converted − baseline):

| Patient | Baseline | UNet-VC-ECAPA Δ | UNet-VC Δ | DLA-VC v3 Δ |
|---|:---:|:---:|:---:|:---:|
| 0045 | 0.7303 | +0.0636 | +0.0092 | −0.0465 |
| 0085 | 0.7238 | +0.0084 | −0.0066 | −0.0343 |
| 0110 | 0.6513 | +0.1110 | +0.1140 | +0.0822 |
| 0122 | 0.7743 | −0.0608 | −0.0507 | −0.1149 |
| 0132 | 0.4736 | +0.1952 | +0.1969 | +0.2038 |
| **Mean** | **0.6707** | **+0.0635** | **+0.0525** | **+0.0450** |

Across methods, patient 0132 consistently gains the most (low baseline → large headroom) and patient 0122 consistently loses (already-high baseline). This inter-patient heterogeneity dominates the std and is itself a clinical-relevance finding.

### Methodological finding: fine-tuned HiFi-GAN overfits

Fine-tuning HiFi-GAN on CUCO post-surgery audio improved validation mel loss but **degraded held-out test ECAPA Δ** across every method. Example: UNet-VC dropped from **+0.049 (stock vocoder)** to **+0.015 (fine-tuned on 23 patients)** on the same checkpoints. On 23 patients the vocoder memorises the training distribution; held-out patients fall outside it. **All numbers above use the stock vocoder for a fair comparison.**

---

## Reference-free counterfactual conversion

The inference setting here is closer to **counterfactual VC** than to standard VC:

| Setting | Reference at inference | Task |
|---|---|---|
| Standard zero-shot VC ([Gusev 2024](https://www.isca-archive.org/interspeech_2024/gusev24_interspeech.html), kNN-VC) | Target speaker utterance | Any-to-any speaker conversion |
| Dysarthric→healthy ([Halpern 2025](https://arxiv.org/html/2501.10256v1)) | Fixed healthy template (e.g. LJSpeech) | Patient → single fixed healthy template |
| **This work** | **None** — pre audio only | **Predict same patient's own post-surgery voice without seeing it** |

Both UNet-VC-ECAPA and DLA-VC operate in this reference-free regime. UNet-VC-ECAPA does so implicitly (the network learns a fixed pre→post translator). DLA-VC does so explicitly, via a `q_shift` MLP that maps a pre-surgery quality embedding to a predicted post-surgery quality embedding, which then conditions a FiLM decoder. This explicit decomposition is what makes the DLA-VC contribution architectural rather than just empirical.

---

## Dataset

**CUCO Dataset** — Paired pre/post-surgery recordings:
- **28 Tonsillectomy patients** (focus of this work)
- Also available: Fess (27 patients), Sept (32 patients), Contr/control (28 patients)
- Audio: resampled to 16 kHz mono
- All 13 audio types used in training: Speech, 4 TDU sentence types, 5 vowels, 3 sustained vowels (23 train patients × ~12 files ≈ **282 paired training files**)
- Test split (fixed, seed=42): patients `[0045, 0085, 0110, 0122, 0132]` held out — never used in any training stage.

### Baselines across surgery types

| Condition | Baseline SpkSim | n | Interpretation |
|---|:---:|:---:|---|
| Contr (control) | 0.886 ± 0.056 | 28 | No surgery — session variability only |
| Fess | 0.828 ± 0.076 | 27 | Mild surgery effect |
| Sept | 0.828 ± 0.065 | 32 | Mild surgery effect |
| **Tonsill** | **0.661 ± 0.118** | **28** | **Largest surgery effect — most headroom for conversion** |

### Voice change vs speaker identity (preliminary analysis)

`Experiments/wavlm_quality_mapper/`

| Comparison | SpkSim |
|---|:---:|
| Pre vs Pre (same speaker, cross-session) | 0.780 ± 0.080 |
| Pre vs Post (surgery effect) | 0.529 ± 0.114 |
| Between-speaker | 0.196 ± 0.117 |

Voice change magnitude: **0.251** (pre-pre minus pre-post). Speaker discrimination: **0.333** (pre-post minus between-speaker). Surgery introduces a substantial measurable shift while speaker identity is largely preserved — both signals that ECAPA-TDNN is a meaningful target metric.

---

## Methods

### Training-free baselines

#### kNN-VC — Nearest-Neighbour Retrieval
`Experiments/knn_vc/` | [Baas et al., Interspeech 2023](https://arxiv.org/abs/2305.18975)

Replaces each source WavLM frame with the mean of its k=4 nearest neighbours from a matching set built from post-surgery audio. No training. **Test Δ = −0.0483.**

**Why it fails on this task:** kNN-VC needs the target patient's own post-surgery features in the matching set. With held-out test patients, the matching set contains only *other* patients' post audio — so kNN-VC converts toward a generic post-surgery voice rather than the specific patient's predicted post voice. This is the canonical example of a reference-dependent method failing in our reference-free setting.

#### Mean-Shift — Global Domain Translation
`Experiments/mean_shift/`

`converted = source + (mean_post − mean_pre)` in WavLM feature space. No training. **Test Δ = +0.0103.**

#### MKL-VC — Factorised Optimal Transport
`Experiments/mkl_vc/` | [MKL-VC, Interspeech 2025](https://arxiv.org/html/2506.09709)

Pre/post domains modelled as factorised Gaussians; Monge–Kantorovich transport map. **Test Δ = +0.0120.**

#### LinearVC — Linear Projection
`Experiments/linear_vc/` | [LinearVC, 2025](https://arxiv.org/html/2506.01510)

Ridge regression on nearest-neighbour-paired frames. **Test Δ = +0.0091.**

---

### Learned methods — UNet-VC family

All four UNet-VC variants share a 1D residual U-Net operating in WavLM-Large layer-6 feature space (~1024 dim), with `output = input + α · network(input)` and a learnable scalar α. Vocoded through HiFi-GAN.

#### UNet-VC (vanilla)
`Experiments/unet_vc/` — **Test Δ = +0.0525, Train Δ = +0.116.**

Single learning signal: paired-patient feature reconstruction. 5-fold CV-trained, 300 epochs.

#### UNet-Adv-VC
`Experiments/unet_adv_vc/` — **Test Δ = +0.0418.**

Adds an adversarial discriminator. The discriminator marginally **hurts** vs vanilla UNet-VC on test (+0.042 vs +0.053). The small pre↔post domain gap means adversarial pressure pulls outputs out of the post manifold more than it pushes them toward it.

#### UNet-VC-Spk
`Experiments/unet_vc_spk/` — **Test Δ = +0.0357.**

FiLM-modulated skip connections conditioned on a target ECAPA speaker embedding. Highest **train** Δ (+0.126) but lowest **test** Δ — the speaker conditioning overfits to the 23 training identities.

#### UNet-VC-ECAPA (best)
`Experiments/unet_vc_ecapa/` — **Test Δ = +0.0635, Std = 0.043. Best overall.**

UNet-VC with a differentiable ECAPA-TDNN similarity loss: every N=3 steps, the model is penalised for `1 − cos(ECAPA(vocode(converted)), ECAPA(post))`. Directly optimises the evaluation metric at training time.

Training stops on val ECAPA plateau (epoch 68 / 300 budget; longer training rebounds toward identity).

---

### DLA-VC — Disentangled-Latent VC (proposed)

`Experiments/dla_vc_v3/` — **Test Δ = +0.0450, Std = 0.032. Best DLA-VC variant.**

**Framing.** DLA-VC is an architectural exploration: rather than learning a single black-box pre→post translator (UNet-VC), DLA-VC factorises the conversion into separate **content** and **quality** pathways, plus an explicit **q_shift** module that predicts the patient's post-surgery quality embedding from their pre-surgery quality embedding. The architecture is designed for reference-free counterfactual conversion *and* for interpretable / interpolatable / controllable use cases downstream (gradual rehab exposure, pre-op triage, cross-patient transfer).

#### Architecture (6.6M params)
```
Raw audio → WavLM-Large (frozen) → 24 hidden states
                  │
   ┌──────────────┴──────────────┐
   │                             │
 Content adapter            Quality adapter
 (softmax weights           (softmax weights
  over 24 layers)            over 24 layers)
   │                             │
 Content encoder           Quality encoder
   │                             │
 Product VQ           ┌───────── pooled quality vector q
 (8 heads × 32 codes) │            │
   │                  │       q_shift MLP (pre → predicted-post)
   │                  │            │
   ↓                  └─→ FiLM ←───┘
 U-Net decoder ←─────────────────────
   ↓
 WavLM-L6-shape features → stock HiFi-GAN → waveform
```

#### Key mechanisms
1. **Dual WavLM-layer adapters** — separate softmax weightings over the 24 WavLM layers for content vs quality. Different parts of WavLM encode linguistic content vs voice quality; let each pathway pick its layers.
2. **Product VQ on content** (8 heads × 32 codes, effective codebook size ~10¹²) — forces content through a discrete bottleneck to strip residual speaker / quality information.
3. **FiLM-modulated decoder** — quality vector γ-β-modulates the decoder; content and quality never mix at the encoder.
4. **`q_shift` MLP** — at inference, given only pre audio, encode `q_pre` and predict `q_post = q_shift(q_pre)`. This is what makes inference reference-free.
5. **Gradient reversal** — adversarial term penalising the content encoder for retaining speaker-discriminative information.

#### Knowledge distillation from UNet-VC-ECAPA
The decisive change in v3 over v2 (−0.067 → +0.045, a +0.112 swing) was adding a frozen UNet-VC-ECAPA teacher and an MSE+cosine distillation loss between the DLA-VC student's output and the teacher's output on the same input WavLM features (`LAMBDA_KD = 5.0`). Distillation gave the student a dense, clean per-frame target that its own competing losses (recon, conv, VQ, q_shift) could not.

#### v3b ablation (`results_v3b/`)
Reducing `LAMBDA_RECON` from 5.0 → 2.0 to free the model from identity attraction yielded a better val ECAPA (0.476 vs v3's 0.508) but **worse test Δ** (+0.024 vs +0.045) — a clean cautionary example of small-N val noise (3 val patients vs 5 test patients). Final DLA-VC number uses v3.

#### Why UNet-VC-ECAPA still wins
- With only 23 training patients, the VQ bottleneck, adversarial gradient reversal, and disentanglement losses all fight against the small training signal. UNet-VC's single-objective end-to-end design is more sample-efficient here.
- The argument *for* DLA-VC is therefore not the headline Δ but the **structural capabilities it unlocks**: a directly inspectable q_pre → q_post mapping, the ability to interpolate conversion strength (`q_blend = α·q_post + (1−α)·q_pre`), and the ability to extend to new surgery types by fine-tuning only the `q_shift` MLP.

---

### Foundation-model baselines — FreeVC family

`Experiments/free_vc/`, `Experiments/free_vc_shift/`, `Experiments/free_vc_ecapa/` | [Li et al., ICASSP 2023](https://arxiv.org/abs/2210.15418)

Pretrained FreeVC (VCTK) evaluated zero-shot and with three adaptation strategies. All variants land well below the simpler from-scratch methods (test Δ between −0.27 and −0.16). The pretrained generator's learned distribution does not contain the surgical voice manifold and 23 patients is insufficient to adapt it.

---

### Analysis tools

#### ECAPA Mapper — embedding-space upper bound
`Experiments/ecapa_mapper/`

Maps pre-surgery ECAPA embeddings directly to post-surgery embeddings, bypassing audio synthesis. Provides a soft upper bound for how much of the surgical identity shift can be recovered in embedding space alone. Best result: MLP, test SpkSim = 0.754 vs baseline 0.727 (Δ = +0.027 in pure ECAPA space).

#### HiFi-GAN fine-tuning
`Experiments/hifigan_finetune/` (step=2500, val_mel_loss=0.396) — **abandoned in final results**. The fine-tuned vocoder improved on-train fidelity but degraded held-out test ECAPA Δ across every method. Final pipeline uses the stock `bshall/knn-vc` HiFi-GAN.

---

## Evaluation

Primary metric: **ECAPA-TDNN cosine similarity** between converted audio and the patient's real post-surgery audio. Δ = converted similarity − baseline (pre vs post) similarity, averaged over the 5 held-out test patients.

| Metric | What it measures | Direction |
|---|---|---|
| **Test Δ (5 patients)** | Mean ECAPA gain over baseline on held-out patients | Higher is better |
| Test Std | Inter-patient variability | Lower = more consistent |
| Train Δ (23 patients) | Same metric on training patients | Diagnostic — large train/test gaps signal overfit |

Embeddings extracted with SpeechBrain's `spkrec-ecapa-voxceleb`. Cosine similarity between mean-pooled embeddings of full utterances. All methods use the same `shared/utils.py` audio loaders, the same 5 test patients, and the stock HiFi-GAN.

### Caveat — N=5 test patients

The headline Δ on N=5 has wide CIs. UNet-VC-ECAPA's +0.0635 with std 0.043 corresponds to a paired 95% CI roughly [+0.010, +0.117]. Patient 0132 dominates the mean for every learned method. The mean is positive and consistent across methods, but **inter-patient variance is the dominant signal** — some patients benefit substantially (0.108, 0.132), some do not (0.122, 0.045).

### Missing evaluation axis

This work reports only **objective** ECAPA-TDNN similarity. The standard companion evaluation for voice conversion (subjective MOS / ABX listening test, e.g. N=15–20 listeners) is **not** included. Future work should add this — every recent VC paper at Interspeech / ICASSP includes one.

---

## Project structure

```
VoiceConversion/
├── README.md
├── shared_evaluate.py              # ECAPA-TDNN + spectral metrics
├── compare_all_spksim.py           # Cross-method comparison
├── Experiments/
│   ├── shared/utils.py             # get_all_audio_pairs, FORCE_STOCK_VOCODER, etc.
│   ├── unet_vc/                    # Vanilla UNet-VC               (test Δ +0.0525)
│   ├── unet_adv_vc/                # + adversarial                  (test Δ +0.0418)
│   ├── unet_vc_spk/                # + FiLM speaker conditioning    (test Δ +0.0357)
│   ├── unet_vc_ecapa/              # + ECAPA loss (best learned)    (test Δ +0.0635)
│   ├── dla_vc/                     # DLA-VC v1 (deprecated)
│   ├── dla_vc_v2/                  # DLA-VC v2 (pre-KD)             (test Δ −0.0668)
│   ├── dla_vc_v3/                  # DLA-VC v3 (KD, FINAL)          (test Δ +0.0450)
│   │   ├── model/dla_vc.py
│   │   ├── scripts/{train_split,run_eval}.py
│   │   ├── results_v3/             #   λ_recon=5.0 → +0.0450
│   │   └── results_v3b/            #   λ_recon=2.0 → +0.0240 (ablation)
│   ├── free_vc/                    # Pretrained FreeVC baselines
│   ├── free_vc_shift/, free_vc_ecapa/
│   ├── knn_vc/                     # kNN-VC                          (test Δ −0.0483)
│   ├── mean_shift/                 # Mean translation                (test Δ +0.0103)
│   ├── mkl_vc/                     # Factorised OT                   (test Δ +0.0120)
│   ├── linear_vc/                  # Ridge linear                    (test Δ +0.0091)
│   ├── unet_vc_v2/                 # Cross-surgery (Sept, Fess)
│   ├── vqvae/                      # VQ disentanglement experiments
│   ├── mask_cyclegan/              # CycleGAN on mel
│   ├── VAE/                        # Initial VAE baseline
│   ├── ecapa_mapper/               # ECAPA embedding-space analysis
│   ├── wavlm_quality_mapper/       # WavLM layer / surgery analysis
│   └── hifigan_finetune/           # (abandoned in final pipeline)
└── launch_all_stock_retrains.sh    # Re-eval everything with stock HiFi-GAN
```

Most experiments are submitted via `sbatch submit_train_eval.sh` from their respective directory. SLURM template:
- `--account=def-zshakeri`
- A100 (full 40GB for DLA-VC; 20GB MIG slice for evals and lighter trainings)
- 4–12 hours wall time depending on model

---

## Key findings

1. **UNet-VC-ECAPA wins.** Adding an ECAPA speaker-similarity loss to a residual U-Net yields the largest test-set ECAPA Δ across all methods (+0.0635, std 0.043). Directly optimising the evaluation metric works when paired with sufficient training data (all 13 audio types).

2. **Fine-tuned HiFi-GAN overfits on N=23.** Vocoders fine-tuned on the 23 training patients improve validation mel loss but hurt held-out test ECAPA Δ across *every* method. The fine-tuned generator memorises the training-patient distribution; held-out patients fall outside it. **All headline numbers use the stock vocoder.** This is a methodological lesson for any future medical-VC work on small datasets.

3. **Reference-dependent methods fail under reference-free conditions.** kNN-VC (+0.071 in self-matched evaluation) collapses to −0.048 when the matching set cannot contain the target patient's own post audio. This is the fundamental limitation of nearest-neighbour-style VC for prospective clinical use.

4. **Simple ≥ complex on N=23.** UNet-VC (single objective) beats UNet-Adv-VC (+adversarial), UNet-VC-Spk (+speaker conditioning), and DLA-VC (full disentanglement) on test. Adversarial pressure, conditioning networks, and bottlenecks all add overfit risk that the small training set cannot absorb. Test/train gap is largest for the most complex variants.

5. **DLA-VC's contribution is structural, not numerical.** DLA-VC v3 lands at +0.045 — competitive but behind UNet-VC-ECAPA. The architectural value is not the headline metric but the **explicit, inspectable q_pre → q_post mapping** that enables capabilities black-box translators cannot offer: gradual conversion (interpolated quality vectors), per-patient shift-magnitude prediction (`‖q_post − q_pre‖`), and cross-condition transfer by fine-tuning only the `q_shift` MLP. Numbers improve substantially with the right training signal — adding KD from UNet-VC-ECAPA shifted DLA-VC from −0.067 (v2) to +0.045 (v3), a +0.112 gain from a single additional loss term.

6. **Inter-patient heterogeneity dominates.** Across every learned method, patient 0132 gains +0.20 (low baseline → large headroom) while 0122 loses −0.05 to −0.13 (already-high baseline). The mean Δ hides this. Future work should report per-patient analysis and consider gating conversion strength by predicted shift magnitude.

7. **The Sept / Fess surgeries show that not every condition benefits.** For surgeries with small baseline domain gaps (Sept baseline 0.828), all conversion methods score *below* baseline — the model "over-converts". Conversion-utility is conditional on surgery-effect magnitude.

8. **The task is reference-free counterfactual VC, not zero-shot VC.** Standard zero-shot VC ([Gusev 2024](https://www.isca-archive.org/interspeech_2024/gusev24_interspeech.html), kNN-VC, FreeVC) requires a target speaker reference utterance at inference. Here, the test patient's post-surgery audio is *not* available — the model must predict it. Directly comparable published numbers in this regime are scarce; the closest analog ([Halpern 2025](https://arxiv.org/html/2501.10256v1), dysarthric→healthy) reports only WER, not speaker similarity. The +0.0635 ECAPA gain is therefore not directly comparable to any zero-shot VC table in the recent literature.

---

## References

- [Baas et al., Interspeech 2023 — kNN-VC](https://arxiv.org/abs/2305.18975)
- [Gusev & Avdeeva, Interspeech 2024 — Zero-Shot Any-to-Any VC](https://www.isca-archive.org/interspeech_2024/gusev24_interspeech.html)
- [Halpern et al., 2025 — Unsupervised Rhythm and Voice Conversion of Dysarthric to Healthy Speech](https://arxiv.org/html/2501.10256v1)
- [Kim et al., ICASSP 2025 — AdaptVC](https://arxiv.org/abs/2501.01347)
- [MKL-VC, Interspeech 2025](https://arxiv.org/html/2506.09709)
- [LinearVC, 2025](https://arxiv.org/html/2506.01510)
- [Li et al., ICASSP 2023 — FreeVC](https://arxiv.org/abs/2210.15418)
- [Vevo, ICLR 2025](https://arxiv.org/abs/2502.07243)
- [Kaneko et al., 2021 — MaskCycleGAN-VC](https://arxiv.org/abs/2102.12841)
- [Wang et al., Interspeech 2021 — VQMIVC](https://arxiv.org/abs/2106.10132)
- [Qian et al., ICML 2019 — AutoVC](https://arxiv.org/abs/1905.05879)
- [Perez et al., 2018 — FiLM](https://arxiv.org/abs/1709.07871)
- [van den Oord et al., 2017 — VQ-VAE](https://arxiv.org/abs/1711.00937)
- [Hinton et al., 2015 — Knowledge Distillation](https://arxiv.org/abs/1503.02531)
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- [Deep Learning for Pathological Speech, 2025 — Survey](https://arxiv.org/html/2501.03536v1)

---

## Environment

- **Compute:** Compute Canada (def-zshakeri), NVIDIA A100 (full 40GB and `a100_4g.20gb` MIG slices)
- **Python:** 3.10, CUDA 11.8
- **Key deps:** PyTorch, torchaudio, librosa, speechbrain, transformers (WavLM-Large)
- **Submission:** `sbatch submit_train_eval.sh` in each experiment directory; `FORCE_STOCK_VOCODER=1` to force stock HiFi-GAN
- **Wall time:** eval-only runs ~5–15 min; full DLA-VC training ~6–10 hr on a full A100
