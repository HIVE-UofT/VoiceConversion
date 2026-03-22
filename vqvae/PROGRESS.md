# VQVAE Voice Conversion — Progress Log


## Architecture Overview

The VQVAE disentangles voice into two independent representations:

- **Content codes** (VQ-quantized): discrete tokens representing *what* is being said — phonemes, timing, pitch. The vector quantization bottleneck strips voice quality by forcing content through a discrete dictionary.
- **Quality vector** (continuous, 32-dim): captures *how* it sounds — resonance, nasality, vocal tract shape changes from surgery.

At inference, conversion is done by encoding content from the source speaker and swapping in the target domain's average quality vector.

### Losses

| Loss | Purpose |
|------|---------|
| Reconstruction (L1) | Decoded output must match input mel |
| VQ Commitment | Encoder output stays close to codebook entries |
| Multi-Res STFT | Preserves harmonic/spectral detail (prevents blurry output) |
| Adversarial (gradient reversal) | Content must NOT predict pre/post surgery (want ~50% accuracy) |
| Quality Classification (BCE) | Quality vector MUST predict pre/post surgery (want ~100% accuracy) |
| Cycle (cross-reconstruction) | Swap quality between domains, re-encode, verify content + quality preserved |

---

## Experiment 1: Baseline VQVAE (no cycle loss)

**Date:** 2026-03-22
**Config:** code_dim=64, num_codes=256, quality_dim=16, commitment_weight=0.25, batch_size=8, epochs=300, lr=2e-4
**Job:** `vqvae_vc-10745528`

### Training Progression

| Epoch | Recon | VQ | Perplexity | Adv (want ~0.693) | Qual (want low) | Val Recon |
|-------|-------|----|------------|--------------------|--------------------|-----------|
| 1 | 0.3250 | 0.0089 | 5 | 0.6965 | 0.6935 | 0.2976 |
| 10 | 0.0784 | 0.0139 | 4 | 0.6936 | 0.6808 | 0.0764 |
| 50 | 0.0502 | 0.0075 | 7 | 0.6884 | 0.5747 | 0.0463 |
| 100 | 0.0478 | 0.0064 | 7 | 0.6876 | 0.3555 | 0.0465 |
| 150 | 0.0462 | 0.0035 | 8 | 0.6940 | 0.1917 | 0.0420 |
| 200 | 0.0440 | 0.0028 | 8 | 0.6915 | 0.1301 | 0.0401 |
| 250 | 0.0440 | 0.0026 | 9 | 0.6934 | 0.0705 | 0.0397 |
| 300 | 0.0433 | 0.0024 | 10 | 0.6929 | 0.0598 | 0.0390 |

**Best val recon loss: 0.0390**

### Analysis

**What worked well:**

1. **Disentanglement is strong.** The adversarial loss stayed at ~0.693 throughout all 300 epochs — the domain classifier on content features is at random chance. This means the content encoder successfully learned to NOT encode surgery-related information.

2. **Quality classification works.** Loss dropped from 0.693 (random chance at epoch 1) to 0.060 (near-perfect prediction). The quality encoder reliably separates pre from post surgery.

3. **Reconstruction converged.** Train/val gap is small (0.043 vs 0.039), indicating no overfitting. The model reconstructs mel-spectrograms reasonably well.

**What failed:**

1. **Codebook collapse (critical).** Perplexity only reached 10 out of 256 possible codes. This means 246 codebook entries are dead — unused. With only 10 codes to represent all speech content, the model can't capture the diversity of phonemes and sounds. Reconstructions lose fine detail.

2. **Conversion is near-identity.** The mel-spectrogram plots at epoch 300 show that "Pre→Post (converted)" looks nearly identical to "Pre (real)". The difference maps show some change, but it's very small. The quality swap has minimal effect on the output.

3. **Reconstruction is blurry.** Comparing real vs reconstructed mel-spectrograms, the horizontal harmonic striping patterns are smoother in the reconstruction. Fine spectral detail is lost.

### Root Cause: No Cross-Domain Training Signal

The fundamental issue is that during training, the model only ever reconstructs **the same sample** it was given:

```
pre_surgery_mel → encode → decode → compare with pre_surgery_mel
```

The decoder always receives *matched* content and quality (both from the same sample). It never sees the combination of "pre-surgery content + post-surgery quality" during training. So when we swap quality at inference, the decoder is operating on an input distribution it has never been trained on.

The decoder learned to reconstruct well from matched pairs, but it has no incentive to meaningfully use a *different* quality vector — it can get low reconstruction loss by relying mostly on the content codes and treating the quality vector as a minor conditioning signal.

### What This Inspired

This analysis motivated three changes for Experiment 2:

1. **Cycle (cross-reconstruction) loss** — The most important fix. During training, we now explicitly swap quality vectors between domains:
   - Take pre-surgery content + post-surgery quality → decode → re-encode
   - Verify the re-encoded content matches original pre-surgery content
   - Verify the re-encoded quality matches the post-surgery quality we injected

   This forces the decoder to actually use the quality vector correctly, even when it comes from a different domain than the content.

2. **Dead code reset** — Periodically detect codebook entries that are unused (dead) and reinitialize them from random encoder outputs. This should push perplexity from 10 toward 50-100+.

3. **Quality dropout** — Randomly zero out the quality vector 30% of the time during training. When quality is missing, the decoder must reconstruct without it. When quality is present, the decoder learns to actually *use* it rather than ignoring it.

---

## Experiment 2: VQVAE + Cycle Loss + Fixes

**Changes from Experiment 1:**
- Added cycle (cross-reconstruction) loss (`LAMBDA_CYCLE=5.0`)
- Added dead code reset in VQ layer (every 50 forward passes, reset codes used < 2 times)
- Increased commitment weight: 0.25 → 1.0
- Increased quality_dim: 16 → 32
- Added quality dropout in decoder (30% rate)
- Added residual blocks in decoder (2x ResBlock2d after upsampling)
- Increased quality classification weight: 1.0 → 2.0
- Domain-specific data loaders for guaranteed paired batches
- batch_size: 8 → 4 (processing 2 domains per step)
- epochs: 300 → 400

**Job:** `vqvae_vc-10753904`

### Training Progression

| Epoch | Recon | VQ | Perp | Cycle | Adv (want ~0.693) | Qual (want low) | Val Recon |
|-------|-------|----|------|-------|--------------------|-----------------|-----------|
| 1 | 0.2650 | 0.0335 | 6 | 0.2227 | 0.6923 | 0.6951 | 0.2091 |
| 10 | 0.0609 | 0.0120 | 13 | 0.1006 | 0.6883 | 0.6839 | 0.0560 |
| 50 | 0.0470 | 0.0039 | 13 | 0.0951 | 0.6907 | 0.6028 | 0.0408 |
| 100 | 0.0413 | 0.0019 | 15 | 0.0889 | 0.6863 | 0.4544 | 0.0377 |
| 150 | 0.0403 | 0.0019 | 15 | 0.0890 | 0.6857 | 0.2678 | 0.0360 |
| 200 | 0.0394 | 0.0016 | 15 | 0.0805 | 0.6866 | 0.1408 | 0.0357 |
| 250 | 0.0385 | 0.0016 | 15 | 0.0667 | 0.6882 | 0.0796 | 0.0340 |
| 300 | 0.0382 | 0.0015 | 15 | 0.0662 | 0.6876 | 0.0532 | 0.0342 |
| 350 | 0.0380 | 0.0015 | 15 | 0.0644 | 0.6859 | 0.0347 | 0.0344 |
| 400 | 0.0378 | 0.0015 | 15 | 0.0612 | 0.6914 | 0.0344 | 0.0337 |

**Best val recon loss: 0.0337**

### Analysis: What Improved

1. **Reconstruction is better.** Val recon improved from 0.0390 (Exp 1) to 0.0337 — a 14% improvement. The residual blocks in the decoder are helping produce sharper output.

2. **Disentanglement still strong.** Adversarial loss stays at ~0.69, quality classification drops to 0.034. Both working as intended.

3. **Cycle loss is decreasing.** Went from 0.22 → 0.06 over 400 epochs, meaning the cross-reconstructed output increasingly preserves both content and quality. The model is learning to use the swapped quality vector.

4. **Perplexity improved slightly.** 10 → 15. Better than before but still far from the 50-100+ target.

### Analysis: What Still Fails

1. **Codebook collapse is still severe.** Perplexity stuck at 15 out of 256. The dead code reset is firing but not enough codes are surviving — they get reset, briefly used, then go dead again. The encoder is still finding a way to encode most information into a small cluster of codes.

2. **Conversion still looks like near-identity.** The epoch 400 mel-spectrogram plots show the "Pre→Post (converted)" output is nearly identical to "Pre (real)". The difference maps (row 3, columns 3-4) show some structured change (blue/red patterns), which is an improvement over Exp 1 where changes were minimal. But the effect is still too subtle to be a meaningful voice conversion.

3. **Cycle loss plateaued at 0.06.** It stopped decreasing meaningfully after epoch 250. The model may have found a local minimum where content is roughly preserved but quality transfer is still weak.

### Root Cause Analysis

The core problem is that the **VQ bottleneck is too leaky**. With 15 codes × 100 time steps, the content representation has enough capacity to encode nearly everything about the input — including voice quality information — through the *pattern* of code selections across time, even though each individual code is surgery-agnostic (passes the adversarial test).

Think of it as steganography: each code is "innocent" on its own, but the sequence of codes encodes hidden information. The adversarial classifier tests each time step independently (via average pooling) and misses sequence-level patterns.

This means:
- The quality vector is redundant — the decoder can reconstruct from content alone
- Swapping quality has minimal effect because the decoder learned to ignore it
- The cycle loss helps but can't fully overcome this if the bottleneck isn't tight enough

### What Should Change for Experiment 3

1. **Tighter VQ bottleneck** — Reduce content temporal resolution. Currently T'=T/4=100 time steps. At 100 steps × 15 active codes, there's too much sequential capacity. Downsampling further to T/8 or T/16 (50 or 25 steps) would force the codebook to carry more per-code and reduce the ability to encode voice quality in the sequence pattern.

2. **Smaller codebook** — Reduce from 256 to 64 codes. Counterintuitive, but with 256 codes and only 15 active, the model self-selects a tiny subset. A smaller codebook that's fully utilized (perplexity ~64/64) is better than a large one that's 94% dead.

3. **Stronger adversarial classifier** — Replace the simple Conv1d + AvgPool classifier with a temporal model (small GRU/LSTM) that can detect sequence-level surgery information, not just per-frame. This would close the steganography loophole.

4. **Information bottleneck on content** — Add noise or dropout directly to the quantized content codes before decoding. This degrades the sequential pattern, forcing the decoder to rely more on the quality vector for domain-specific reconstruction.

---

## Experiment 3: Tighter Bottleneck

**Date:** 2026-03-22
**Changes from Experiment 2:**
- Content temporal downsampling: T/4 → T/8 (50 time steps instead of 100)
- Codebook size: 256 → 64
- GRU-based adversarial classifier (bidirectional, catches sequence-level patterns)
- Content noise injection (std=0.1) before decoder to degrade sequential patterns
- Decoder upsampling: 3-stage 8x to match T/8 encoder
- All Exp 2 changes kept (cycle loss, quality dropout, residual decoder)

**Job:** `vqvae_vc-10756936`

### Training Progression

| Epoch | Recon | VQ | Perp | Cycle | Adv (want ~0.693) | Qual (want low) | Val Recon |
|-------|-------|----|------|-------|--------------------|-----------------|-----------|
| 1 | 0.3567 | 0.0395 | 5 | 0.2035 | 0.6934 | 0.6940 | 0.2132 |
| 10 | 0.1122 | 0.0146 | 8 | 0.0872 | 0.6925 | 0.6851 | 0.1024 |
| 50 | 0.0653 | 0.0064 | 8 | 0.0929 | 0.6857 | 0.5912 | 0.0648 |
| 100 | 0.0529 | 0.0036 | 9 | 0.0942 | 0.6787 | 0.4334 | 0.0486 |
| 150 | 0.0510 | 0.0026 | 8 | 0.0836 | 0.6828 | 0.2137 | 0.0467 |
| 200 | 0.0490 | 0.0039 | 8 | 0.0937 | 0.6770 | 0.1530 | 0.0446 |
| 250 | 0.0482 | 0.0035 | 8 | 0.0764 | 0.6772 | 0.0650 | 0.0440 |
| 300 | 0.0473 | 0.0038 | 9 | 0.0778 | 0.6817 | 0.0474 | 0.0434 |
| 350 | 0.0473 | 0.0044 | 9 | 0.0694 | 0.6808 | 0.0256 | 0.0435 |
| 400 | 0.0470 | 0.0040 | 9 | 0.0691 | 0.6812 | 0.0182 | 0.0433 |

**Best val recon loss: 0.0430**

### Comparison with Experiment 2

| Metric | Exp 2 Final | Exp 3 Final | Change |
|--------|-------------|-------------|--------|
| Val Recon | 0.0337 | 0.0433 | **Worse** (+28%) |
| Perplexity | 15/256 (6%) | 9/64 (14%) | **Worse** in absolute (fewer active codes) |
| Quality | 0.034 | 0.018 | Better |
| Cycle | 0.061 | 0.069 | Similar |
| Adv | 0.691 | 0.681 | Similar (near random chance) |

### Analysis: What Improved

1. **Quality classification is excellent.** BCE=0.018, the best across all experiments. The quality encoder perfectly separates pre/post surgery.

2. **Disentanglement still strong.** Adversarial loss at ~0.68 (random chance is 0.693). The GRU-based classifier, which can detect sequence-level patterns, still can't distinguish domains from content — confirming content is truly surgery-agnostic.

3. **No overfitting.** Train/val gap is small (0.047 vs 0.043).

### Analysis: What Still Fails

1. **Codebook collapse is WORSE.** Perplexity stuck at 9 out of 64 codes. Across all 3 experiments (10/256 → 15/256 → 9/64), the model stubbornly uses ~10-15 codes regardless of codebook size, dead code resets, or bottleneck tightness. Dead codes get reset, briefly used, then go dead again.

2. **Reconstruction got worse.** Val recon 0.0433 vs 0.0337 (28% worse). The tighter T/8 bottleneck needs MORE active codes to compensate for reduced temporal resolution, but it got fewer. With only 9 codes × 50 time steps, content representation capacity is severely limited.

3. **Cycle loss didn't improve.** 0.069 vs 0.061 — slightly worse despite all the bottleneck tightening. The model is struggling because the content codes can't faithfully represent speech with only 9 active entries.

### Root Cause: Codebook Collapse is the Blocker

The VQ codebook collapse has persisted across all 3 experiments despite dead code resets, data-initialized codebooks, and commitment weight tuning. The root causes:

1. **EMA decay=0.99 is too slow** — With 58 batches/epoch, reset codes can't move toward encoder outputs fast enough before going dead again.
2. **Commitment weight=1.0 is too high** — Pulls encoder outputs toward existing cluster centers, reinforcing collapse.
3. **No positive incentive for uniform usage** — Dead code reset is reactive (fix dead codes after the fact). There's no proactive gradient signal encouraging the encoder to spread across all codes.

Everything else (disentanglement, quality prediction, adversarial) is already working well. **Fixing codebook utilization is the single highest-impact change.**

### What Should Change for Experiment 4

1. **Product Quantization (Multi-Head VQ)** — Instead of 1 codebook × 64 entries, use 4 independent codebooks × 16 entries each. Each head selects from its own 16-entry codebook, and results are concatenated (4 × 16-dim = 64-dim). This gives 16^4 = 65,536 effective combinations with only 64 total entries, and collapse in one head doesn't affect others.

2. **Codebook Entropy Regularization** — Add a loss term that maximizes the entropy of code usage: `entropy_loss = -sum(p * log(p))` where p is the average selection probability per code. This gives an explicit gradient signal to spread usage uniformly across all codes.

3. **Lower Commitment Weight + EMA Decay** — Commitment: 1.0 → 0.25 (let encoder explore freely). EMA decay: 0.99 → 0.95 (codebook adapts faster).

---

## Experiment 4: Product VQ + Entropy Regularization

**Status:** Not yet started. Changes planned:
- Replace VectorQuantizer with ProductVectorQuantizer (4 heads × 16 codes × 16-dim)
- Add codebook entropy regularization loss
- Commitment weight: 1.0 → 0.25
- EMA decay: 0.99 → 0.95
- Keep all other Exp 3 changes (T/8, GRU classifier, content noise, cycle loss)
