# CinC 2026 — Development Roadmap

> **Task**: Predict future cognitive impairment (MCI / Alzheimer's / dementia) from a single polysomnography night using the George B. Moody PhysioNet Challenge 2026 dataset.
>
> **Team**: Revenger  |  **Key Deadlines**: Wild-card 2026-08-07  |  Official phase 2026-08-20  |  CinC 2026 Madrid 2026-09-20–23

---

## Data Facts

| Fact | Unofficial Phase | Official Phase |
|------|:---------------:|:--------------:|
| Training records (total) | 780 | 1,103 (small set) |
| Records with CAISR | 766 (14 missing: 8 I0006, 6 S0001) | TBD |
| CI positive rate (training) | ~50% (balanced) | prevalence-matched to large set |
| Estimated CI rate (hidden test) | ~6% (inferred from AUPRC) | ~5–15% (real-world) |
| Site distribution | S0001: 572 (73%), I0006: 154 (20%), I0002: 54 (7%) | S0001: 857, I0006: 192, I0002: 54 |
| CI time window | 3–7 years post-PSG | **1–6 years** post-PSG |
| Primary metric | AUROC | **Age-conditioned AUROC** |
| Secondary metric | AUPRC, Accuracy, F1 | Prevalence-based reward, AUPRC |
| Local val size (80/20 split) | ~156 samples | ~220 samples |

---

## Approach Overview

We use **CAISR-annotation-based epoch-sequence models**.

The current locked baseline is **`EpochCRNN_M` + the binary-arousal 21-dim CAISR feature set** (submission 3, AUROC = 0.555). Each PSG night is decomposed into N × 30-second epochs (≈ 730–1100 epochs per night). Each epoch is represented as a compact CAISR-derived feature vector, and an epoch-sequence model (CRNN or Transformer) outputs a single binary CI prediction.

**Why CAISR-derived features?**

- **Site-agnostic**: CAISR outputs a canonical feature space regardless of the underlying hardware differences across S0001 / I0002 / I0006.  No need for per-site channel name normalisation, montage conversion, or sampling-rate harmonisation.
- **Available for all splits**: The challenge organisers pre-ran CAISR on training, validation, and test sets; annotation EDF files ship alongside the physiological data.
- **Memory-efficient**: 21 floats per epoch vs. ≈ 36 M raw EEG samples per night.  A full night fits in ~30 KB for CAISR features vs. ~144 MB for the raw EEG alone.

**Feature layout** (binary-arousal, 21 dims):

| Indices | Content | Dims | Source |
|:------:|---------|:---:|--------|
| [0:6] | Sleep stage one-hot (N3, N2, N1, REM, W, Unknown) | 6 | `stage_caisr` |
| [6:11] | Stage softmax probabilities (re-normalised) | 5 | `caisr_prob_{n3,n2,n1,r,w}` ÷ 9 |
| [11] | Arousal fraction | 1 | `arousal_caisr` mean per epoch |
| [12:17] | Respiratory event fractions (OA, CA, MA, HY, RERA) | 5 | `resp_caisr` |
| [17:19] | Limb event fractions (isolated, periodic) | 2 | `limb_caisr` |
| [19:21] | Sin/cos time-position encoding | 2 | Computed |

For the 1.8% of training records lacking CAISR annotations (all due to missing EEG/EOG/EMG — see `_CINC2026_INFO` issue 5), a dedicated fallback returns `(0, 0.5)`; an ECG-HRV MLP fallback is planned (Phase 4).

---

## Design Decisions & Literature Support

### Why epoch-sequence models

Cognitive decline alters sleep *architecture* — not just the total amount of each stage, but the temporal organisation across the night.  A sequence model over 30-second epochs can capture:

- **NREM3 dominance shifts** — in healthy sleep, slow-wave activity is concentrated in the first half of the night (driven by sleep pressure homeostasis); in MCI/AD this gradient flattens.  A Transformer attention mechanism can learn to weight early-night epochs differently from late-night epochs.
- **Arousal periodicity** — the cyclic alternating pattern (CAP) has a period of ~20–40 seconds; a BiLSTM with its recurrent state can implicitly model this rhythm.
- **Stage transition patterns** — frequent NREM→Wake transitions (sleep fragmentation) are more common in CI patients; a sequence model can learn transition probabilities from the data.

### Why FiLM for demographics

Feature-wise Linear Modulation (FiLM) injects demographic context (age, sex, BMI) as a learned affine transformation of the pooled night representation, rather than simple concatenation.  This allows the model to learn *conditional* feature weighting: e.g., the same NREM3 fraction may have different implications for a 45-year-old vs. an 85-year-old.  Age is the strongest known risk factor for CI; FiLM lets the model modulate its interpretation of sleep features by age rather than treating age as just another input feature.

### Literature backing for Phase 8 spectral features

| Biomarker | Physiological mechanism | Key references |
|-----------|------------------------|----------------|
| NREM delta power (0.5–4 Hz) | Glymphatic clearance of amyloid-β during slow-wave sleep; reduced SWA predicts cognitive decline | Xie et al., *Science* 2013; Ju et al., *Brain* 2017; Mander et al., *Neuron* 2016 |
| Sleep spindle density (12–15 Hz) | Thalamocortical spindle activity supports memory consolidation; reduced fast spindle density in MCI/AD | Gorgoni et al., *J Sleep Res* 2016; Winer et al., *J Neurosci* 2019; Mander et al., *Neuron* 2016 |
| Spindle–SW coupling | Temporal precision of spindle nesting in slow-wave up-states is critical for hippocampal-neocortical replay; impaired in AD | Helfrich et al., *Neuron* 2018; Winer et al., *Curr Biol* 2021 |
| HRV (SDNN, RMSSD, LF/HF) | Autonomic dysfunction is an early marker of neurodegeneration; reduced parasympathetic tone during sleep | Lanfranchi et al., *Circulation* 1999; Toledo et al., *Sleep Med Rev* 2022 |
| Theta/alpha ratio | EEG slowing (increased theta, decreased alpha) is a hallmark of cortical dysfunction in early AD | Babiloni et al., *Neurobiol Aging* 2016; Rossini et al., *Clin Neurophysiol* 2020 |

### Literature backing for Night-Level features

Clinical sleep metrics (sleep efficiency, N3%, WASO, arousal index, AHI) are the features a sleep physician would use in a clinical assessment.  They are interpretable, standardised across labs, and have decades of evidence linking them to cognitive outcomes (Blackwell et al., *Sleep* 2014; Yaffe et al., *JAMA* 2011; Diem et al., *J Am Geriatr Soc* 2014).

---

## Unofficial Phase Submission History

| # | ID | Date | Model | Params | AUROC | AUPRC | Acc | F1 | Feature Set | Key Hyperparams |
|:--:|:---:|------|-------|:-----:|:-----:|:-----:|:---:|:--:|-------------|----------------|
| **1** | 1173 | 2026-04-03 | `EpochTransformer_M` | 826 K | 0.522 | 0.059 | 0.201 | 0.065 | `binary_arousal` (21d) | lr=3e-4, bs=16, OneCycle, pct_start=0.3, grad_clip=1.0, no norm |
| **2** | 1192 | 2026-04-04 | `EpochCRNN_resnetNC_BNse_M` | ~1.08 M | 0.497 | 0.038 | 0.080 | 0.075 | `binary_arousal` (21d) | lr=3e-4, bs=64, OneCycle, grad_clip=0, no norm |
| **3** | 1240 | 2026-04-07 | **`EpochCRNN_M`** 🏆 | **437 K** | **0.555** | **0.076** | 0.059 | 0.075 | `binary_arousal` (21d) | lr=3e-4, bs=16, OneCycle, pct_start=0.3, grad_clip=1.0, wd=1e-2, label_smoothing=0.05 |
| 4 | 1270 | 2026-04-08 | `EpochTransformer_L` | ~4.8 M | 0.491 | 0.037 | 0.124 | 0.071 | `binary_arousal` (21d) | lr=3e-4, bs=16, OneCycle, pct_start=0.3, grad_clip=1.0, wd=1e-2, label_smoothing=0.1 |
| 5 | 1348 | 2026-04-09 | `EpochCRNN_M` | 437 K | 0.448 | 0.034 | 0.072 | 0.076 | `arousal_prob_stats` (21d) ⚠️ | lr=3e-4, bs=16, OneCycle, pct_start=0.3, grad_clip=1.0, wd=1e-2, label_smoothing=0.1, **per-record z-score**, **no time enc for CRNN** |

### Architecture comparison (all unofficial submissions)

| Model | Params | Best AUROC | Notes |
|-------|:-----:|:----------:|-------|
| `EpochTransformer_M` | 826 K | 0.522 (sub1) | Pre-LN Transformer, sinusoidal PE, masked mean pooling |
| `EpochCRNN_resnetNC_BNse_M` | ~1.08 M | 0.497 (sub2) | 4-stage bottleneck+SE CNN; batch_size=64 may have caused convergence issues |
| **`EpochCRNN_M`** | **437 K** | **0.555 (sub3)** 🏆 | 3-stage ResNet-N + BiLSTM; simplest, least overfitting |
| `EpochTransformer_L` | ~4.8 M | 0.491 (sub4) | Larger Transformer; likely overfitted with ≥6k params/sample |
| `EpochCRNN_M` (alt features) | 437 K | 0.448 (sub5) | Same model as sub3; different feature set catastrophically degraded perf |

**Key insight**: Among models using the same binary-arousal features, **parameter count and AUROC are inversely correlated** (sub3: 437K→0.555 > sub1: 826K→0.522 > sub4: 4.8M→0.491).  This is a classic small-data regime — regularisation through limited capacity beats expressivity.

### Unofficial Phase Leaderboard Standing

| Metric | Best (sub3) | Leaderboard Rank | Notes |
|--------|:----------:|:----------------:|-------|
| AUROC | 0.555 | 102 / 244 | Top half of leaderboard |
| AUPRC | 0.076 | — | Consistent with ~6% test prevalence |
| Accuracy | 0.059 | — | Model predicts positive for most cases but is mostly wrong (F=0.075) |

---

## Unofficial Phase Feedback

### 1. Richer sub-epoch statistics can *degrade* performance

Submission 5 replaced the 1-dim `arousal_fraction` with 3-dim arousal-probability statistics (mean, std, max from `caisr_prob_arous`), added per-record z-score normalization, and removed time-position encoding for the CRNN — **all in one edit**.  AUROC dropped from 0.555 (submission 3) to 0.448.

Hypothesised reasons:
- **Arousal probability channels may be noisier** than the binary arousal label.  The `caisr_prob_arous` signal is a soft classifier output, not a calibrated probability — its higher-order moments (std, max) may amplify annotation errors.
- **Per-record z-score normalization destroys cross-site signal.**  If Site S0001 has higher baseline arousal fractions than I0006 (due to different patient demographics or equipment), z-scoring each record independently removes this potentially informative site-level variation.
- **Confounded changes.**  Three factors were changed simultaneously; we cannot isolate which one (or which combination) caused the degradation.

**Principle**: Adding features should add *independent* information, not just re-parameterize existing information.  When in doubt, add features *cumulatively* to the working baseline and test each addition with a single-factor ablation.

### 2. The CAISR-to-CI information bottleneck

Current CAISR features summarize one epoch into a 21-dim vector covering sleep stage, arousals, respiratory events, and limb movements.  While this is site-agnostic and practical, it discards several sleep biomarkers with established links to cognitive decline:

| Biomarker | Present in CAISR? | Evidence for CI prediction |
|-----------|:-----------------:|---------------------------|
| NREM delta power (0.5–4 Hz EEG) | ✗ | Strongest sleep biomarker.  Drives glymphatic amyloid-β clearance.  Consistently reduced in MCI/AD. |
| Sleep spindle density / amplitude (12–15 Hz EEG) | ✗ | Fast spindles linked to memory consolidation.  Reduced in MCI. |
| Spindle–slow-wave coupling | ✗ | Temporal coordination critical for hippocampal replay.  Impaired in AD. |
| Heart rate variability (SDNN, RMSSD, LF/HF) | ✗ | Autonomic dysfunction is an early marker of neurodegeneration. |
| SpO₂ desaturation depth / hypoxic burden | ✗ | CAISR detects events but not the *severity* of oxygen drops. |
| Sleep architecture summaries (N3%, REM latency, WASO) | Partially | Could be computed from CAISR stages but not currently used as model features. |

**Principle**: The highest-priority additions are features that capture *independent dimensions* of sleep physiology not represented in the current 21-dim vector — not richer re-parameterizations of the same CAISR signals.

### 3. The prevalence-shift calibration gap

Training (50% positive) vs. hidden validation (~6% positive) creates severe miscalibration.  Local val AUROC ≈ 0.70 but leaderboard AUROC ≈ 0.555 is a ~0.14 gap.  The local val split (80/20 random, 156 records) is dominated by S0001 (~73%); the leaderboard validation set has a different site mix and prevalence.  Simple calibration techniques (Platt scaling, temperature scaling, pos_weight tuning) have not yet been systematically evaluated.

### 4. Smaller models generalise better in this data regime

Across all submissions with the binary-arousal feature set, the smallest model (EpochCRNN_M, 437 K params) achieved the best AUROC, and the largest model (EpochTransformer_L, 4.8 M params) the worst.  With only 624 training records and severe cross-site heterogeneity, model capacity must be constrained to avoid learning site-specific shortcuts.

---

## Phase 1 — Data Exploration & Pipeline Foundation ✅

- [x] Explore raw data at `/Data1/wenh06/physionetchallenge2026data`.
- [x] Document site heterogeneity (S0001 / I0002 / I0006 signal differences) in `data_reader.py`.
- [x] Discover and fix the `caisr_prob_*` EDF scale bug (divide by 9.0, re-normalise).
- [x] Define constants in `const.py`: `BINARY_AROUSAL_CAISR_EPOCH_DIM=21`, `CAISR_PROB_EDF_SCALE=9.0`, `STAGE_LABEL_TO_IDX`, `STAGE_ONEHOT_DIM=6`, `DEMOGRAPHIC_DIM=3`, annotation samples-per-epoch.
- [x] Update `cfg.py`: add `ModelCfg.epoch_transformer` config block; set `TrainCfg.batch_size=16`, `TrainCfg.n_epochs=100`, `TrainCfg.max_seq_len=768`.
- [x] Rewrite `dataset.py`:
  - `build_epoch_features()`: CAISR annotation dict → `(N, 21)` float32 array.
  - `CINC2026Dataset`: stratified 80/20 train/val split (stratified by SiteID × label), split cached to `utils/cinc2026-data-split.json`.
  - `collate_fn`: variable-length padding + `padding_mask` (True = padding, matching `TransformerEncoder.src_key_padding_mask` convention).
- [x] End-to-end smoke test: DataLoader batch shape `(4, 965, 21)` ✓.

---

## Phase 2 — Model Implementation ✅

### 2.1 `EpochTransformer` (`models/epoch_transformer.py`)

Architecture:
```
epoch_features (B, T, 21)
       │
       ├─ Linear projection → (B, T, d_model)
       │
       ├─ Sinusoidal positional encoding (added, not concatenated)
       │
       ├─ TransformerEncoder (Pre-LN, GELU, dropout=0.1)
       │   └─ src_key_padding_mask = padding_mask (B, T)
       │
       ├─ Masked mean pooling over valid (non-padding) positions → (B, d_model)
       │
       ├─ FiLM demographic modulation
       │   └─ demographics (B, 3) → Linear → (scale, shift) applied to pooled repr.
       │
       └─ Linear → scalar logit → BCEWithLogitsLoss
```

Size presets (swap via `TrainCfg.model_name`):

| Preset | d\_model | heads | layers | ff | ~params |
|--------|---------|-------|--------|-----|---------|
| `epoch_transformer_S` | 64 | 2 | 2 | 256 | ~116 K |
| `epoch_transformer_M` *(default)* | 128 | 4 | 4 | 512 | ~826 K |
| `epoch_transformer_L` | 256 | 8 | 6 | 1024 | ~4.8 M |

Key design choices:
- **FiLM (Feature-wise Linear Modulation)** for demographics instead of simple concatenation.
- **Masked mean pooling**: average only over the non-padding epochs to avoid length-bias.
- Model config lives entirely in `model_configs/epoch_transformer.py` + `cfg.ModelCfg`; no magic numbers in the model file.

### 2.2 Register model in `models/__init__.py`

- [x] Add `EpochTransformer` to `__all__` and import map.
- [x] Smoke tests: forward pass (B=2, T=50, D=21), backward pass, `CINC2026Outputs` wrapping, single-sample inference.

---

## Phase 3 — Training Loop (`trainer.py`) ✅

Implement `CINC2026Trainer`:

- **Loss**: `BCEWithLogitsLoss` embedded in model; `_setup_criterion` is a no-op.
- **Optimizer**: AdamW (AMSGrad), `lr=3e-4`, `weight_decay=1e-2`.
- **Scheduler**: OneCycleLR, `max_lr=1e-3`, `pct_start=0.3` (Transformer) / `0.1` (CRNN for faster convergence).
- **Gradient clipping**: `max_norm=1.0`.
- **Label smoothing**: `ε=0.05` (sub3 best), `ε=0.1` (sub4/5 over-smoothed).
- **Metric**: AUROC (primary); also log AUPRC, per-site AUROC (S0001 / I0002 / I0006).
- **Early stopping**: patience=20, min_delta=0.001.
- **Checkpointing**: save best val AUROC to `checkpoints/`.

Training run command:
```bash
python train_model.py -d /path/to/training_set -m saved_models/run1
```

---

## Phase 4 — Fallback for CAISR-Missing Records ⏳

14/780 training records (1.8%) have no CAISR annotations because the underlying recording contains no EEG/EOG/EMG (equipment failure). Strategy:

1. **Detection**: `FastDataReader.__getitem__` checks `os.path.exists(algo_ann_path)` before loading.
2. **Primary fallback — ECG-HRV MLP**: All 14 records do have an ECG channel. Extract 5-minute windowed HRV features (SDNN, RMSSD, LF/HF ratio, pNN50) using `neurokit2` or `biosppy` over the full night, aggregate to a fixed-length vector, and pass through a small MLP.
3. **Secondary fallback**: output the training-set label prior (≈ 0.5) as a maximally uncertain prediction.

> **Current**: `team_code.run_model` returns `(0, 0.5)` when the CAISR EDF is missing.  The full ECG-HRV MLP branch remains to be implemented.

---

## Phase 5 — Validation & Analysis ⏳

- Plot ROC curve and precision–recall curve on the validation set.
- Check per-site AUROC (S0001 / I0002 / I0006 separately) to detect domain shift.
- Inspect attention weights: do the Transformer heads attend to NREM3-heavy regions of the night?
- Calibrate output probabilities with Platt scaling / temperature scaling.

---

## Phase 6 — Alternative Models & Raw-Signal Pathway 🔄

### 6.1 EpochCRNN as an Alternative Sequence Encoder ✅

Implemented in `models/epoch_crnn.py`, config in `model_configs/epoch_crnn.py`.

**Architecture** (ResNet-N + BiLSTM, inheriting `torch_ecg.models.ECG_CRNN`):
```
epoch_features (B, T, 21)
       │
       ├─ Transpose → (B, 21, T)  [21 CAISR features = "channels", T epochs = "time"]
       │
       ├─ ResNet-N CNN (3 stages, kernel k=5,3,3, stride=2 each → T/8)
       │   CNN out: (B, C_out, T/8)
       │
       ├─ Bidirectional LSTM (retseq=False → last hidden, both directions cat'd)
       │   LSTM out: (B, 2·hidden)
       │
       ├─ FiLM demographic modulation
       │
       └─ MLP head → scalar logit → BCEWithLogitsLoss
```

Size presets:

| Preset | CNN channels | LSTM hidden | clf | ~params |
|--------|-------------|------------|-----|---------|
| `epoch_crnn_S` | 16→32→64 | [64] | [32] | ~123 K |
| `epoch_crnn_M` *(default)* | 32→64→128 | [128] | [64] | ~437 K |
| `epoch_crnn_L` | 64→128→256 | [256] | [128] | ~1.65 M |

Additional backbone variants (change `config.cnn.name`):
- `resnetNS_M` — separable convolutions (~368 K)
- `resnetNB_M` — bottleneck residual blocks (~1.08 M)

**Modular config system** (`model_configs/` package): `EPOCH_CRNN_CONFIG` registers all 5 backbone variants.  `team_code.py` is fully config-driven via `_MODEL_CLASS_MAP`.

**Verified:** forward pass, inference API, save/load round-trip for all 6 presets.

### 6.2 Time-Series Foundation Models

Foundation models (TimesFM, Chronos, Moirai, MOMENT) are **not recommended** for this task.  Key reasons: scale mismatch (200M+ params → catastrophic overfitting on 624 training samples), forecasting-first design (no native classification head), and the fact that our input is CAISR features (not raw waveforms).

**Exception — Philosopher's Stone** (`github.com/bdsp-core/philosophers-stone`): A public pretrained sleep EEG model from BDSP (the same organisation that provides CinC 2026 data).  Outputs a 1024-D brain-health latent space + four cognitive scores.  Published to the challenge forum during the official phase.  **Worth evaluating as a frozen feature extractor** — the 1024-D latent vector could serve as a per-epoch embedding, replacing or augmenting CAISR features.  Data-domain match (BDSP → HSP), so transfer may be effective.

**Exception — MOMENT-Small**: At ~25M params, has a native classification head, proven on ECG data.  More suitable for the raw-signal pathway than the CAISR pathway.

### 6.3 Raw Physiological Data Pathway

> **Scale reality check:** physiological EDFs are ~160 GB for 780 records. A single 7-hour recording at 200 Hz with 18 channels contains ≈ 91 M samples. This pathway requires a fundamentally different data pipeline.

#### Recommended strategy — 30-second epoch spectral augmentation (see Phase 8)

Instead of end-to-end raw-signal processing, augment the CAISR feature vector with per-epoch spectral and statistical features computed from the raw signals.  This is an incremental upgrade that preserves the CAISR pipeline.

#### Full end-to-end raw-signal model (longer term)

If the augmented-feature approach shows headroom, a full end-to-end model can be built with a shared EpochCNN (ResNet1d / EEGNet) feeding a sequence model (Transformer / BiGRU).

---

## Phase 7 — Challenge Submission Pipeline 🔄

- [x] `team_code.py`: fully config-driven via `_MODEL_CLASS_MAP`; official phase API synced.
- [x] `test_docker.py`: all `test_*` functions updated for official phase `evaluate_model.py` API.
- [x] Official baseline synced to official phase commit; `create_labels.py` added.
- [x] `sync_official.py`: includes all 6 official scripts.
- [x] Mini training-set subset: 171 records, ~28 MB (CAISR EDFs only), uploaded to Google Drive.
- [x] Reduced training-set subset: 766 records, ~125 MB (CAISR EDFs only).
- [x] `status: alpha` set in CI workflow; strict-test env var active.
- [ ] CI pipeline passes end-to-end (Docker build → dataset download → `docker run` → `test_entry`).
- [ ] Full official-phase training run with new data.
- [ ] Submit to the official evaluation system.

---

## Phase 8 — Spectral Feature Augmentation 🔜

Extend the per-epoch CAISR feature vector with spectral EEG features computed from raw physiological signals.  These features capture *independent* physiological dimensions not represented in CAISR.

### Why this should work when submission 5 did not

Submission 5 re-parameterized *existing* CAISR arousal information (mean/std/max of probability).  Spectral features capture *new* physiological dimensions — EEG power in specific frequency bands — that are not derivable from CAISR annotations at all.  NREM delta power is arguably the most replicated sleep biomarker of cognitive decline in the literature.

### Features to add

Per-epoch (30 s), using one "universal" EEG channel (C3-M2 or site-equivalent bipolar derivation):

```
Per-epoch raw EEG (30 s × 200 Hz = 6000 samples)
       │
       ├─ Bandpass filter 0.5–40 Hz
       │
       ├─ Welch PSD → spectral band powers:
       │   delta (0.5–4 Hz)     → 1 feature
       │   theta (4–8 Hz)       → 1 feature
       │   alpha (8–12 Hz)      → 1 feature
       │   sigma (12–15 Hz)     → 1 feature  (spindle band)
       │   beta (15–30 Hz)      → 1 feature
       │   total power          → 1 feature
       │
       ├─ Derived ratios:
       │   theta/delta ratio    → 1 feature  (EEG slowing index)
       │   alpha/delta ratio    → 1 feature
       │
       └─ → 8 spectral features per epoch
```

Optionally add per-epoch ECG HRV and SpO₂ statistics.

**New feature dimension**: 21 (CAISR) + 8 (spectral) = **29 dims**.  With HRV (+4) and SpO₂ (+3): **36 dims**.

### Channel selection

S0001 and I0002 use bipolar `C3-M2`.  I0006 uses unipolar `C3` + `M2` → compute bipolar by subtraction.  All three sites have either `C3-M2` or the raw channels to derive it.  Fall back to any available EEG channel.

### Caching

Spectral features are expensive (Welch PSD per epoch).  Cache to `cache/spectral_features/<record_id>.npy` on first computation; `FastDataReader.__getitem__` checks the cache first.

### Validation protocol

**Single-factor ablation**: Train `EpochCRNN_M` with (a) baseline 21-dim CAISR features only, (b) 21-dim + spectral features.  Keep all other hyperparameters identical.  Compare val AUROC and per-site AUROC.  Only proceed if (b) > (a).

---

## Phase 9 — Night-Level Aggregation Features 🔜

Add per-night summary statistics as a separate feature branch, fused with the epoch-sequence output via late concatenation.

### Motivation

Clinical sleep reports summarize nights into single-number metrics (total N3 time, AHI, arousal index, etc.).  These aggregates are the features a sleep physician would use.  They are complementary to the epoch-level sequence — the sequence model sees fine-grained temporal patterns, while night-level aggregates provide explicit clinical summaries.

### Features (all computable from CAISR annotations)

```python
night_features = {
    # Sleep architecture
    "total_sleep_time": ...,
    "sleep_efficiency": ...,
    "nrem3_pct": ...,            # N3%
    "rem_pct": ...,
    "wake_after_sleep_onset": ...,

    # Clinical event indices
    "arousal_index": ...,         # events per hour
    "ahi": ...,                   # apnea-hypopnea index
    "plmi": ...,                  # periodic limb movement index

    # Temporal dynamics
    "nrem3_latency": ...,         # minutes to first N3
    "rem_latency": ...,
    "sleep_stage_transitions": ..., # count of stage shifts
    "first_half_nrem3_pct": ...,  # N3% in first half of night (glymphatic proxy)
}
```

**Architecture**:
```
Epoch features (B, T, 21)                    Night features (B, 12)
       │                                           │
       ├─ EpochCRNN / Transformer                  ├─ MLP (small, e.g. 12→32→16)
       │   → (B, d_model)                          │   → (B, 16)
       │                                           │
       └────────── Concat ─────────────────────────┘
                         │
                         ├─ FiLM demographics
                         └─ Linear → scalar logit
```

---

## Phase 10 — Calibration & Training Robustness 🔜

### 10.1 Official Phase Metric Alignment

The official phase uses **age-conditioned AUROC** for final ranking.  This metric compares only age-matched positive/negative pairs — penalising models that simply learn "older = CI".  Our current trainer monitors plain AUROC; adding age-binned AUROC monitoring will help detect whether the model is learning age-invariant features.

### 10.2 Prevalence-shift calibration

Training is balanced (~50% CI positive) but test reflects real-world prevalence (~5–15%).  Evaluate:
- **Temperature scaling**: learn a single temperature on the val set after training.
- **Platt scaling**: fit a logistic regression on val-set logits.
- **pos_weight tuning**: train with `BCEWithLogitsLoss(pos_weight=w)` for w ∈ {2, 4, 8, 16}.

### 10.3 Cross-site robustness

- **StratifiedGroupKFold** (by SiteID × label): ensures each validation fold contains proportional representation from all three sites.
- **Domain-adversarial training**: add a site classifier head with gradient reversal to the pooled representation.

### 10.4 Ensemble

After identifying top-2 model configurations, ensemble their probability outputs:
```python
final_prob = α · prob_model_a + (1-α) · prob_model_b
```
Optimise α on the validation set.  Even a simple average (α=0.5) usually helps with small-data generalisation.

---

## Updated Immediate Next Steps

1. **Download official phase data** and verify CAISR annotations are present for all new records.
2. **Regenerate train/val split** with new labels (1–6 year window) and new record counts.
3. **Spectral feature extraction pipeline** (Phase 8): implement per-epoch EEG bandpower computation with caching; single-factor ablation against baseline.
4. **Night-level features** (Phase 9): compute night-level aggregates from CAISR; add MLP branch; compare val AUROC.
5. **Calibration sweep** (Phase 10): test temperature scaling, Platt scaling, and pos_weight on the current best model.
6. **Evaluate Philosopher's Stone** as a frozen feature extractor for per-epoch embeddings.
7. **Full training run** with the winning configuration; monitor per-site and per-age-bin AUROC.
8. **Phase 4**: Implement ECG-HRV MLP fallback for CAISR-missing records.
9. **Docker submission**: ensure CI passes; submit to official evaluation system.

---

## Feature Enrichment Backlog

Items marked `[quick]` can be done without changing the model architecture (just `CAISR_EPOCH_DIM`).
Items with ⚠️ were tested in submission 5 and found harmful — they should only be retried as isolated single-factor ablations.

### Richer CAISR feature extraction  ⚠️ see Unofficial Phase Feedback §1

`build_epoch_features` currently reduces sub-epoch signals to simple scalar means/fractions per 30 s epoch, discarding temporal structure within the epoch:

| Current | What is lost | Better representation |
|---|---|---|
| `arousal_fraction` (scalar mean of binary `arousal_caisr`) | Arousal burst pattern within epoch | ⚠️ Using `caisr_prob_arous` mean/std/max instead was tested in submission 5 and dropped AUROC from 0.555 → 0.448. If retrying: test mean-only first (single-factor), skip std/max. |
| `resp_OA/CA/MA/HY` fractions (4 scalars) | Cluster vs spread of events | Add fraction of each class AND count per epoch (absolute burden, not just density) |
| `limb_iso/PLM` fractions (2 scalars) | PLM periodicity / clustering | Add run-length features: max consecutive PLM seconds, count of isolated bursts |
| `stage_caisr` one-hot (6 dims) | Epoch-to-epoch transitions | Add 5-epoch rolling transition entropy |

Currently unused CAISR channels (see `data_reader.py` issue 7):
- `caisr_prob_no-ar` (idx 1, 2 Hz) and `caisr_prob_arous` (idx 2, 2 Hz).  ⚠️ **Mean/std/max of this channel was the key change in submission 5 and made performance worse.**  If revisiting, try using only the *mean* of `caisr_prob_arous` (a soft version of `arousal_fraction`), tested as a single-factor ablation.

### Remove time-position encoding for CRNN  ⚠️ see Unofficial Phase Feedback §1

Cols [19:21] (sin/cos positional encoding) were designed for the Transformer variant.  The CRNN's recurrent backbone already tracks sequence position implicitly.  ⚠️ This was bundled into submission 5 and cannot be evaluated independently.  If retrying: test as a single-factor ablation.

### Per-record normalization of CAISR features  ⚠️ see Unofficial Phase Feedback §1

`normalize_epoch_features()` added to `dataset.py`; applied in `FastDataReader.__getitem__` and mirrored in `team_code._run_model_impl`.  ⚠️ Per-record z-score normalization was part of submission 5 and is hypothesised to have destroyed between-site distributional signal.  If retrying, test as a single-factor ablation and monitor per-site AUROC before/after.

---

## Experiment Log

Template for tracking training runs.  Fill in one row per experiment.

| ID | Date | Model | Feat Dim | New Features | lr / bs / epochs | Val AUROC | Δ vs Baseline | Notes |
|:--:|------|-------|:--------:|-------------|------------------|:---------:|:------------:|-------|
| B0 | — | `EpochCRNN_M` | 21 | (baseline) | 3e-4 / 16 / 100 | — | — | Locked binary-arousal baseline |
| E1 | | | | | | | | |
| E2 | | | | | | | | |

### Ablation protocol

For each new feature / change:

1. Start from the locked baseline config (`EpochCRNN_M`, binary-arousal 21-dim, all hyperparams as sub3).
2. Make exactly **one** change.
3. Train for 100 epochs; record val AUROC, per-site AUROC.
4. If Δ > 0: accept the change and update the baseline.
5. If Δ ≤ 0: reject, document the hypothesis for why it failed, move on.

---

## Key Dates & Deadlines

| Date | Event |
|------|-------|
| 2026-07-31 | Early-bird registration deadline |
| 2026-08-07 | Wild-card entry deadline |
| 2026-08-20 | **Official phase final submission deadline** |
| 2026-09-01 | 4-page preprint submission deadline |
| 2026-09-20–23 | **CinC 2026, Madrid** |
| 2026-10-10 | Final 4-page paper deadline |
