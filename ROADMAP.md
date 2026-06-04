# CinC 2026 — Development Roadmap

> **Task**: Predict future cognitive impairment (MCI / Alzheimer's / dementia) from a single polysomnography night using the George B. Moody PhysioNet Challenge 2026 dataset.

---

## Approach Overview

We use **CAISR-annotation-based epoch-sequence models**.

The current locked baseline after the unofficial phase is **`EpochCRNN_M` + the binary-arousal 21-dim CAISR feature set** (submission 3, AUROC = 0.555). Each PSG night is decomposed into N × 30-second epochs (≈ 730–1100 epochs per night). Each epoch is represented as a compact CAISR-derived feature vector, and an epoch-sequence model (CRNN or Transformer) then outputs a single binary CI prediction.

**Why CAISR-derived features?**

- Site-agnostic: CAISR outputs a canonical feature space regardless of the underlying hardware differences across S0001 / I0002 / I0006.
- Available for all sets: The challenge organisers pre-ran CAISR on training, validation, and test sets; annotation EDF files ship alongside the physiological data.
- Memory-efficient: 21 floats per epoch vs. ≈ 36 M raw EEG samples per night.

For the 1.8 % of training records that lack CAISR annotations (all due to missing EEG/EOG/EMG — see `_CINC2026_INFO` issue 5), a dedicated fallback branch is provided (see Phase 4).

---

## Unofficial Phase Recap

- **Best result**: submission 3 (`EpochCRNN_M` + binary-arousal CAISR features) reached **AUROC 0.555** on the hidden validation set.
- **What worked**: compact CAISR features, moderate model size, and keeping the original night-level burden summaries intact.
- **What did not work**: the last arousal-probability-statistics run (submission 5, AUROC 0.448) bundled too many changes at once: replacing `arousal_fraction`, removing CRNN time encoding, and adding per-record z-score normalization.
- **Lesson**: future changes should be tested by **single-factor ablation** against the locked binary-arousal baseline rather than by stacked edits.

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
|-----------|-------------------|---------------------------|
| NREM delta power (0.5–4 Hz EEG) | ✗ | Strongest sleep biomarker.  Drives glymphatic amyloid-β clearance.  Reduced in MCI/AD. |
| Sleep spindle density / amplitude (12–15 Hz EEG) | ✗ | Fast spindles linked to memory consolidation.  Reduced in MCI. |
| Spindle–slow-wave coupling | ✗ | Temporal coordination critical for hippocampal replay. |
| Heart rate variability (SDNN, RMSSD, LF/HF) | ✗ | Autonomic dysfunction is an early marker of neurodegeneration. |
| SpO₂ desaturation depth / hypoxic burden | ✗ | CAISR detects events but not the *severity* of oxygen drops. |
| Sleep architecture summaries (N3%, REM latency, WASO) | Partially | Could be computed from CAISR stages but not currently used. |

**Principle**: The highest-priority additions are features that capture *independent dimensions* of sleep physiology not represented in the current 21-dim vector — not richer re-parameterizations of the same CAISR signals.

### 3. The prevalence-shift calibration gap

Training (50% positive) vs. hidden validation (~6% positive) creates a severe miscalibration.  Local val AUROC ≈ 0.70 but leaderboard AUROC ≈ 0.555 is a ~0.14 gap.  Simple calibration techniques (Platt scaling, temperature scaling, pos_weight tuning) have not yet been systematically evaluated.

---

## Phase 1 — Data Exploration & Pipeline Foundation ✅

- [x] Explore raw data at `/Data1/wenh06/physionetchallenge2026data`.
- [x] Document site heterogeneity (S0001 / I0002 / I0006 signal differences) in `data_reader.py`.
- [x] Discover and fix the `caisr_prob_*` EDF scale bug (divide by 9.0, re-normalise).
- [x] Define constants in `const.py`: `CAISR_EPOCH_DIM=21`, `CAISR_PROB_EDF_SCALE=9.0`, `STAGE_LABEL_TO_IDX`, `STAGE_ONEHOT_DIM=6`, `DEMOGRAPHIC_DIM=3`, annotation samples-per-epoch.
- [x] Update `cfg.py`: add `ModelCfg.epoch_transformer` config block; set `TrainCfg.batch_size=16`, `TrainCfg.n_epochs=100`, `TrainCfg.max_seq_len=768`.
- [x] Rewrite `dataset.py`:
  - `build_epoch_features()`: CAISR annotation dict → `(N, 21)` float32 array.
  - `CINC2026Dataset`: stratified 80/20 train/val split (stratified by SiteID × label), split cached to `cache/cinc2026-data-split.json`.
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

Implement `CINC2026Trainer` (can subclass `torch_ecg`'s base `Trainer` if it fits, otherwise write from scratch):

- **Loss**: `BCEWithLogitsLoss` (the training set is balanced, so no pos_weight needed for now).
- **Metric**: AUROC (primary); also log AUPRC, accuracy, F1.
- **Optimizer**: AdamW, `lr=1e-3`, `weight_decay=1e-4`.
- **Scheduler**: CosineAnnealingLR over 50 epochs (or ReduceLROnPlateau on val AUROC).
- **Gradient clipping**: `max_norm=1.0` (standard for Transformers).
- **Checkpointing**: save best val AUROC checkpoint to `checkpoints/`.
- **Logging**: log to `log/` via the existing logger infrastructure; optionally add W&B / TensorBoard.

- [x] `BCEWithLogitsLoss` embedded in model; `_setup_criterion` is a no-op.
- [x] AUROC primary metric; NaN-safe evaluation (`nan_to_num` + `clip` before `roc_auc_score`).
- [x] Per-site AUROC (S0001 / I0002 / I0006) logged every epoch for domain-shift monitoring.
- [x] Gradient clipping (`max_norm=1.0`).
- [x] Smoke test: 1 epoch over 624 training records, loss decreases, best-model checkpoint saved.

Training run command:
```bash
python train_model.py -d /path/to/training_set -m saved_models/run1
```

---

## Phase 4 — Fallback for CAISR-Missing Records ⏳

14/780 training records (1.8 %) have no CAISR annotations because the underlying recording contains no EEG/EOG/EMG (equipment failure). Strategy:

1. **Detection**: `FastDataReader.__getitem__` checks `os.path.exists(algo_ann_path)` before loading.
2. **Primary fallback — ECG-HRV MLP**: All 14 records do have an ECG channel. Extract 5-minute windowed HRV features (SDNN, RMSSD, LF/HF ratio, pNN50) using `neurokit2` or `biosppy` over the full night, aggregate to a fixed-length vector, and pass through a small MLP that outputs a calibrated CI probability.
3. **Secondary fallback** (if HRV feature extraction fails): output the training-set label prior (≈ 0.5) as a maximally uncertain prediction.

The main `EpochTransformer` forward pass is never called for these records.

> **Current fallback**: `team_code.run_model` returns `(0, 0.5)` when the CAISR EDF is missing (covers all 1.8 % of affected records).  The full ECG-HRV MLP branch remains to be implemented.

---

## Phase 5 — Validation & Analysis ⏳

After training converges:

- Plot ROC curve and precision–recall curve on the validation set.
- Check per-site AUROC (S0001 / I0002 / I0006 separately) to detect domain shift.
- Inspect attention weights: do the Transformer heads attend to NREM3-heavy regions of the night? (Expected for cognitive biomarkers.)
- Calibrate output probabilities with `sklearn.calibration.CalibratedClassifierCV` (Platt scaling) if needed for the AUROC metric.

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
       ├─ ResNet-N CNN (3 stages, epoch-scale kernels k=5,3,3, stride=2 each → T/8)
       │   CNN out: (B, C_out, T/8)
       │
       ├─ Bidirectional LSTM (retseq=False → last hidden, both directions cat'd)
       │   LSTM out: (B, 2·hidden)
       │
       ├─ FiLM demographic modulation (same as EpochTransformer)
       │
       └─ MLP head → scalar logit → BCEWithLogitsLoss
```

Size presets (swap via `TrainCfg.model_name`):

| Preset | CNN channels | LSTM hidden | clf | ~params |
|--------|-------------|------------|-----|---------|
| `epoch_crnn_S` | 16→32→64 | [64] | [32] | ~123 K |
| `epoch_crnn_M` *(default)* | 32→64→128 | [128] | [64] | ~437 K |
| `epoch_crnn_L` | 64→128→256 | [256] | [128] | ~1.65 M |

Additional backbone variants (change `config.cnn.name`):
- `resnetNS_M` — separable convolutions (~368 K)
- `resnetNB_M` — bottleneck residual blocks (~1.08 M)

**Modular config system** (`model_configs/` package):
- `EPOCH_CRNN_CONFIG` registers all 5 backbone variants; switching backbone is one line.
- `EPOCH_TRANSFORMER_BASE` holds size-agnostic Transformer params.
- `cfg.py` helper functions `_make_epoch_crnn` / `_make_epoch_transformer` assemble presets DRY.
- `team_code.py` is fully config-driven via `_MODEL_CLASS_MAP`; switch model by changing `TrainCfg.model_name` only.

**Verified:** forward pass, inference API, save/load round-trip for all 6 presets.

### 6.2 Time-Series Foundation Models (TimesFM etc.)

Foundation models (TimesFM, Chronos, Moirai, MOMENT) are **not recommended** for this task:

| Model | Designed for | Parameters | Multivariate | Classification |
|-------|-------------|------------|--------------|----------------|
| TimesFM 2.5 (Google) | Univariate forecasting | 200 M | ✗ | Forecasting only |
| Chronos (Amazon) | Univariate forecasting | 710 M | ✗ | Forecasting only |
| Moirai (Salesforce) | Multivariate forecasting | 310 M | ✓ | Forecasting only |
| MOMENT-small (CMU) | Multiple tasks incl. classification | ~40 M | ✓ | ✓ (native head, ECG-tested) |
| MOMENT-large (CMU) | Multiple tasks incl. classification | 125 M | ✓ | ✓ |

Key reasons why they don't fit for the **CAISR-feature pathway**:
1. **Scale mismatch**: TimesFM (200M), Chronos (710M), Moirai (310M) — 250k+ params per training sample → catastrophic overfitting.
2. **Forecasting-first design**: TimesFM, Chronos, Moirai have no native classification head and require architectural surgery.
3. **Univariate bias**: TimesFM and Chronos are univariate only; multivariate XReg support in TimesFM 2.5 is brand-new and unproven on medical data.
4. **Our input is already features, not raw waveforms**: the pretrained representations don't transfer.

**Exception — MOMENT-Small**: At ~25M params (vs. 200M+ for others), the param/sample ratio (~31k:1) is borderline acceptable for fine-tuning with a frozen backbone. It has a native multi-channel classification task, is proven on ECG data (PTB-XL tutorial), and is as simple as `pip install momentfm`. However, MOMENT expects raw time series patches (512 samples each), so it operates more naturally on raw signals than on CAISR epoch feature vectors. **For the raw-signal pathway (§6.3)**, MOMENT-Small is the most promising foundation-model option.

**Verdict: For CAISR-feature pathway, use EpochTransformer + EpochCRNN. For raw-signal pathway, MOMENT-Small is worth evaluating alongside a custom EpochCNN.**

### 6.3 Raw Physiological Data Pathway

> **Scale reality check:** physiological EDFs are ~160 GB for 780 records. A single 7-hour recording at 200 Hz with 18 channels contains ≈ 91 M samples. This pathway requires a fundamentally different data pipeline.

#### Why it's hard

| Challenge | Detail |
|-----------|--------|
| Channel heterogeneity | S0001: `E1-M2`, I0002: `E1` (unipolar), I0006: `E1` unipolar + `M1` separate |
| Sampling rate heterogeneity | SpO2: 10–25 Hz; EEG/EOG/EMG: 200 Hz; airflow: 20–200 Hz |
| Memory | Even a single EEG channel for 7 h at 200 Hz = 5.04 M floats per record |
| Dataset size | 160 GB total; cannot be held in RAM or even SSD cache |

#### Recommended strategy — 30-second epoch CNN encoder

Instead of end-to-end raw-signal processing, **augment the CAISR feature vector** by adding per-epoch spectral and statistical features computed from the raw signals. This is an incremental upgrade that preserves the CAISR pipeline:

```
Per-epoch raw signals (30 s × 200 Hz = 6000 samples per channel)
       │
       ├─ Select "universal" channels present in all sites:
       │   EEG (C3-M2 or C3-M2 equivalent), ECG, SpO2/SaO2
       │
       ├─ Channel-level pre-processing (per site):
       │   - Resample to 200 Hz if needed
       │   - Compute bipolar derivation for I0006 (C3 - M2)
       │   - Bandpass filter: 0.5–40 Hz (EEG), 0.67–40 Hz (ECG)
       │
       ├─ Spectral features per EEG channel (Welch PSD):
       │   delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–12 Hz),
       │   sigma (12–15 Hz), beta (15–30 Hz), total power
       │   → 6 features per channel
       │
       ├─ ECG HRV per epoch:
       │   SDNN, RMSSD, pNN50, LF/HF ratio → 4 features
       │
       └─ SpO2 statistics per epoch:
           mean, std, % time < 90% → 3 features
```

This adds ~13 features per epoch on top of the 21 CAISR features → **34-dim epoch vector**, same pipeline, same model (just `CAISR_EPOCH_DIM = 34`).

**Caching:** spectral features are expensive. Cache to `cache/spectral_features/<record_id>.npy` on first computation; `CINC2026Dataset.__getitem__` checks the cache first.

#### Full end-to-end raw-signal model (longer term)

If the augmented-feature approach shows headroom, a full end-to-end model can be built:

```
Per-epoch raw signals (30 s, 3 selected channels)
       │
       ├─ Shared EpochCNN (ResNet1d / EEGNet / STFT encoder)
       │   → (B × T, cnn_dim) per-epoch embedding
       │
       ├─ Reshape → (B, T, cnn_dim) sequence
       │
       ├─ Transformer or BiGRU → (B, d_model) night summary
       │
       └─ Classification head (+ FiLM demographics)
```

Key implementation decisions:
- **Channel selection**: use only EEG (C3-M2), ECG, and SpO2 — available in all three sites with a site-specific preprocessing layer to unify names/montage.
- **Epoch CNN**: EEGNet (compact, 4-layer depthwise separable CNN, works well with <1000 samples) or a small ResNet1d. Input: `(B × T, C, L)` where `C=3` channels, `L=6000` samples.
- **Memory management**: load one epoch at a time (30 s), compute CNN features, accumulate, then run the sequence model. Alternatively, cache CNN features to disk.
- **Training**: freeze epoch CNN for the first N epochs; fine-tune jointly after.

**Data pipeline additions needed:**
- `PhysioDataReader`: loads physiological EDFs, normalises channel names, resamples.
- `RawEpochDataset`: replaces `CINC2026Dataset`; reads 30-s windows from disk on the fly.
- Site-specific channel-mapping config (in `const.py` or `cfg.py`).

**Verdict:** Phase 6.3a (spectral augmentation of CAISR features) is the next practical step and should be tried before the full end-to-end approach. It reuses all existing infrastructure and is likely to improve AUROC without requiring a new data pipeline.

---

## Phase 7 — Challenge Submission Pipeline 🔄

- [x] `team_code.py`: `train_model`, `load_model`, `run_model` wrappers; fully config-driven via `_MODEL_CLASS_MAP` — switching model/size requires only changing `TrainCfg.model_name` in `cfg.py`.
- [x] `test_docker.py`: all `test_*` functions implemented (`test_dataset`, `test_models`, `test_challenge_metrics`, `test_trainer`, `test_entry`); `test_models` covers both `EpochTransformer` and `EpochCRNN`; `test_trainer` uses `_MODEL_CLASS_MAP` (config-driven); `test_entry` uses the official `run_model.py` / `evaluate_model.py` entry points.
- [x] `post_docker_build.py`: no pretrained models to cache; minimal environment check.
- [x] Mini training-set subset (`create_mini_dataset.py`): 171 records, ~28 MB (CAISR EDFs only), uploaded to Google Drive; CI workflow downloads via `gdown`.
- [x] Reduced training-set subset (`create_reduced_dataset.py`): 766 records, ~125 MB (CAISR EDFs only); upload to Google Drive and set `REDUCED_DATASET_GDRIVE_ID` in workflow.
- [x] `status: alpha` set in `.github/workflows/docker-test.yml` — full CI pipeline active.
- [x] Strict-test env var (`CINC2026_REVENGER_STRICT_TEST=1`) active in `test_docker.py`; `run_model` has production fallback `(0, 0.5)`.
- [ ] CI pipeline passes end-to-end (Docker build → dataset download → `docker run` → `test_entry` score printed).
- [ ] Full training run (100 epochs, monitor val AUROC, save best checkpoint).
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

Optionally add per-epoch ECG HRV and SpO₂ statistics (see Phase 6.3 in the original roadmap).

**New feature dimension**: 21 (CAISR) + 8 (spectral) = **29 dims**.  With HRV (+4) and SpO₂ (+3): **36 dims**.

### Channel selection

S0001 and I0002 use bipolar `C3-M2`.  I0006 uses unipolar `C3` + `M2` → compute bipolar by subtraction.  All three sites have either `C3-M2` or the raw channels to derive it.  Fall back to any available EEG channel for the remaining records.

### Caching

Spectral features are expensive (Welch PSD per epoch).  Cache to `cache/spectral_features/<record_id>.npy` on first computation.  `FastDataReader.__getitem__` checks the cache first, then appends spectral features to the CAISR vector.

### Validation protocol

**Single-factor ablation**: Train `EpochCRNN_M` with (a) baseline 21-dim CAISR features only, (b) 21-dim + spectral features.  Keep all other hyperparameters identical.  Compare val AUROC.  Only proceed if (b) > (a).

---

## Phase 9 — Night-Level Aggregation Features 🔜

Add per-night summary statistics as a separate feature branch, fused with the epoch-sequence output via late concatenation.

### Motivation

Clinical sleep reports summarize nights into single-number metrics (total N3 time, AHI, arousal index, etc.).  These aggregates are the features a sleep physician would use to assess a patient.  They are complementary to the epoch-level sequence — the sequence model sees fine-grained temporal patterns, while night-level aggregates provide explicit clinical summaries.

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

### 10.1 Prevalence-shift calibration

The training set is balanced (~50% CI positive) but the hidden validation/test sets reflect real-world prevalence (~5–15%).  This creates systematic miscalibration.

Approaches (evaluate on val set; pick the best):
- **Temperature scaling**: learn a single scalar temperature parameter on the val set after training.
- **Platt scaling**: fit a logistic regression on val-set logits.
- **pos_weight tuning**: train with `BCEWithLogitsLoss(pos_weight=w)` for w ∈ {2, 4, 8, 16}.
- **Threshold optimization**: find the optimal binary threshold on val set (not only 0.5).

### 10.2 Cross-site robustness

- **StratifiedGroupKFold** (by SiteID × label): ensures each validation fold contains proportional representation from all three sites.
- **Per-site batch normalization** (or domain-adversarial training) if per-site AUROC shows systematic gaps.

### 10.3 Ensemble

After identifying top-2 model architectures, ensemble their probability outputs:
```python
final_prob = α · prob_model_a + (1-α) · prob_model_b
```
Optimise α on the validation set.  Even a simple average (α=0.5) usually helps.

---

## Updated Immediate Next Steps

1. **Spectral feature extraction pipeline** (Phase 8): implement per-epoch EEG bandpower computation with caching; test with single-factor ablation against the binary-arousal baseline (`EpochCRNN_M`, 21-dim vs 29-dim).
2. **Night-level features** (Phase 9): compute night-level aggregates from CAISR annotations; add the small MLP branch; compare val AUROC with and without night-level features.
3. **Calibration sweep** (Phase 10.1): test temperature scaling, Platt scaling, and pos_weight ∈ {2, 4, 8} on the current best model; pick the best calibration method.
4. **Full training run** with the winning feature configuration (100 epochs, monitor val AUROC per site).
5. **Phase 4**: Implement ECG-HRV MLP fallback for the 14 CAISR-missing records (replace current `(0, 0.5)` constant).
6. **Phase 5**: Validation analysis — ROC curves, per-site AUROC, attention maps.
7. **Docker submission**: Set `status: final`, ensure CI passes, submit.

---

## Feature Enrichment Backlog

Items marked `[quick]` can be done without changing the model architecture (just `CAISR_EPOCH_DIM`).
Items with ⚠️ were tested in submission 5 and found harmful — they should only be retried as isolated single-factor ablations.

### Richer CAISR feature extraction  ⚠️ see Unofficial Phase Feedback §1

`build_epoch_features` currently reduces sub-epoch signals to simple scalar means/fractions per 30 s epoch, discarding temporal structure within the epoch:

| Current | What is lost | Better representation |
|---|---|---|
| `arousal_fraction` (scalar mean of binary `arousal_caisr`) | Arousal burst pattern within epoch | ⚠️ Using `caisr_prob_arous` mean/std/max instead was tested in submission 5 and dropped AUROC from 0.555 → 0.448. If retrying: test mean-only first (single-factor), skip std/max. |
| `resp_OA/CA/MA/HY` fractions (4 scalars) | Cluster vs spread of events | Add fraction of each class AND count per epoch (absolute burden, not just density) → or add variance of inter-event intervals |
| `limb_iso/PLM` fractions (2 scalars) | PLM periodicity / clustering | Add run-length features: max consecutive PLM seconds, count of isolated bursts |
| `stage_caisr` one-hot (6 dims) | Epoch-to-epoch transitions | Add 5-epoch rolling transition entropy (applied at dataset level, not epoch level) |

Currently unused CAISR channels (see `data_reader.py` issue 7):
- `caisr_prob_no-ar` (idx 1, 2 Hz) and `caisr_prob_arous` (idx 2, 2 Hz) — sub-epoch arousal probability.  ⚠️ **Mean/std/max of this channel was the key change in submission 5 and made performance worse.**  If revisiting, try using only the *mean* of `caisr_prob_arous` (i.e. replacing the binary `arousal_fraction` with a soft version of the same quantity, without adding std/max), tested as a single-factor ablation.

### Remove time-position encoding for CRNN  ⚠️ see Unofficial Phase Feedback §1

Cols [19:21] (sin/cos positional encoding) were designed for the Transformer variant (which is permutation-invariant and needs explicit position info). The CRNN's recurrent backbone already tracks sequence position implicitly. Removing these 2 dims reduces `CAISR_EPOCH_DIM` from 21 → 19 and eliminates spurious signal for the CRNN.  ⚠️ This was bundled into submission 5 and cannot be evaluated independently.  If retrying: test as a single-factor ablation.  Requires:
1. `const.py`: `CAISR_EPOCH_DIM = 19`
2. `dataset.py` `build_epoch_features`: drop the `features[:, 19:21] = sin/cos` block
3. `cfg.py` model configs: verify `in_channels=21` is read from `CAISR_EPOCH_DIM` (it is via `BaseCfg.caisr_epoch_dim`)
4. Re-train and compare AUROC vs 21-dim baseline

### Per-record normalization of CAISR features  ⚠️ see Unofficial Phase Feedback §1

`normalize_epoch_features()` added to `dataset.py`; applied in `FastDataReader.__getitem__` and mirrored in `team_code._run_model_impl`. Cols 19-20 (time-position encoding) are skipped. Zero-std columns left unchanged.

⚠️ Per-record z-score normalization was part of submission 5 and is hypothesised to have destroyed between-site distributional signal.  If retrying, test as a single-factor ablation and monitor per-site AUROC before/after.
