# CinC 2026 — Development Roadmap

> **Task**: Predict future cognitive impairment (MCI / Alzheimer's / dementia) from a single polysomnography night using the George B. Moody PhysioNet Challenge 2026 dataset.

---

## Approach Overview

We use a **CAISR-annotation-based epoch-sequence Transformer**.

Each PSG night is decomposed into N × 30-second epochs (≈ 730–1100 epochs per night). Each epoch is represented as a 21-dimensional feature vector derived entirely from the pre-computed CAISR algorithmic annotations (sleep stage one-hot + stage posteriors + arousal density + respiratory event fractions + limb movement fractions + temporal position). A Transformer encoder then models the full-night temporal sequence and outputs a single binary CI prediction.

**Why CAISR-derived features?**

- Site-agnostic: CAISR outputs a canonical feature space regardless of the underlying hardware differences across S0001 / I0002 / I0006.
- Available for all sets: The challenge organisers pre-ran CAISR on training, validation, and test sets; annotation EDF files ship alongside the physiological data.
- Memory-efficient: 21 floats per epoch vs. ≈ 36 M raw EEG samples per night.

For the 1.8 % of training records that lack CAISR annotations (all due to missing EEG/EOG/EMG — see `_CINC2026_INFO` issue 5), a dedicated fallback branch is provided (see Phase 4).

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
       ├─ Linear projection → (B, T, d_model=128)
       │
       ├─ Sinusoidal positional encoding (added, not concatenated)
       │
       ├─ TransformerEncoder (4 layers, nhead=4, dim_feedforward=512, GELU, dropout=0.1)
       │   └─ src_key_padding_mask = padding_mask (B, T)
       │
       ├─ Masked mean pooling over valid (non-padding) positions → (B, d_model)
       │
       ├─ FiLM demographic modulation
       │   └─ demographics (B, 3) → Linear → (scale, shift) applied to pooled repr.
       │
       └─ Linear → scalar logit → BCEWithLogitsLoss
```

Key design choices:
- **FiLM (Feature-wise Linear Modulation)** for demographics instead of simple concatenation, following the `ModelCfg.epoch_transformer.use_film=True` config flag.
- **Masked mean pooling**: average only over the non-padding epochs to avoid length-bias.
- Model config lives entirely in `cfg.ModelCfg.epoch_transformer`; no magic numbers in the model file.

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

## Phase 6 — Alternative Models & Raw-Signal Pathway ⏳

### 6.1 EpochCRNN as an Alternative Sequence Encoder

A natural alternative to `EpochTransformer` for the same CAISR-feature input is a **Bidirectional GRU (BiGRU)** operating directly on the epoch sequence. This is the model family we used in cinc2025 (and prior years), where it performed well.

**Architecture sketch:**
```
epoch_features (B, T, 21)
       │
       ├─ Linear projection → (B, T, 64)
       │
       ├─ 2-layer BiGRU (hidden=128, dropout=0.2)
       │   └─ final hidden states cat'd → (B, 256)
       │
       ├─ FiLM demographic modulation (same as EpochTransformer)
       │
       └─ Linear → scalar logit
```

**Pros vs. EpochTransformer:**
- ~3× fewer parameters (~270K vs. ~825K) → less overfitting on 624 samples.
- Naturally handles variable-length sequences without padding masks.
- Inductive bias: temporal ordering is built in (no positional encoding needed).

**Cons:**
- GRU gradient flow weakens over 800-step sequences → hard to capture sleep-stage transitions at the start/end of the night.
- Parallelism within a sequence is limited.

**Verdict:** Worth implementing as a competitive baseline. If EpochTransformer overfits, EpochCRNN may be more robust. Both use the same `CINC2026Dataset` and `CINC2026Trainer` — only the model changes.

**Implementation:** add `EpochCRNN` to `models/epoch_crnn.py`, register in `models/__init__.py`, add `ModelCfg.epoch_crnn` config block.

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

- [x] `team_code.py`: `train_model`, `load_model`, `run_model` wrappers using `EpochTransformer` + CAISR pipeline.
- [x] `test_docker.py`: all `test_*` functions implemented (`test_dataset`, `test_models`, `test_challenge_metrics`, `test_trainer`, `test_entry`); `test_entry` uses the official `run_model.py` / `evaluate_model.py` entry points.
- [x] `post_docker_build.py`: no pretrained models to cache; minimal environment check.
- [x] Mini training-set subset (`create_mini_dataset.py`): 171 records, ~28 MB (CAISR EDFs only), uploaded to Google Drive; CI workflow downloads via `gdown`.
- [x] Reduced training-set subset (`create_reduced_dataset.py`): 766 records, ~125 MB (CAISR EDFs only); upload to Google Drive and set `REDUCED_DATASET_GDRIVE_ID` in workflow.
- [x] `status: alpha` set in `.github/workflows/docker-test.yml` — full CI pipeline active.
- [x] Strict-test env var (`CINC2026_REVENGER_STRICT_TEST=1`) active in `test_docker.py`; `run_model` has production fallback `(0, 0.5)`.
- [ ] CI pipeline passes end-to-end (Docker build → dataset download → `docker run` → `test_entry` score printed).
- [ ] Full training run (100 epochs, monitor val AUROC, save best checkpoint).
- [ ] Submit to the official evaluation system.

---

## Immediate Next Steps

1. **Upload reduced dataset** (`/Data1/wenh06/cinc2026-reduced-training-set.zip`, 125 MB) to Google Drive; set `REDUCED_DATASET_GDRIVE_ID` in `.github/workflows/docker-test.yml`.
2. **Full training run** (100 epochs, monitor val AUROC per site, save best checkpoint to `saved_models/run1/`).
3. **Phase 6.1**: Implement `EpochCRNN` as an alternative to `EpochTransformer`; compare val AUROC after 100 epochs each.
4. **Phase 6.3a**: Add per-epoch spectral features (EEG delta/theta/alpha/sigma, ECG HRV, SpO2 stats) to extend the 21-dim CAISR vector to ~34-dim; cache to `cache/spectral_features/`.
5. **Phase 4**: Implement ECG-HRV MLP fallback for the 14 CAISR-missing records (replace current `(0, 0.5)` constant).
6. **Phase 5**: Validation analysis — ROC curves, per-site AUROC, attention maps.
7. **Docker submission**: Set `status: final`, ensure CI passes, submit.
