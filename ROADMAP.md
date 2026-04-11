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

## Immediate Next Steps

1. **Upload reduced dataset** (`/Data1/wenh06/cinc2026-reduced-training-set.zip`, 125 MB) to Google Drive; set `REDUCED_DATASET_GDRIVE_ID` in `.github/workflows/docker-test.yml`.
2. **CI pipeline end-to-end**: verify `docker run` passes `test_entry` and score is printed.
3. **Full training run** (100 epochs, monitor val AUROC per site, save best checkpoint to `saved_models/run1/`).
4. **Phase 6.3a**: Add per-epoch spectral features (EEG delta/theta/alpha/sigma, ECG HRV, SpO2 stats) to extend the 21-dim CAISR vector to ~34-dim; cache to `cache/spectral_features/`.
5. **Compare EpochTransformer vs EpochCRNN**: run both at M-size for 100 epochs; pick the winner (or ensemble).
6. **Phase 4**: Implement ECG-HRV MLP fallback for the 14 CAISR-missing records (replace current `(0, 0.5)` constant).
7. **Phase 5**: Validation analysis — ROC curves, per-site AUROC, attention maps.
8. **Docker submission**: Set `status: final`, ensure CI passes, submit.

---

## Feature Enrichment Backlog

Items marked `[quick]` can be done without changing the model architecture (just `CAISR_EPOCH_DIM`).

### Richer CAISR feature extraction  `[quick]`

`build_epoch_features` currently reduces sub-epoch signals to simple scalar means/fractions per 30 s epoch, discarding temporal structure within the epoch:

| Current | What is lost | Better representation |
|---|---|---|
| `arousal_fraction` (scalar mean of binary `arousal_caisr`) | Arousal burst pattern within epoch | Use `caisr_prob_arous` (2 Hz, 60 samples/epoch): add mean + std + max of arousal probability → 3 features instead of 1 |
| `resp_OA/CA/MA/HY` fractions (4 scalars) | Cluster vs spread of events | Add fraction of each class AND count per epoch (absolute burden, not just density) → or add variance of inter-event intervals |
| `limb_iso/PLM` fractions (2 scalars) | PLM periodicity / clustering | Add run-length features: max consecutive PLM seconds, count of isolated bursts |
| `stage_caisr` one-hot (6 dims) | Epoch-to-epoch transitions | Add 5-epoch rolling transition entropy (applied at dataset level, not epoch level) |

Currently unused CAISR channels (see `data_reader.py` issue 7):
- `caisr_prob_no-ar` (idx 1, 2 Hz) and `caisr_prob_arous` (idx 2, 2 Hz) — sub-epoch arousal probability. Richer than binary `arousal_caisr`. A simple addition: replace current 1-dim arousal feature with `[mean, std, max]` of `caisr_prob_arous` across the 60 sub-epoch samples → **+2 dims, total 23**.

### Remove time-position encoding for CRNN  `[quick]`

Cols [19:21] (sin/cos positional encoding) were designed for the Transformer variant (which is permutation-invariant and needs explicit position info). The CRNN's recurrent backbone already tracks sequence position implicitly. Removing these 2 dims reduces `CAISR_EPOCH_DIM` from 21 → 19 and eliminates spurious signal for the CRNN. Requires:
1. `const.py`: `CAISR_EPOCH_DIM = 19`
2. `dataset.py` `build_epoch_features`: drop the `features[:, 19:21] = sin/cos` block
3. `cfg.py` model configs: verify `in_channels=21` is read from `CAISR_EPOCH_DIM` (it is via `BaseCfg.caisr_epoch_dim`)
4. Re-train and compare AUROC vs 21-dim baseline

### Per-record normalization of CAISR features  ✅ done

`normalize_epoch_features()` added to `dataset.py`; applied in `FastDataReader.__getitem__` and mirrored in `team_code._run_model_impl`. Cols 19-20 (time-position encoding) are skipped. Zero-std columns left unchanged.
