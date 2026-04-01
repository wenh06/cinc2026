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
- [x] Update `cfg.py`: add `ModelCfg.epoch_transformer` config block; set `TrainCfg.batch_size=16`, `TrainCfg.max_seq_len=None`.
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

## Phase 6 — Optional: Raw EEG Spectral Features ⏳

If time permits and Phase 5 reveals headroom, augment each epoch's feature vector with spectral band powers computed from raw EEG, expanding the 21-dim vector to ~30 dims:

| Feature | Band | Notes |
|---------|------|-------|
| Slow-wave activity (SWA) | delta 0.5–4 Hz | Strongest known predictor of cognitive trajectory |
| Sleep spindle density | sigma 12–15 Hz | Thalamo-cortical integrity marker |
| Theta power | 4–8 Hz | Hippocampal dysfunction indicator |
| Alpha power | 8–12 Hz | Cortical arousal index |

Implementation notes:
- Apply 0.5 Hz high-pass + 40 Hz low-pass Butterworth filter before FFT (DC removal, muscle artefact rejection).
- For I0002 records with 500 Hz EEG: downsample to 200 Hz first with anti-aliasing.
- For I0006 records: compute `f3 - m2` (or `f4 - m1`) bipolar derivation from unipolar channels before spectral analysis.
- Spectral features are cached to `cache/spectral_features/` as `.npy` files to avoid repeated computation.

The model receives the extended feature vector transparently; only `CAISR_EPOCH_DIM` in `const.py` changes (21 → 29 or similar).

---

## Phase 7 — Challenge Submission Pipeline 🔄

- [x] `team_code.py`: `train_model`, `load_model`, `run_model` wrappers using `EpochTransformer` + CAISR pipeline.
- [x] `test_docker.py`: all `test_*` functions implemented (`test_dataset`, `test_models`, `test_challenge_metrics`, `test_trainer`, `test_entry`); `test_entry` uses the official `run_model.py` / `evaluate_model.py` entry points.
- [x] `post_docker_build.py`: no pretrained models to cache; minimal environment check.
- [x] Mini training-set subset (`create_mini_dataset.py`) for CI: ≈ 24 records, ~10 MB, stored as `cinc2026-mini-training-set.zip`.
- [ ] Upload mini dataset to GitHub Releases and set `status: alpha` in `.github/workflows/docker-test.yml` to activate the full CI pipeline.
- [ ] Build and smoke-test Docker image locally:
  ```bash
  docker build -f Dockerfile -t cinc2026 . && bash test_run_challenge.sh
  ```
- [ ] Full training run (50 epochs) and submit to the official evaluation system.

---

## Immediate Next Steps

1. **Upload mini dataset** (run `create_mini_dataset.py`, upload zip to GitHub Releases, update `MINI_DATASET_URL` in `docker-test.yml`).
2. **Full training run** (50 epochs, monitor val AUROC, save best checkpoint).
3. **Phase 4**: Implement ECG-HRV MLP fallback for the 14 CAISR-missing records.
4. **Phase 5**: Validation analysis — ROC curves, per-site AUROC, attention maps.
5. **Docker submission**: Set `status: final`, run CI, submit.
