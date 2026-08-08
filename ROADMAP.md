# CinC 2026 — Development Roadmap

> **Task**: Predict future cognitive impairment (MCI / Alzheimer's / dementia) from a single polysomnography night using the George B. Moody PhysioNet Challenge 2026 dataset.
>
> **Team**: Revenger  |  **Key Deadlines**: Official phase final 2026-08-20  |  CinC 2026 Madrid 2026-09-20–23

---

## Data Facts

| Fact | Unofficial Phase | Official Phase |
|------|:---------------:|:--------------:|
| Training records (total) | 780 | 1,103 (small) / **6,600 (large)** |
| Records with CAISR | 766 (14 missing: 8 I0006, 6 S0001) | 1,090 small / 6,587 large (13 missing: S0001:10, I0002:1, I0006:2) |
| CI positive (training) | 392 / 780 (50.3%) — artificially balanced | 84 / 1,103 (7.6%) small / 498 / 6,600 (7.5%) large — prevalence-matched |
| Estimated CI rate (hidden test) | ~6% (inferred from AUPRC) | **5–15% (challenge page, confirmed)** |
| Age (mean ± std, range) | 70.4 ± 8.3, [50, 89] | 62.0 ± 8.5, [50, 88] (small = large, same distribution) |
| Sex (M / F) | 471 / 309 (60% / 40%) | small 585 / 518; large 3,501 / 3,099 (53% / 47%) |
| Site distribution | S0001: 572 (73%), I0006: 154 (20%), I0002: 54 (7%) | small: S0001 857, I0006 192, I0002 54; large: **S0001 5,139 (78%), I0006 1,142 (17%), I0002 319 (5%)** |
| CI time window | 3–7 years post-PSG | **1–6 years** post-PSG |
| Primary metric | AUROC | **Age-conditioned AUROC** |
| Secondary metric | AUPRC, Accuracy, F1 | Prevalence-based reward, AUPRC |
| Local val size | ~156 (80/20) | 1,320 (canonical 5-fold val, large) |
| **Official validation cohort** | — | **I0004 — a single unseen source, hidden** (not S0001/I0002/I0006; CAISR-held-out; 5–15% CI; no human annotations) |

> **Dataset continuity**: Only 116 records (BidsFolder IDs) are shared between the unofficial and official training sets — 987 records are new, 664 were removed.  The official phase is effectively a fresh dataset, not an expansion.  On the 116 shared records, CI labels, Age, and Sex are **100% consistent** (only 3 records show minor age deltas of ±1–2 years, likely data corrections).  The CI rate change (50% → 7.6%) is therefore entirely driven by dataset re-composition — adding 987 younger, predominantly CI-negative records and removing 664 older, CI-heavy records — not by label redefinition.  Contributing factors: (1) prevalence-matching to the real population instead of artificial balancing, (2) a narrower CI time window (1–6 years post-PSG instead of 3–7), and (3) a younger average age (62 vs 70).
>
> **EEG montage & CAISR domain heterogeneity (verified 2026-08-06)** — the cross-site mechanism behind the official-val gap:
> - Montage: S0001 & I0002 record **bipolar mastoid-referenced** derivations (`F3-M2, C3-M2, O1-M2`); I0006 records **monopolar** (`C3, O1, F3` + separate M1/M2 channels).  Verified on raw EDFs at `/Data1/.../physiological_data/`.
> - CAISR was trained on **S0001 (MGH) + MESA/MrOS/SHHS** (CAISR paper, *Sleep* 2025) — S0001 is CAISR in-distribution; **I0002/I0004/I0006 (and the test source I0007) are CAISR-held-out**, with the additional montage/hardware differences (I0004: mixed Grass/SD32+/Sandman/SOMNOmedics generations; I0007: a different lower-bandwidth system).
> - Consequence: the "canonical" CAISR feature space still shifts by site, and our model (78% S0001 training weight) is calibrated to the S0001 feature regime.  This is why LO-site I0006-holdout (0.562, monopolar+CAISR-OOD) ≈ official scores (0.59–0.62) while I0002-holdout (0.742, mastoid) transfers far better.

---

## Approach Overview

We use **CAISR-annotation-based epoch-sequence models**.

The current official-phase baseline is **`EpochCRNN_M` + the binary-arousal 21-dim CAISR feature set** on the large training set (6,600 records): O0 = 0.762 same-site age-cond; current default config = **O8 5-fold focal+LS ensemble** (0.8045 same-site full-train).  Each PSG night is decomposed into N × 30-second epochs (≈ 730–1100 epochs per night). Each epoch is represented as a compact CAISR-derived feature vector, and an epoch-sequence model (CRNN or Transformer) outputs a single binary CI prediction.  *(Unofficial-phase best was sub3, AUROC 0.555 on 780 records — see Unofficial Phase Submission History.)*

**Why CAISR-derived features?**

- **Site-agnostic — assumed, partially refuted (2026-08-06)**: CAISR outputs a canonical feature space regardless of the underlying hardware differences across S0001 / I0002 / I0006, and needs no channel-name normalisation, montage conversion, or sampling-rate harmonisation.  **However, CAISR was trained on S0001 (MGH) + MESA/MrOS/SHHS only** — I0006 and the hidden sources (I0004, I0007) are CAISR-out-of-distribution, with different EEG montages (bipolar vs monopolar) and hardware generations.  The "canonical" space still shifts by site (see Data Facts); cross-site robustness of the features themselves is now a first-class problem, not an assumption.
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

For the 1.8% of training records lacking CAISR annotations (all due to missing EEG/EOG/EMG — see `_CINC2026_INFO` issue 5), a dedicated fallback returns `(0, 0.5)`; an ECG-HRV MLP fallback was planned (Phase 4) but is **deferred** per the Official Phase Strategy.

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

Clinical sleep metrics (sleep efficiency, N3%, WASO, arousal index, AHI) are the features a sleep physician would use in a clinical assessment.  They are interpretable, standardised across labs, and have decades of evidence linking them to cognitive outcomes (Blackwell et al., *Sleep* 2014; Yaffe et al., *JAMA* 2011; Diem et al., *Am J Geriatr Psychiatry* 2016 — original citation "JAGS 2014" corrected 2026-08-02).

> **Evidence grades per feature** (web-verified 2026-08-02): strong — AHI, sleep efficiency, WASO, PLMI, REM latency, N3%; mixed/weak — REM%, TST, arousal index; inferred (physiology-based hypothesis, no direct study) — N3 latency, stage transitions, first-half N3%, cycle count/length, N3 decay slope.  Full per-feature table + complete references in [Phase 9 → Literature verification](#phase-9--night-level-aggregation-features--strategy-p1-top-priority).

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

## Official Phase Submission History

| # | ID | Date | Config | Training set | Age-cond AUROC | Reward | Notes |
|:--:|:---:|------|-------|:------------:|:-------------:|:------:|-------|
| 1 | 2372 | 2026-08-02 | `EpochCRNN_M` single, BCE, pos_weight 12.16, LS 0.0 | large | 0.617 | 0.027 | SessionID int/str bug in run_model (local-eval artefact only — see §10.5 correction) |
| 2 | 2407 | 2026-08-05 | sub1 training + SessionID fix + sliding-window inference | large | 0.592 | −0.013 | ≈ sub1 within noise — official scorer uses its own ages, so the fix was officially irrelevant |
| 3 | 2471 | 2026-08-08 | 5-fold focal+LS ensemble, early-stop floor (O8) | large | PENDING | PENDING | first cross-site-aware submission |

> Official val = unseen source **I0004**; leaderboard top ≈ 0.773 (large-trained) / 0.75 (small-trained).  Full details in `submissions` file.

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

Across all submissions with the binary-arousal feature set, the smallest model (EpochCRNN_M, 437 K params) achieved the best AUROC, and the largest model (EpochTransformer_L, 4.8 M params) the worst.  With only 624 training records (unofficial phase) and severe cross-site heterogeneity, model capacity must be constrained to avoid learning site-specific shortcuts.  The official phase's 1,103 records alleviate this somewhat but do not eliminate the fundamental constraint.

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
python train_model.py -d /path/to/training_set_small -m saved_models/run1
```

---

## Phase 4 — Fallback for CAISR-Missing Records ⏳ (deferred)

In the unofficial phase 14/780 training records (1.8%) had no CAISR annotations because the underlying recording contained no EEG/EOG/EMG (equipment failure).  In the official phase (small set), 13/1,103 records (1.2%) lack CAISR annotations.

Strategy:

1. **Detection**: `FastDataReader.__getitem__` checks `os.path.exists(algo_ann_path)` before loading.
2. **Primary fallback — ECG-HRV MLP**: All such records do have an ECG channel. Extract 5-minute windowed HRV features (SDNN, RMSSD, LF/HF ratio, pNN50) using `neurokit2` or `biosppy` over the full night, aggregate to a fixed-length vector, and pass through a small MLP.
3. **Secondary fallback**: output a neutral prediction.

> **Current**: `team_code.run_model` returns `(0, 0.5)` when the CAISR EDF is missing.  The full ECG-HRV MLP branch remains to be implemented.  **Deferred** per Official Phase Strategy — HRV implementation cost is high relative to expected gain.

---

## Phase 5 — Validation & Analysis 🔄

- [ ] Plot ROC curve and precision–recall curve on the validation set.
- [x] Check per-site AUROC (S0001 / I0002 / I0006 separately) to detect domain shift. → Tracked per-epoch, gap ~0.05.
- [x] **Age-dependence diagnosis** (→ P0 in Strategy): plot predicted probability vs. age; check if model uses age as a shortcut.
- [ ] Inspect attention weights: do the Transformer heads attend to NREM3-heavy regions of the night? (N/A for CRNN)
- [ ] Calibration moved to → [Official Phase Strategy P2](#official-phase-strategy-post-abstract-acceptance).

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

Foundation models (TimesFM, Chronos, Moirai, MOMENT) are **not recommended** for this task.  Key reasons: scale mismatch (200M+ params → catastrophic overfitting on ~1,100 training samples), forecasting-first design (no native classification head), and the fact that our input is CAISR features (not raw waveforms).

**Exception — Philosopher's Stone** (`github.com/bdsp-core/philosophers-stone`): A public pretrained sleep EEG model from BDSP (the same organisation that provides CinC 2026 data).  Outputs a 1024-D brain-health latent space + four cognitive scores.  Published to the challenge forum during the official phase.  **Worth evaluating as a frozen feature extractor** — the 1024-D latent vector could serve as a per-epoch embedding, replacing or augmenting CAISR features.  Data-domain match (BDSP → HSP), so transfer may be effective.

**Exception — MOMENT-Small**: At ~25M params, has a native classification head, proven on ECG data.  More suitable for the raw-signal pathway than the CAISR pathway.

### 6.3 Raw Physiological Data Pathway

> **Scale reality check:** physiological EDFs are ~214 GiB for 1,103 records (official small set). A single 7-hour recording at 200 Hz with 18 channels contains ≈ 91 M samples. This pathway requires a fundamentally different data pipeline.

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
- [x] Official phase data layout support: `data_reader.py` auto-detects `training_set_small` / `training_set_large` / `training_set` partitions; `dataset.py` partition filter broadened accordingly.
- [x] Official phase training data downloaded and verified.
- [x] Full official-phase training run with new data. → O0 baseline done, AUROC=0.833, age-cond=0.762.

---

## Phase 8 — Spectral Feature Augmentation 🔜 (→ Strategy P3, reduced scope)

> **Scope reduced 2026-07-30** per internal review.  Full band decomposition deferred due to cross-site spectral confound risk.  Start with the most robust signal only.

### Conservative starting point

Per-epoch (30 s), using bipolar C3-M2 (or derive C3 − M2 for I0006), **NREM epochs only**:

```
Per-epoch raw EEG (30 s × 200 Hz = 6000 samples)
       │
       ├─ Bandpass filter 0.5–40 Hz
       │
       ├─ Welch PSD → relative delta power:
       │   delta (0.5–4 Hz) / total power (0.5–40 Hz)  → 1 feature
       │
       └─ → 1 spectral feature per epoch
```

**Why relative delta only**: absolute power depends on electrode impedance and amplifier gain, which vary across sites and equipment.  Relative power (delta / total) is hardware-independent by construction.  NREM only because Wake delta is a different phenomenon (drowsiness artifact, not glymphatic).

**New feature dimension**: 21 (CAISR) + 1 (relative delta) = **22 dims**.

### Cross-site harmonisation (prerequisite)

- [ ] **Before using any spectral feature**, compute and plot C3 power spectra separately for S0001, I0002, I0006.  If systematic site-level differences are visible, apply site-level ComBat harmonisation or per-site z-score.
- [x] **Montage heterogeneity confirmed (2026-08-06)**: S0001/I0002 bipolar mastoid-referenced (`F3-M2...`), I0006 monopolar (`C3, O1, F3` + separate M1/M2) — the bipolar-derivation step for I0006 is mandatory, and the hidden sources (I0004/I0007) add further montage + hardware variation.  Per-site spectra comparison is a hard prerequisite for any spectral feature.
- [x] **Raw-signal access at official inference confirmed (2026-08-06)**: the official python-example-2026 baseline computes physiological (raw-signal) statistics in `run_model` — the inference environment mounts raw PSG + `channel_table.csv` aliasing.  This unblocks Phase 8/P5-10.

### Potential extension (only after delta validates)

- Theta/alpha ratio (EEG slowing index) — 1 additional dim, strong literature support (Babiloni et al. 2016).

### Deferred from original Phase 8 plan

- Full band decomposition (theta, alpha, sigma, beta) — wait until cross-site spectra are characterised.
- ECG HRV and SpO₂ — implementation complexity too high for current timeline.

---

## Phase 9 — Night-Level Aggregation Features 🔜 (→ Strategy P1, top priority)

> **Priority raised 2026-07-30**: highest expected ROI — computed entirely from CAISR, zero raw-signal dependency, zero site-confounding risk.

Add per-night summary statistics as a separate feature branch, fused with the epoch-sequence output via late concatenation.

### Expanded feature set (15 dims)

```python
night_features = {
    # Sleep architecture
    "total_sleep_time": ...,
    "sleep_efficiency": ...,
    "nrem3_pct": ...,              # N3%
    "rem_pct": ...,
    "wake_after_sleep_onset": ...,

    # Clinical event indices
    "arousal_index": ...,           # events per hour
    "ahi": ...,                     # apnea-hypopnea index
    "plmi": ...,                    # periodic limb movement index

    # Temporal dynamics — epoch level
    "nrem3_latency": ...,           # minutes to first N3
    "rem_latency": ...,
    "sleep_stage_transitions": ..., # count of stage shifts per hour
    "first_half_nrem3_pct": ...,    # N3% in first half of night (glymphatic proxy)

    # ✨ New — NREM-REM cycle features (2026-07-30)
    "nrem_rem_cycle_count": ...,    # number of complete sleep cycles
    "mean_cycle_length": ...,       # mean cycle duration (minutes)
    "n3_decay_slope": ...,          # N3% per cycle (flattens in CI)
}
```

**Architecture**:
```
Epoch features (B, T, 21)                    Night features (B, 15)
       │                                           │
       ├─ EpochCRNN / Transformer                  ├─ MLP (15→32→16)
       │   → (B, d_model)                          │   → (B, 16)
       │                                           │
       └────────── Concat ─────────────────────────┘
                         │
                         ├─ FiLM demographics
                         └─ Linear → scalar logit
```

### Literature verification (2026-08-02, web-verified)

> The 15-dim feature list above was defined internally (2026-07-30 review).  Every feature was
> **independently verified against the literature** on 2026-08-02 by searching PubMed / journal sites.
> Verdict per feature — **strong** (direct prospective evidence linking the metric to cognitive
> outcomes), **mixed** (conflicting cohort results), or **inferred** (physiological rationale,
> no direct "metric → cognition" study found).  Also note: the original "Literature backing"
> paragraph below cited Diem as *J Am Geriatr Soc* 2014 — the correct citation is
> Diem et al., *Am J Geriatr Psychiatry* 2016 (verified via DOI).

| # | Feature | Evidence | Key findings |
|---|---------|:--------:|--------------|
| 0 | TST | mixed/weak | **Negative in most cohorts** — Yaffe 2011 & Blackwell 2014 both report TST *not* associated with cognitive outcomes ("quality not quantity"). Retained as a clinically-standard covariate. |
| 1 | Sleep efficiency | **strong** | SE < 74% → MCI/dementia OR 1.53 (Diem 2016, 1,245 women); SE < 70% → executive-decline OR 1.53 (Blackwell 2014, 2,822 men) |
| 2 | N3% | **strong** | Reduced SWS% in AD vs controls, associated with CI severity (Zhang 2022 meta-analysis, 28 studies); SWS correlated with NF-L (neurodegeneration marker) |
| 3 | REM% | mixed | Framingham: lower REM% → dementia risk HR 0.91/%-point (Pase 2017); but 5-cohort Sleep & Dementia Consortium meta (4,657) finds no association (Yiallourou 2025) |
| 4 | WASO | **strong** | WASO ≥ 90 min → executive-decline OR 1.47 (Blackwell 2014); higher WASO with AD (Zhang 2022 meta) |
| 5 | Arousal index | mixed | Yaffe 2011: arousal index **not** associated (driven by hypoxia instead); fragmentation via actigraphy is a strong AD-risk factor (Lim 2013) — AI's own evidence is weaker |
| 6 | AHI | **strong** | AHI ≥ 15 → MCI/dementia OR 1.85 (Yaffe 2011, 298 women, 4.7 y) |
| 7 | PLMI | **strong** | PLMI ≥ 30 → cognitive impairment OR 1.48 (Leng 2016, 2,636 men); PLMS more frequent in aMCI/AD (Liu 2020) |
| 8 | N3 latency | inferred | No direct study found; related metrics (sleep-onset latency, delayed REM) are altered in AD (Zhang 2022; Jin 2025) |
| 9 | REM latency | **strong** | Highest REM-latency tertile → +16% amyloid, +29% tau, −39% BDNF (Jin 2025, *Alzheimers Dement*); AD patients show increased REM latency (Zhang 2022 meta) |
| 10 | Stage transitions/h | **strong** (concept) | Sleep fragmentation → AD HR 1.22/SD (Lim 2013); transitions/h is the PSG-version of this concept |
| 11 | First-half N3% | inferred | SWA is physiologically front-loaded (Borbély homeostatic model); total SWA ↓ with Aβ (Mander 2015) — but no study tests the *gradient* directly |
| 12 | Cycle count | inferred | Cycle number declines with age (established physiology); no direct cognition-outcome study found |
| 13 | Mean cycle length | inferred | Same as #12 |
| 14 | N3 decay slope | inferred | Homeostatic SWA decay is established physiology (Borbély); Mander 2015 links *overall* SWA to Aβ/memory — slope-flattening in CI is an extrapolation, not a tested biomarker |

**Caveats (honest reading):**
1. The strongest direct evidence is for **AHI, SE, WASO, PLMI, REM latency, N3%** — the "clinical index" features.
2. **TST and arousal index** have notable *negative* findings in major cohorts (Yaffe 2011; Blackwell 2014) — they are retained as standard covariates, not because the literature is uniformly positive.
3. **REM%** is genuinely contested (positive Framingham vs null 5-cohort meta-analysis).
4. The **temporal-dynamics features** (#11–14, and partly #8, #10) are physiologically-motivated *hypotheses*, not validated biomarkers.  This is exactly what experiment O2 is testing — if the age-cond AUROC gains come from these, they are novel contributions; if not, the model already extracts them from the epoch sequence.

#### Full reference list (verified 2026-08-02)

1. **Yaffe K, Laffan AM, Harrison SL, et al.** Sleep-Disordered Breathing, Hypoxia, and Risk of Mild Cognitive Impairment and Dementia in Older Women. *JAMA*. 2011;306(6):613–619. doi:10.1001/jama.2011.1115. — AHI≥15 → MCI/dementia OR 1.85 (95% CI 1.11–3.08); hypoxia-driven; arousal/WASO/TST not associated.
2. **Blackwell T, Yaffe K, Ancoli-Israel S, et al.** Associations of Objectively and Subjectively Measured Sleep Quality with Subsequent Cognitive Decline in Older Community-Dwelling Men: the MrOS Sleep Study. *Sleep*. 2014;37(4):655–663. — SE<70% OR 1.53; WASO≥90 min OR 1.47; TST not associated.
3. **Lim ASP, Kowgier M, Yu L, Buchman AS, Bennett DA.** Sleep Fragmentation and the Risk of Incident Alzheimer's Disease and Cognitive Decline in Older Persons. *Sleep*. 2013;36(7):1027–1032. doi:10.5665/sleep.2802. — Fragmentation → AD HR 1.22/SD; 90th-percentile vs 10th: 1.5× risk.
4. **Diem SJ, Blackwell TL, Stone KL, Yaffe K, Tranah G, Cauley JA, Ancoli-Israel S, Redline S, Spira AP, Hillier TA, Ensrud KE.** Measures of Sleep-Wake Patterns and Risk of Mild Cognitive Impairment or Dementia in Older Women. *Am J Geriatr Psychiatry*. 2016;24(3):248–258. PMID 26964485. — SE<74% → MCI/dementia OR 1.53; longer sleep latency associated; TST not associated. ⚠️ (Originally cited in this roadmap as "J Am Geriatr Soc 2014" — corrected 2026-08-02.)
5. **Pase MP, Himali JJ, Grima NA, et al.** Sleep Architecture and the Risk of Incident Dementia in the Community. *Neurology*. 2017;89(12):1244–1250. — Lower REM% → dementia risk (HR 0.91 per %-point).
6. **Yiallourou SR, et al.** Sleep Macro-architecture and Dementia Risk in Adults: Meta-Analysis of 5 Cohorts from the Sleep and Dementia Consortium. *Sleep*. 2025 (PMC11722510). — No consistent association of N1/N2/N3/REM% with incident dementia (4,657 participants; marginal N3% HR 1.06 in a 3-cohort subset).
7. **Leng Y, Blackwell T, Stone KL, Hoang TD, Redline S, Yaffe K.** Periodic Limb Movements in Sleep are Associated with Greater Cognitive Decline in Older Men without Dementia. *Sleep*. 2016;39(10):1807–1810. doi:10.5665/sleep.6158. — PLMI≥30 → cognitive impairment OR 1.48 (95% CI 1.05–2.07); executive function.
8. **Zhang Y, Ren R, Yang L, et al.** Sleep in Alzheimer's Disease: a Systematic Review and Meta-Analysis of Polysomnographic Findings. *Transl Psychiatry*. 2022;12:136. — AD: reduced SWS%/REM%, increased REM latency and sleep latency vs controls.
9. **Jin J, Chen J, Cavaillès C, Yaffe K, Winer J, Stankeviciute L, Lucey BP, Zhou X, Gao S, Peng D, Leng Y.** Association of Rapid Eye Movement Sleep Latency with Multimodal Biomarkers of Alzheimer's Disease. *Alzheimers Dement*. 2025;21(2):e14495. PMID 39868572. — Highest REM-latency tertile: +16% Aβ, +29% p-tau181, −39% BDNF (128 participants).
10. **Mander BA, Marks SM, Vogel JW, et al.** β-Amyloid Disrupts Human NREM Slow Waves and Related Hippocampus-Dependent Memory Consolidation. *Nat Neurosci*. 2015;18(7):1051–1057. — mPFC Aβ ↔ reduced 0.6–1 Hz SWA (r=−0.45); mediates memory-consolidation deficit. Supports overall SWA, not the within-night gradient.
11. **Liu S, et al.** Sleep Spindles, K-complexes, Limb Movements and Sleep Stage Proportions May Be Biomarkers for Amnestic Mild Cognitive Impairment and Alzheimer's Disease. *Sleep Breath*. 2020. — More PLMS in aMCI/AD than controls; correlated with MMSE/MoCA.

> Physiology background for the *inferred* features: SWA homeostatic front-loading across the night (Borbély's two-process model) and the decline of NREM-REM cycle number with age are established sleep physiology; glymphatic clearance of Aβ during SWS (Xie et al., *Science* 2013) motivates the N3%/first-half-N3% proxies.


---

## Phase 10 — Calibration & Training Robustness 🔜 (→ Strategy P0 + P2)

### 10.1 Age dependence diagnosis (→ Strategy P0) 🔴

The official phase uses **age-conditioned AUROC** for final ranking.  This metric compares only age-matched positive/negative pairs — penalising models that simply learn "older = CI".  Our current trainer monitors plain AUROC; age-conditioned AUROC monitoring must be added *before* any feature work so we can track whether changes actually help or hurt.

- [x] Plot predicted probability vs. age on validation set.
- [x] Add age-conditioned AUROC as a validation metric in `CINC2026Trainer`.
- [x] Consider age-adversarial head (gradient reversal) if strong age dependence is detected. → Implemented 2026-08-02 (`--age-adv` CLI flag; GRL + age-regression head in `EpochCRNN`, tap point before FiLM; `TrainCfg.age_adv` config block). **Result: failed (experiment O1)** — see Experiment Log.

### 10.2 Prevalence-shift calibration (→ Strategy P2)

Training is balanced (~50% CI positive) but test reflects real-world prevalence (~5–15%).  Evaluate:
- **pos_weight tuning**: sweep `pos_weight ∈ {2, 4, 8, 16}` in `BCEWithLogitsLoss`.
- **Temperature scaling**: learn a single scalar temperature on val logits post-training.
- **Platt scaling**: logistic regression on val logits (more expressive than temperature).

### 10.3 Cross-site robustness (partially implemented 2026-08-03)

- ✅ **Multi-factor stratified split**: both the canonical dynamic split
  (`CINC2026Dataset`) and the 5-fold split (`utils/make_5fold_split.py`)
  stratify on **label × SiteID × Sex × age band** via torch_ecg's
  `stratified_train_test_split` — each fold mirrors the population on every
  axis (val deviations < 0.8% in the 5-fold report).  Replaces the earlier
  label-only `StratifiedShuffleSplit`.
- ✅ **Leave-one-site-out (LO-site) experiment** (2026-08-04, complete): train
  on 2 sites, evaluate on the *full* held-out site (3 runs).  Results
  (held-out age-cond; OOF per-site reference in parens):
  S0001 **0.552** (0.738, −0.186, n_train=1461) / I0006 **0.562** (0.682,
  −0.120, n_train=5458) / I0002 **0.742** (0.653, **+0.089**, n_train=6281).
  **A1 rerun under the sub3 config (focal+LS+floor, 2026-08-09)**: I0006
  **0.638** (+0.076) / S0001 **0.594** (+0.042) / I0002 0.703 ± 0.02 (3 runs,
  ≈0) — **the config change transfers cross-site**; I0006-holdout is the
  I0004-like proxy, so sub3's official score should beat sub2's 0.592.
  Run-to-run noise ±0.02–0.04 → intervention pass line Δ > 0.04 (see
  Experiment Log **A1**).
  → **Cross-site shift is real and asymmetric, but the "I0002 is the sub1
  drop source" hypothesis is refuted** — a model that never saw I0002 scores
  it *better* than the in-distribution OOF model.  Caveats: (a) n_train
  differs across runs (S0001 held-out trains on only 1,461 recs), so the drop
  ordering (S0001 > I0006 > I0002) partly reflects training-set size, not
  pure domain shift; (b) all 3 models hit their best by epoch 2–5 with fast
  early stop; (c) eval used center-crop (not sliding-window, −0.007 scale).
  Script `/tmp/lo_site/lo_site.py` (patches
  `dataset.FIXED_DATA_SPLIT_FILE` to a per-site split); results in
  `/tmp/lo_site/metrics_{site}.json`.  See Experiment Log row **O6**.
- **Domain-adversarial training**: add a site classifier head with gradient reversal to the pooled representation. (deferred — O1 age-adv variant failed; site-adv not yet attempted)

### 10.4 Ensemble — 5-fold CV (O5, implemented 2026-08-03)

5-fold CV ensemble implemented end-to-end (multi-factor stratified split →
5× training → equal-weight probability averaging at inference):

- **Split** — `utils/make_5fold_split.py` carves 5 non-overlapping val folds
  (≈ 20% each, disjoint, covering all 6600 records) via recursive
  `torch_ecg.utils.utils_data.stratified_train_test_split` with multi-factor
  stratification on **label × site × sex × age band** (per-fold deviations
  < 0.8% on every axis).  Shipped as `utils/cinc2026-5fold-split.json`.
- **Dataset** — `CINC2026Dataset` reads `train_config.fold` (e.g.
  `CINC2026_OVERRIDE_JSON {"fold": k}`) and serves that fold's train/val
  split; the dynamic canonical split now also uses the same multi-factor
  stratified split (replacing the old sklearn label-only
  StratifiedShuffleSplit; see §10.3).
- **Training** — `TrainCfg.folds = [0..4]` makes `team_code.train_model`
  train one model per fold, saving each to `model_folder/fold_{k}/`.
  `None` (default) keeps the single-model behaviour.
- **Inference** — `load_model` auto-detects the `fold_*` layout and loads
  all folds.  Probability = equal-weight average of the fold probabilities
  (AUROC-family metrics); binary prediction = **majority vote** of per-fold
  binaries, each thresholded at its fold's **tuned binary threshold**
  (reward-maximising scan on the val split after training, serialised into
  the checkpoint's model config; old checkpoints fall back to 0.5).
  Long nights (> `max_seq_len`=768) are inferred via **sliding windows**
  (window 768 / stride 384, per-window probability average) to match the
  training-time sequence-length distribution (measured +0.013 age-cond on
  fold_0 val vs full-night).  Official re-training cost:
  `len(folds) ×` single-fold time.
- **Evaluation** — `utils/evaluate_oof.py` scores each fold on its own val
  fold and aggregates: unbiased out-of-fold AUROC / age-conditioned AUROC /
  per-site.  OOF is a conservative lower bound of the 5-model average the
  official test set measures.

Status: ✅ full 5-fold run complete (2026-08-04).  Per-fold best age-cond
0.785/0.704/0.756/0.740/0.747 (folds 0-4); OOF aggregate AUROC 0.8223 /
age-cond 0.7169 (conservative single-model bound — the test-time 5-model
probability average is expected to be higher).  fold_0 (0.785) vs O0repro
(0.758) on the identical split: +0.027, consistent with run-to-run variance.
Details in the Experiment Log row **O5**.

The older α-weighted top-2-config ensemble idea (below) remains available:
```python
final_prob = α · prob_model_a + (1-α) · prob_model_b
```
Optimise α on the validation set.

### 10.5 Submission 1 (ID 2372) + inference-chain fixes (2026-08-04) 🐛

- **sub1 result**: official score on the (unreleased) validation set — age-cond
  **0.617** / AUROC 0.614 / Reward 0.027 / F1 0.083, far below local val
  (0.762/0.833).  The organisers re-train with our code on their own split,
  then score on a hidden validation set (not released).
- **Root cause found**: a **SessionID type bug** in `run_model` — the
  demographics CSV stores `SessionID` as int64, but `_run_model_impl` cast it
  to str, so `load_demographics`' strict-type mask never matched and **every
  inference demographics vector silently fell back to the defaults
  `[0.6, 0, 0.5]`** (60 y / female / BMI 25).  Age is the core of the primary
  metric (FiLM conditioning + official age-matched pairing), so the model
  lost its age modulation entirely.  Measured impact on O5 fold_0 val (same
  model/records, only the session_id handling differs): AUROC 0.7444→0.8455,
  age-cond 0.6960→0.7798.  On the official small set (1103, ⊂ large — leaked
  for eval, relative comparison only): 0.6863→0.753.
- **Fix** (`7c1187a`): pass the original SessionID type to
  `load_demographics`; keep `str()` only for filename assembly.  Together with
  the sliding-window inference (§10.4), the fixed chain re-evaluates to
  0.8455/0.7798 on fold_0 val.  **Ready to re-submit as sub2** (training code
  unchanged — the organisers' re-training is unaffected).
- **Lesson**: pure-numeric IDs are a classic int-vs-str trap — pandas masks
  compare strictly typed values and fail silently (no error, just fallback).
- **⚠️ 2026-08-06 correction — local-eval artefact, not the official driver**:
  the official scorer reads ages from **its own labels file**
  (`evaluate_model.py: df_labels[...id_age]`), never from our predictions — so
  the demographics fallback could not have moved the *official* age-cond
  metric.  sub2 (with the fix) scored 0.592 ≈ sub1's 0.617 within run-to-run
  noise.  The real sub1/sub2 drop driver is the **unseen-source validation
  cohort (I0004)** — see Data Facts (montage/CAISR-domain note) and §10.6.
  The root-cause analysis above remains correct *as a local-evaluation
  artefact* (it silently inflated our own val estimates).

### 10.6 O7 loss experiments + sub3 config (2026-08-05)

- **O7a (age-matched pairwise) failed** — see Experiment Log; the λ=1.0 hinge
  fights the BCE signal on this small dataset (unstable, −0.04 age-cond).
- **O7b (focal γ=2 + label_smoothing 0.05) strong gain** — full-train
  official-flow eval: age-cond **0.8018 vs 0.7525 (+0.049)**, AUROC 0.8670,
  every site up.  Mechanism: at 7.6% prevalence the abundant easy negatives
  dominate BCE gradients; focal (1−pt)^γ re-centres them on hard samples.
  **Adopted as the new default** (cfg: `focal.enable=True`,
  `label_smoothing=0.05`).
- **sub3 default config** = 5-fold ensemble (§10.4) × focal+LS:
  `TrainCfg.folds=[0..4]`, early-stop floor `min_epochs=30` + patience 20→15.
  Local 5-fold × focal validation (O8): full-train age-cond **0.8045**, mean
  per-fold val 0.767 — above O7b single (0.8018).  **Merged to master
  (`1f84c1c`, PR #15) 2026-08-06; submitted 2026-08-08 (ID 2471, training
  set = large).**
- **⚠️ Official val = unseen source `I0004`** (challenge page; hidden; test
  set = another unseen source `I0007`).  Official sub1/sub2 age-cond
  0.617/0.592 ≈ LO-site held-out drops (O6: 0.552–0.742) → local same-site
  val overestimates cross-site transfer; the official scorer uses its own
  ages (SessionID fix was a local-eval artifact).  Mechanism (verified):
  CAISR is S0001-in-distribution only; I0004 is CAISR-held-out with
  monopolar EEG + mixed hardware generations (see Data Facts).  The closest
  local proxy = **I0006-holdout** (0.562, monopolar + CAISR-OOD) ≈ official
  (0.59–0.62).  Leaderboard top 0.75–0.77, some teams better trained on
  *small* (1,103) than *large* (6,600).  **Pivot (2026-08-06): LO-site as
  local proxy; candidates = small-set training / per-record normalisation /
  mixup / regularisation / ComBat harmonisation / multi-arch ensemble /
  test-time adaptation — see §P5.**
- **Empty-batch crash found & fixed** (`4f267fb`): 13 records lack CAISR
  annotations → empty epoch matrix (n_epochs=0).  With batch_size ≤ 13 (CI
  uses 4), a whole batch of such records makes collate `t_max=0` → CNN first
  conv (kernel 5) crashes ("padded input size 4").  Official re-training
  (batch 16) **cannot** hit this (13 < 16), but CI did — a probabilistic
  crash that cost a CI run.  Fix: `CINC2026Dataset._filter_missing_caisr()`
  drops annotation-missing records from both train and val splits right
  after `_train_test_split`; inference (`run_model`) still emits the
  per-record (0, 0.5) fallback.
- **Tool**: `scripts/eval_all_models.py` — official-flow full-train eval
  (`find_patients` → per-record `run_model` → official metric set + per-site)
  for FAIR cross-model comparison on the (leaked) training data.

### 10.7 Research queue — sub4 candidates (2026-08-05)

> **2026-08-06 update**: both items below are same-site tuning (validated on
> the S0001-dominated local val) — **deprioritised** by the cross-site pivot
> (§P5), which supersedes them for the remaining submissions.  Kept as
> reference; a winning cross-site config should be re-tested with bs=32 / _L
> only if capacity gains remain after the pivot experiments.

- **batch_size 16→32** (√-scale lr 3e-4→~4e-4, re-tune weight decay): a
  5-fold per-fold trainset is only ~859 recs — bs=64 → 13 steps/epoch, the
  unofficial sub2 failure regime; bs=32 → ~27 steps/epoch, a reasonable
  middle ground.  Scaling-law basis: critical batch size scales with dataset
  size (∝ D^0.4–0.5), largely independent of model size (Kempner 2024;
  "Power Lines", NeurIPS 2025).
- **EpochCRNN_M → _L (1.65 M params)**: 8.5× more data (6600 vs 780 recs)
  invalidates the unofficial "smaller is better" finding; single-fold
  controlled run first (~2 h), adopt 5-fold only if it wins.

---

## Official Phase Strategy (post-abstract-acceptance)

> **Strategic priority shift**: The official phase metric is **age-conditioned AUROC**, which penalises models that rely on raw age as a predictive signal.  Our Unofficial Phase Feedback §3 noted a prevalence-shift calibration gap (train 50% → test ~6%), but the metric change is even more fundamental.  The following priorities were refined after an internal review (2026-07-30) against this constraint.

### P5 — Cross-site robustness (current priority, 2026-08-06) 🔴

> **Why this supersedes P0–P4**: official sub1/sub2 (0.617/0.592) and the
> leaderboard show the real task is generalising from the 3 training sources
> to **unseen sources** (val = I0004, test = I0007) under CAISR-domain +
> montage/hardware shift.  Same-site val numbers (0.76–0.80) do not transfer.
> Local evaluation proxy = **LO-site, I0006-holdout configuration first**
> (monopolar + CAISR-OOD ≈ I0004's profile; 0.562 ≈ official 0.59–0.62),
> with n_train held constant across holdout runs for comparability.

Candidate interventions, ranked by evidence × cost (full research notes in
the 2026-08-06 analysis; sources: cross-center PSG studies, sleep-domain
domain-adaptation literature):

1. **Small-set (1,103) training** — leaderboard evidence (DKAW 0.75 small vs
   0.63 large; MATLAB baseline 0.578 vs 0.545) + O6 n_train effect.  Cheap
   (~40 min single-fold).  Test against large on the I0006-holdout proxy.
2. **Per-record z-score normalisation** — direct cross-center PSG evidence
   (+11–20% F1 on CAP detection, EMBC 2023).  ⚠️ unofficial sub5 bundled it
   with other changes and failed — must be a single-factor ablation now.
3. **Mixup / feature-noise / epoch-masking augmentation** — sleep evidence
   (XSleepFormer; BEETL 2021; EEG augmentation +5.8% staging acc).
4. **Model selection on the LO-site proxy** (early-stop/checkpoint by
   held-out-site metric instead of same-site val) — directly optimises the
   cross-site objective.
5. **Stronger regularisation** (weight decay / dropout / smaller capacity) —
   cross-institutional EHR: simple models + site control ≥ deep nets.
6. **ComBat / NeuroHarmonize feature harmonisation** (site as batch;
   NM-ComBat variant for unseen sites) — 5-site EEG spectral-feature evidence.
7. **Multi-architecture soft voting** (CRNN + Transformer + seed diversity) —
   SOMNUS: soft voting beats every single model in 94.9% of comparisons.
8. **Test-time adaptation on the 10 supplementary I0004 examples** (feature
   statistics matching / light BN-affine adaptation) — needs re-downloading
   the supplementary set (Kaggle, 4.6 GB); TTA evidence in sleep is mixed
   (PSDNorm positive; general TTA transfers poorly to EEG — must test).
9. **Site-adversarial GRL** — mixed evidence (ADG-RANet still needs
   fine-tuning; some imaging studies report degradation).  O1 (age-adv)
   failed; treat as last resort.
10. **Raw-PSG spectral features** (relative delta power etc.) — the CAISR
    information bottleneck (strongest missing biomarkers); the official
    baseline already computes raw-signal statistics at inference, so raw
    access in the official environment is confirmed.  Highest cost; Phase 8
    scope + /Data1 raw data available.

**Not recommended**: site-ID as input (useless for unseen sites + shortcut
hazard); BBSE/prior-shift correction (rank-invariant; official prevalence
shift is mild); SAM (no sleep evidence).

---

### P0 — Diagnose age dependence; add age-conditioned AUROC monitoring

The model's 0.555 plain AUROC may partially reflect "age → CI" heuristics, which age-conditioned scoring nullifies.  Before any new features:

- [x] Plot **predicted probability vs. age** on the validation set.  A strong positive correlation means the model is using age as a shortcut. → Done 2026-08-02: r(age, prob)=+0.39, age gap=0.08 (best model).  Strong age dependence confirmed.
- [x] Add **age-conditioned AUROC** to `CINC2026Trainer` as a validation metric (separate from the existing plain-AUROC monitor).
- [x] Evaluate whether FiLM appropriately modulates features by age, or whether an age-adversarial head (gradient reversal on age prediction from the pooled representation) is needed. → Implemented 2026-08-02 (`--age-adv`; GRL + age head before FiLM; experiment O1). **Result: failed** — CSV best age-cond 0.735 < O0's 0.762, and r(age,prob) *rose* to +0.61 (the GRL pushes the age signal into the FiLM channel, which per-sample modulation keeps leaky). Current config (α=0.5, λ=1.0, before-FiLM) not viable; a retry would need λ≫1 (e.g. 10–50) or an after-FiLM tap, with mechanism risk remaining — or drop this route and proceed to P1.
- [ ] Consider **age-stratified sampling** in the DataLoader to ensure each batch contains diverse ages.

### P1 — Night-level aggregation features (Phase 9)

> **Status 2026-08-02**: Implementation complete (commit `2658f57`) — `build_night_features` (15 dims, fixed-divisor normalisation, AASM-consistent AHI excl. RERA), late fusion `backbone → concat MLP(15→32→16) → FiLM → clf`, config block `TrainCfg.night_features` (default OFF = O0 baseline unchanged), `--night-features` CLI, full train/inference plumbing.  Feature sanity cross-checked against the official baseline's event counts on real records (exact match); 3-epoch smoke train→run→evaluate passed; disabled default bit-identical (436,865 params, old O0 checkpoints load fine).  Literature backing verified 2026-08-02 (see [Literature verification](#literature-verification-2026-08-02-web-verified)).  **O2 full-run experiment complete 2026-08-02: ❌ failed** — age-cond 0.738 < O0's 0.762 (Δ −0.024), plain AUROC 0.817 < 0.833.  The 15-dim aggregate signal (computed over the *full* night) is partly redundant with what the epoch-sequence backbone already extracts, and per-site it only helped I0006 (+0.032) while hurting S0001 (−0.022) and I0002 (−0.041) — no consistent benefit.  Route closed; next candidate P2 (calibration).

**Why first**: Computed entirely from CAISR annotations — zero raw-signal dependency, zero site-confounding risk.  Clinical interpretability is the highest of any planned change.

Features to add (12–15 dims → small MLP → late fusion with epoch-sequence output):

| Feature | Source | Clinical relevance |
|---------|--------|-------------------|
| Total sleep time (TST) | `stage_caisr` duration sum | ↓ in CI |
| Sleep efficiency | TST / time in bed | ↓ in CI |
| N3% | fraction of TST in N3 | ↓ in CI |
| REM% | fraction of TST in REM | ↓ in some AD studies |
| WASO | Wake minutes after sleep onset | ↑ in CI |
| Arousal index | `arousal_caisr` events / TST (hours) | ↑ in CI |
| AHI (apnea-hypopnea index) | `resp_caisr` events / TST (hours) | ↑ contributes to vascular CI |
| PLMI | `limb_caisr` periodic events / TST (hours) | ↑ fragments sleep |
| N3 latency | minutes to first N3 epoch | disrupted in CI |
| REM latency | minutes to first REM epoch | disrupted in CI |
| Stage transitions | count of stage changes / hour | ↑ in fragmented sleep |
| First-half N3% | N3% in sleep-first-half vs second-half | Gradient flattening in CI |
| NREM-REM cycle count | number of sleep cycles | ↓ in CI |
| NREM-REM cycle length | mean cycle duration (minutes) | ↓ in CI |
| Cycle N3 dominance | N3% per cycle (decay slope) | flattens in CI |

**Architecture**: Epoch features → EpochCRNN → (d_model) + Night features → small MLP (15→32→16) → concat → FiLM → linear head.

### P2 — Calibration (Phase 10.2)

**Why second**: 50% training prevalence → 5–15% validation prevalence is a known source of miscalibration.  These changes are low-risk and low-cost:

- [x] **pos_weight tuning**: sweep `pos_weight ∈ {2, 4, 8, 16}` in `BCEWithLogitsLoss` → **O3 (2026-08-03): no gain over default 12.16** (best pw=4 ties O0; pw=16 degrades).  Route closed.
- [x] **Temperature scaling**: learn a single scalar temperature on val logits post-training → implemented in `scripts/calibrate.py` (analysis-only).  **AUROC is rank-invariant to temperature/Platt** — they cannot move the official leaderboard metric; only Brier/ECE.
- [x] **Platt scaling**: logistic regression on val logits (more expressive than temperature) → same utility; ECE 0.033→0.014, mean_p→prevalence on the pw=2 checkpoint.

### P3 — Conservative spectral features (Phase 8, reduced scope)

**Why third**: Strong literature basis, but raw-signal pipeline introduces **cross-site spectral confound risk**.  Start minimal:

- [ ] **Relative delta power** only: delta_power / total_power, computed on **NREM epochs only** (Wake delta is meaningless).  1 extra dimension.
- [ ] **Cross-site analysis FIRST**: compare C3 power spectra of S0001 vs I0002 vs I0006 *before* feeding into the model.  If spectral profiles differ systematically by site, add site-level harmonisation.
- [ ] Use bipolar C3-M2 where available; derive by subtraction for I0006 (unipolar C3, M2 present).
- [ ] Only after validation: add theta/alpha ratio (EEG slowing index) — also 1 dim, high literature support.

### P4 — Philosopher's Stone quick evaluation

**Why last**: BDSP-published pretrained sleep EEG model, same data domain.  1024-dim latent → PCA to 64 dims → frozen feature extractor.

- [ ] Clone and run inference on a small subset (10–20 records) first to check throughput.
- [ ] If the latent space visibly separates CI labels: add PCA-reduced 64-dim epoch embedding as **additional** channel alongside CAISR features.
- [ ] Keep investment capped at 1–2 days.

### Deferred

| Item | Reason for deferral |
|------|-------------------|
| Full EEG band decomposition | Wait until relative delta power is validated & cross-site spectra are characterised |
| ECG HRV (SDNN, RMSSD, LF/HF) | High implementation cost (R-peak detection), ECG quality varies across sites |
| Spindle detection | Requires dedicated detector (e.g. A7, Luna); high complexity for uncertain gain |
| Set Transformer / Deep Sets | Attention-heavy, parameter-inefficient for 1,103 samples |
| Raw end-to-end model | 1.2 TiB data, fundamentally different pipeline — not in scope for this phase |

---

## Feature Enrichment Backlog

> **Note**: These items are lower priority than the [Official Phase Strategy](#official-phase-strategy-post-abstract-acceptance) P0–P4.  They represent incremental CAISR feature refinements that can be explored when the main strategy items are complete.

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

> **2026-08-06 update**: cross-center PSG literature gives per-record
> z-scoring direct evidence **for** cross-site robustness (+11–20% F1 on CAP
> A-phase detection when train/test populations differ, EMBC 2023; standard
> practice in a 8-database cross-center study).  Submission 5's failure was
> confounded (3 changes bundled, unofficial 50%-prevalence regime, S0001
> dominated everything).  **Re-ranked as P5-2: single-factor ablation on the
> I0006-holdout proxy.**

---

## Experiment Log

> **Official phase baseline reset**: All unofficial-phase experiments used 780 records with a 3–7 year CI window.  The official phase uses 1,103 records with a 1–6 year CI window.  The old baseline (sub3, AUROC 0.555) is **not** directly comparable to official-phase scores.  A new baseline must be established first.

Template for tracking training runs.  Fill in one row per experiment.

| ID | Date | Model | Feat Dim | New Features | lr / bs / epochs | Val AUROC | Age-cond AUROC | Δ vs Baseline | Notes |
|:--:|------|-------|:--------:|-------------|------------------|:---------:|:--------------:|:------------:|-------|
| **O0** | 2026-08-01 | `EpochCRNN_M` | 21 | (baseline) | 3e-4 / 16 / 100 | **0.833** | **0.762** | — | Official phase baseline (binary-arousal, 6,600 records, 5,280/1,320 split). Best @ epoch 51 (early stop 71). Per-site: S0001=0.840, I0006=0.786, I0002=0.791. Age gap ~0.069. Run B (same config): 0.836/0.769 @ ep58. ⚠️ Saved `official_baseline` checkpoint (Run B, pre-fix) hit the best-state-dict snapshot bug — reloading it reproduces 0.824/0.744, not the CSV's 0.836/0.769; CSV numbers (training-log) remain trustworthy. |
| **O1** | 2026-08-02 | `EpochCRNN_M` + age-adv | 21 | age-adversarial head (GRL α=0.5, λ=1.0, before FiLM) | 3e-4 / 16 / 100 | 0.816 | 0.735 | −0.027 | ❌ Failed. CSV true best (ep45) still below O0; GRL shoved age signal into FiLM channel — r(age,prob) rose to +0.61, gap 0.106; degrades post-best (0.735 → 0.685 @ ep65, no plateau). (Saved BestModel also hit the best-state-dict snapshot bug.) |
| **O2** | 2026-08-02 | `EpochCRNN_M` + night-feat | 21 + 15 | night-level aggregation features (15 dims, fixed-divisor norm, late fusion before FiLM) | 3e-4 / 16 / 100 | 0.817 | 0.738 | −0.024 | ❌ Failed. Best @ epoch 21 (early stop 41). Full-night aggregates largely redundant with the epoch-sequence backbone's own summary; per-site only I0006 gained (0.786→0.818), S0001 (0.840→0.818) and I0002 (0.791→0.750) lost. Literature hypotheses (TST/SE/arousal-index negative findings in Yaffe 2011 / Blackwell 2014) correctly predicted weak signal. P1 route closed. |
| **O3** | 2026-08-03 | `EpochCRNN_M` pw∈{2,4,8,16} | 21 | pos_weight sweep (P2 calibration; default 12.16 = prevalence-matched) | 3e-4 / 16 / 100 | 0.825–0.827 | pw4 0.761 / pw8 0.755 / pw2 0.750 / pw16 0.731 | ≈0 | ❌ No gain over default. Best pw=4 (0.7607 @ ep74, stop 94) ties O0 Run A (0.762, within noise), below Run B (0.769); pw=16 collapses fast (0.731 @ ep17, stop 36). Per-site pw=4: S0001=0.844, I0006=0.783, I0002=0.751. Default 12.16 stays. Temperature/Platt (analysis-only, `scripts/calibrate.py`): AUROC rank-invariant by construction; Platt ECE 0.033→0.014, mean_p→prevalence. P2 route closed. |
| **O0repro** | 2026-08-03 | `EpochCRNN_M` | 21 | baseline re-run on the **new multi-factor canonical split (= 5-fold fold_0)** | 3e-4 / 16 / 100 | 0.845 | 0.758 | — | Baseline on the new split (best @ ep42, early stop 62). Same split/config as O5's fold_0 — the reference for all post-2026-08-03 experiments. Per-site: S0001=0.855, I0006=0.795, I0002=0.822. |
| **O4** | 2026-08-03 | `EpochCRNN_M` + no-age | 21 | zero the age channel in FiLM demographics (age is constant within each age-stratum → cannot help within-stratum ranking) | 3e-4 / 16 / 100 | 0.816 | 0.760 | ≈0 (vs O0repro +0.002, vs O0 −0.002) | ❌ Failed. Best @ ep44 (early stop 65). age-cond ties both baselines within noise; plain AUROC clearly below O0repro (0.816 vs 0.845). Zeroing the age channel neither helps nor hurts ranking — the model's within-stratum ranking was already age-independent (the age input powered only between-stratum shortcuts). Note: O4 trained on the pre-alias multi-factor canonical split (record composition differs slightly from fold_0); conclusion unchanged vs both references. Route closed. |
| **O5** | 2026-08-03→04 | `EpochCRNN_M` ×5 | 21 | 5-fold CV ensemble (multi-factor stratified split, equal-weight probability average; see §10.4) | 3e-4 / 16 / 100 | 0.822 | 0.717 | −0.041 (OOF, conservative single-model bound; fold-0 same-split +0.027 vs O0repro) | ✅ Complete. Per-fold best age-cond (monitor): 0.785/0.704/0.756/0.740/0.747 (folds 0-4, @ ep 37/15/48/26/22) — run-to-run spread ≈ 0.08. OOF aggregate (6600 recs, one prediction per record, checkpoints = best-by-monitor): AUROC 0.8223 / age-cond 0.7169; per-site S0001 0.836/0.738, I0002 0.730/0.653, I0006 0.802/0.682. fold_0 (0.785) vs O0repro (0.758, same split): +0.027, within expected init/shuffle variance — no systematic split artifact. OOF is the honest lower bound; the test-time 5-model average should be ≥. |
| **O6** | 2026-08-04 | `EpochCRNN_M` ×3 | 21 | leave-one-site-out (train 2 sites → eval full held-out site, 3 runs) | 3e-4 / 16 / 100 | 0.636 / 0.627 / 0.769 | **0.552 / 0.562 / 0.742** (S0001/I0006/I0002 held-out) | −0.186 / −0.120 / **+0.089** (vs O5 OOF per-site 0.738/0.682/0.653) | ✅ Complete. Held-out age-cond: S0001 0.552 (n_train=1461, best ep5), I0006 0.562 (n_train=5458, ep2), I0002 0.742 (n_train=6281, ep3). **Refutes the "I0002 (2.5× prevalence) is the sub1 drop source" hypothesis — held-out I0002 *beats* its in-distribution OOF.** Drop ordering tracks n_train (1461/5458/6281), so the S0001/I0006 drops confound domain shift with training-set size; all 3 models best at ep 2–5 with fast early stop; eval used center-crop (not sliding window, −0.007 scale). Implication: cross-site shift is real and asymmetric but not obviously site-identity-driven for I0002; hidden-val site-mix shift remains an uncontrolled risk, and the SessionID-bug fix remains the main lever for sub2. *(2026-08-06 update: the SessionID fix did NOT move the official score — sub2 0.592 ≈ sub1 0.617; official sub1/sub2 match the **I0006-holdout** profile (monopolar + CAISR-OOD ≈ unseen source I0004), so I0006-holdout, not S0001-holdout, is the primary local proxy going forward; n_train must be held constant across holdout runs.)* |
| **O7a** | 2026-08-05 | `EpochCRNN_M` + age-matched pairwise | 21 | age-matched pairwise hinge loss (λ=1.0, margin=0.5, tol=2y, memory bank 512) — direct optimisation proxy of age-cond AUROC | 3e-4 / 16 / 100 | 0.831 (full-train eval) | 0.719 (val best @ep29, stop 49) | −0.039 (val; full-train −0.0185) | ❌ Failed. Full-train official-flow eval (scripts/eval_all_models.py): AUROC 0.8312 / age-cond 0.7340 vs baseline (o0_repro_split) 0.8417/0.7525 — worse on every site. Degrades fast after ep29 (0.719 → 0.65), unstable training. Pairwise (λ=1.0, margin=0.5) fights the BCE signal rather than helping — the BCE already learns age-matched ranking, pairwise adds noise on this small dataset. Route closed unless retried with much smaller λ/margin (low priority). |
| **O7b** | 2026-08-05 | `EpochCRNN_M` + focal | 21 | focal loss (γ=2.0, FocalBCEWithLogitsLoss, same pos_weight 12.16) + label_smoothing 0.05 | 3e-4 / 16 / 100 | 0.867 (full-train eval) | 0.784 (val best @ep67, stop 87) | **+0.049** (full-train 0.8018 vs 0.7525) | ✅ **Strong gain — new default.** Full-train official-flow eval: AUROC 0.8670 / age-cond 0.8018 / age-wtd 0.8138 / AUPRC 0.4282 (baseline 0.8417/0.7525/0.7666/0.3452); every site improves (S0001 0.7513→0.7989, I0006 0.7393→0.8189, I0002 0.7739→0.8274). Mechanism: at 7.6% prevalence the abundant easy negatives dominate BCE gradients; focal (1−pt)^γ concentrates them on hard samples. Accuracy 0.4141 / F1 0.2009 lower than baseline — the tuned binary threshold shifts (reward +0.2636 vs 0.2206); primary metric unaffected. **Adopted into the default config for sub3: focal ON + label_smoothing 0.05 + folds [0..4].** |
| **O8** | 2026-08-05 | `EpochCRNN_M` ×5 focal | 21 | sub3 config: 5-fold ensemble × focal+LS (§10.4 + O7b), early-stop floor `min_epochs=30` + patience 20→15 | 3e-4 / 16 / 100 | 0.863 (full-train) | 0.767 (per-fold val mean: 0.773/0.735/0.769/0.737/0.819); full-train 0.8045 | +0.052 (full-train vs baseline 0.7525); +0.0027 (vs O7b single 0.8018) | ✅ **Complete; submission config for sub3.** Full-train official-flow eval: AUROC 0.8632 / age-cond 0.8045 / age-wtd 0.8148 / AUPRC 0.4305. Per-site age-cond: S0001 0.8244, I0006 0.7692, I0002 0.7527. The early-stop floor (countdown starts at ep 30) exists so an early lucky-spike best can't cut a fold off mid-climb — fold_4, which suffered exactly that in the pre-floor run (best 0.708@ep9, stop@29), now trains to ep 59 with best 0.7364@ep44 (+0.028, the largest single-fold gain). Enough to put the ensemble *above* O7b single even on the leaked full-train eval — the single-vs-ensemble tradeoff is resolved in the ensemble's favour. |
| **A1** | 2026-08-09 | `EpochCRNN_M` (sub3 cfg) ×3 | 21 | O6 rerun under the sub3 config (focal+LS+early-stop floor) — same LO-site protocol | 3e-4 / 16 / 100 | — | **0.638 / 0.594 / 0.703** (I0006/S0001/I0002 held-out; I0002 run ×3: 0.704/0.724/0.683) | **+0.076 / +0.042 / ≈0** (vs O6 BCE config) | ✅ **The sub3 config change transfers cross-site.** I0006-holdout (the I0004-like proxy: monopolar + CAISR-OOD) 0.562→0.638 → sub3's official score should beat sub2's 0.592 (score due within ~72 h of the 08-08 submission). 3× I0002 repeats give run-to-run noise ±0.02–0.04 → B-wave intervention pass line Δ>0.04. Proxy pipeline: `/tmp/lo_site/lo_site_b1.py` (extends O6's script with `--train-version small\|large`, `--eval-version`, `--max-train`, `--normalize`). |

### Ablation protocol

For each new feature / change:

1. Start from the locked official-phase baseline config (`EpochCRNN_M`, binary-arousal 21-dim, all hyperparams as sub3).
2. Make exactly **one** change.
3. Train for 100 epochs; record val AUROC, **age-conditioned AUROC**, and per-site AUROC.
4. If Δ > 0: accept the change and update the baseline.
5. If Δ ≤ 0: reject, document the hypothesis for why it failed, move on.

---

## Key Dates & Deadlines

| Date | Event |
|------|-------|
| 2026-04-09 | Unofficial phase end |
| 2026-04-15 | CinC abstract deadline |
| 2026-06-03 | Official phase begins |
| 2026-08-20 | **Official phase final submission deadline** |
| 2026-09-01 | 4-page preprint deadline |
| 2026-09-20–23 | **CinC 2026, Madrid** |
| 2026-10-10 | Final 4-page paper deadline |
