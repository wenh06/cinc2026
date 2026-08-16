# CinC 2026 — Development Roadmap

> **Task**: Predict future cognitive impairment (MCI / Alzheimer's / dementia) from a single polysomnography night using the George B. Moody PhysioNet Challenge 2026 dataset.
>
> **Team**: Revenger  |  **Key Deadlines**: Official phase final 2026-08-20  |  CinC 2026 Madrid 2026-09-20–23

---

## Current Status (2026-08-12)

| Item | State |
|------|-------|
| Official submissions | 3/10 used: sub1 0.617, sub2 0.592, sub3 **0.611 (confirmed)** — within noise of sub1 (Δ0.006) and +0.019 above sub2; proxy 0.638 discounted −0.027 (same order as the old-config proxy error 0.562 vs 0.59–0.62).  Reward −0.130 is not monotonic in age-cond (sub1 0.617→+0.027, sub2 0.592→−0.013) and is not the primary metric |
| Official val / test | Unseen sources **I0004** (val) / **I0007** (test); metric = age-conditioned AUROC; leaderboard top ≈ 0.773 |
| Local proxy | **I0006-holdout** (monopolar + CAISR-OOD ≈ I0004 profile); A1 baseline **0.6375** under the sub3 config; run-to-run noise ±0.02–0.04 → pass line Δ > 0.04 |
| B-wave | ❌ all refuted: small pool (−0.092), z-score flat (−0.004), n_train not the S0001 driver |
| C-wave probes | ✅ real I0004 features: arousal ×5–6, **limb ÷100**; I0007 = opposite extreme → per-site empirical harmonisation is the only feature-side fix.  ✅ raw spectra overlap across all 5 sites → no spectral site confound.  ❌ model-side TTA refuted (entropy-min −0.069) |
| C-wave ComBat | ❌ all 3 variants refuted on the proxy: empirical n=10 recs −0.023 / −0.026, two-pass transductive −0.005.  Distributional repair on real I0004/I0007 is perfect (both land back in training range) yet the proxy score never gains — feature harmonisation cannot recover the drop.  **ComBat out of sub4** |
| Decision gates | **sub4 = sub3 config unchanged** — O7c λ=0.05 rerun refuted (0.6664 → 0.6199, Δ0.047 ≈ 2× noise floor; seed 1 even below baseline −0.018).  sub4 re-submits sub3's config to (1) measure official-side evaluation noise, (2) give the proxy→official discount a second anchor.  No new method adopted since D1's rerun lesson.  **Next experiment: P3 delta power** (only untried new-information channel; large-raw download is the gate) |
| D-wave (08-11→12) | O-wave re-tests on the proxy: D1 no-age 0.6616 → rerun 0.6392 (noise, not adopted); D2 night +0.002 ≈ 0; D3 drop −0.050 ❌; D4 aug +0.016 ~.  **O7c** (tanh pairwise + age-stratified sampler): λ=0.05 **+0.029**, λ=0.1 −0.031, λ=0.3 −0.022, λ=0.1×no-age −0.004 — pairwise helps only at minimal weight (consistent with O7a's λ=1 failure).  **warmup-cosine scheduler** +0.009 (0.6469, noise-adjacent).  O7c λ=0.05 **rerun refuted** (0.6664 → 0.6199, Δ0.047; seed 1 below baseline) — pairwise helps only at minimal weight but even that is seed noise; λ=0.05 **not adopted** |

---

## Current Status (2026-08-15)

| Item | State |
|------|-------|
| Official submissions | **4/10 used**: sub1 0.617, sub2 0.592, sub3 0.611, **sub4 ID 2620 (08-14, sub3 config unchanged) — processing, score pending**; sub4 anchors official-side evaluation noise |
| Leaderboard (08-15) | top age-cond **0.847 (Matcha, small)**; ReCognition 0.795, Leicester Fox 0.774, bashlab_wpi 0.751; Revenger 0.617 |
| Docker / CI | docker-test **green** (run 31875296556, 20m53s); opt-in Phi inference test on a 30-min segment passes in ~86 s — the full-night wavelet stage (~9–18 GB intermediates) would OOM the 7 GB runner, so CI truncates the signal but pads the spectrogram back to the canonical 11 h |
| P3 spectral (D1) | 641-dim Ye-2023-style extractor done + deterministic; full 1,103-record `tmp/spectral_features/features.csv`; univariate age-cond AUC: **N1 θ/α 0.63–0.64** (3 sites consistent), N3 delta / edge95 / REM coherence weak positives |
| D2 tabular (08-15) | small-set I0006-holdout (911 train / 192 eval): **LightGBM on 641-dim 0.655 ± 0.019** (15 seeds, min 0.626), XGB 0.653 ± 0.012, RF 0.576; the 390-dim `spec` block alone 0.659 ± 0.007, CAISR `arch` block 0.576, meta-only LR 0.601 — **+0.109 over the small-pool CRNN (0.546)**; large-pool comparable validation pending |
| P4 Philosopher's Stone | submodule pinned `0b1b49a`; 2.4 GB checkpoint (SHA `b2a9…5af87`) baked into the image + sha-verified; cache-extract script smoke 3/3 on GPU (~78 s/record); full 1,103-record cache pending AutoDL 5090 (~4–8 h) |
| Decision gates | sub4 already in; D2-small passed (0.655 ± 0.019 vs 0.546) → next gate = **tabular on the large set** (A1-comparable, 0.6375) before touching the CRNN training loop |

## Next Steps (2026-08-15, deadline 08-20)

1. Read sub4/2620 score (~08-18): Δ(sub4, sub3) = official-side noise, the second anchor for the proxy→official discount (only −0.027 so far).
2. ~~D2 tabular baseline (small)~~ ✅ **done** (0.655 ± 0.019 vs small-pool CRNN 0.546).  Next: (a) tabular on the LARGE CAISR features — locally computable, directly comparable to A1 0.6375; (b) decide whether to wire the spectral+GBDT pipeline into `team_code.py` (organisers extract from raw on their side) for sub5.
3. AutoDL 5090: run the full Phi cache; wire cache read + on-the-fly fallback into `team_code.py`.
4. Candidate methods (single-factor + seed rerun, Δ > 0.04): age-gated ranking loss, CORAL/SAM, survival, CreationTime metadata, raw spectral bank — detail in `tmp/agecond-improvement-research-2026-08-15.md` (uncommitted).
5. sub5 by 08-17/18 to stay inside the 72 h feedback window; confirm whether final ranking uses the last or the best submission.
6. Paper 09-01 (4-page preprint); keep the cross-site robustness narrative.

---

## Current Status (2026-08-16)

| Item | State |
|------|-------|
| Official submissions | **4/10 used**; sub4/2620 (sub3 config unchanged) still pending (≤72 h → ~08-17); best = sub1 0.617 |
| Docker / CI | **green** for `53a27ce` (run 31933319823): sub5 tabular default entry, Dockerfile MEGA feature-cache bake + `post_docker_build.py` SHA verification, `test_tabular` all pass |
| **sub5 locked** | small-pool tabular: **XGB + 390-dim `spec`, `include_meta=False`** (`cfg.py` default now `enable=True`). 15-seed validation: I0006-holdout 0.6446±0.0144, S0001-holdout 0.6473±0.0076, I0002-holdout 0.7486±0.0318 (small-pool CRNN = 0.546); adding age/sex/bmi meta *hurts* (~−0.01).  Submission plan: train set = **small**, submit 08-17/18 |
| D2-large gate | **FAILED / rejected**. Large-pool CAISR-only `arch`+meta XGB = 0.6452±0.0084 vs A1 0.6375 on the full I0006 eval — but the edge is a **rec_year follow-up-window artifact** (see below), not signal |
| rec_year trap | negative requires ≥6 y follow-up ⇒ records from ~2019+ are almost all positive (2019: 89%, 2020+: 100%) and every local holdout carries this tail. Official val **I0004 = 2004–2016**, test **I0007 = 2011–2017** have no such tail (supplementary demographics), and the official test prevalence 5–15% confirms it. Year-restricted eval (≤2017) collapses arch+meta to 0.5931 and rec_year-only to 0.6410.  **Rule: never ship rec_year; year-restrict any eval of models that use it** |
| P4 Philosopher's Stone (08-16) | full-signal CWT peaks ~60–70 GB → OOM-killed the 62 GB shared box (bare `Killed`).  Built the **hybrid low-memory wavelet stage** (`utils/phi_preprocess.py`): ≤2 Hz rows computed on the full signal with the upstream non-vectorised `cwt` (bit-identical); >2 Hz rows computed on overlapping 40-min chunks (10-min overlap) and stitched.  Fixed a stitching time-axis bug (`right = t1 - t_end`).  Validation vs the full-signal reference on 3 full nights: latent max|Δ| ≈ 1e-4–5e-4, correlation 1.000000.  **66 s/record, 14.2 GB peak** (vs 2–4 min / 60–70 GB) → full 1,103-record cache running locally, 2 workers, ~10 h |
| Next gates | Phi PCA-64 + ranker on the I0006 proxy (Δ>0.04 over 0.546) once the cache completes; sub5 submission |

## Current Status (2026-08-16, night)

| Item | State |
|------|-------|
| **Component registry** | submissions can now bundle several independently trained models with a per-record fallback chain.  `TrainCfg.components` (default = the single `sub5_tabular_xgb`) trains each component into `model_folder/components/<name>/` and writes `model_manifest.json`; `load_model` routes from the on-disk manifest (not the current `TrainCfg`), and `run_model` walks components in priority order — a failing primary (e.g. a Phi ranker on a no-C4 montage) falls through to the next, with sub5's tabular XGB as the montage-agnostic terminal fallback.  `TrainCfg.tabular.enable` stays as the legacy master switch; old layouts load unchanged.  `test_docker.test_model_components` covers manifest round-trip, e2e train/load/run and the fallback chain |
| Feature-cache gap fixed | the Dockerfile-baked MEGA spectral cache was **never wired at runtime** (`feature_cache` default empty).  `component_registry.resolve_feature_cache` now auto-resolves `data/spectral_features/features.csv` when no explicit path is set; official mounts land on `training_data` / `holdout_data`, so `/challenge/data` is not shadowed — train hits the cache, misses (organiser-added records) extract on the fly |
| Official runtime constraints | verified against `official_baseline` README: data at `/challenge/training_data` + `/challenge/holdout_data`, only `/challenge/model` writable, no network — the registry only writes under `model_folder` and performs no network I/O |
| Phi full cache | chunked 2 workers running; 5 records failed = **montage without any C4** (5/54 I0002 + 1 S0001, ≈0.5% of 1,103); `_resolve_c4m1` has no fallback for those.  Decision: drop them (NaN rows), no C3-M2 fallback (distribution shift risk).  Data-folder-only rule for training: cache is looked up per mounted row — organisers' add/remove perturbations change the trained model as they expect |
| Local e2e | full 1,103-record tabular training (on-the-fly) running in tmux `e2e` (~23:15); run_model + evaluate to follow on the produced model |

## Next Steps (2026-08-16, night)

1. Merge dev → master once the component-registry docker-test CI is green (official evaluation pulls master); sub5 submit 08-17/18, training set = small.
2. Phi cache completes (~08-17 morning) → PCA-64 + XGB ranker on the I0006 holdout, adopt only on Δ>0.04; then add as a `phi` component with the sub5 tabular fallback.
3. Read sub4/2620 (~08-17): second anchor for the proxy→official discount.
4. Paper 09-01 (4-page preprint); keep the cross-site robustness narrative.

## Next Steps (2026-08-16, deadline 08-20)

1. sub5 submit **08-17/18**, training set = small (config already default; CI green; full 1,103-record end-to-end training running locally for a final dry-run).
2. Phi cache completes (~08-17 morning) → PCA-64 + XGB ranker on the I0006 holdout (`scripts/phi_pca_ranker.py`), adopt only on Δ>0.04; candidate stream for a later submission, not sub5.
3. Read sub4/2620 (~08-17): second anchor for the proxy→official discount.
4. Paper 09-01 (4-page preprint); keep the cross-site robustness narrative.

---

## Data Facts

| Fact | Unofficial Phase | Official Phase |
|------|:---------------:|:--------------:|
| Training records (total) | 780 | 1,103 (small) / **6,600 (large)** |
| Records with CAISR | 766 (14 missing) | 1,090 small / 6,587 large (13 missing) |
| CI positive (training) | 392 / 780 (50.3%) — artificially balanced | 84 / 1,103 (7.6%) small / 498 / 6,600 (7.5%) large — prevalence-matched |
| Estimated CI rate (hidden test) | ~6% | **5–15% (challenge page, confirmed)** |
| Age (mean ± std) | 70.4 ± 8.3, [50, 89] | 62.0 ± 8.5, [50, 88] |
| Sex (M / F) | 60% / 40% | 53% / 47% |
| Site distribution | S0001: 572 (73%), I0006: 154 (20%), I0002: 54 (7%) | small: S0001 857, I0006 192, I0002 54; large: **S0001 5,139 (78%), I0006 1,142 (17%), I0002 319 (5%)** |
| CI time window | 3–7 years post-PSG | **1–6 years** post-PSG |
| Primary metric | AUROC | **Age-conditioned AUROC** |
| Local val size | ~156 (80/20) | 1,320 (canonical 5-fold val, large) |
| **Official validation cohort** | — | **I0004 — a single unseen source, hidden** (CAISR-held-out; 5–15% CI; no human annotations) |

> Only 116 records are shared between phases — the official phase is a fresh dataset; the 50% → 7.6% CI-rate change is dataset re-composition, not label change.
>
> **Montage & CAISR domain heterogeneity** (the cross-site mechanism behind the official-val gap):
> - S0001 & I0002: **bipolar mastoid-referenced** (`F3-M2, C3-M2, O1-M2`); I0006: **monopolar** (`C3, O1, F3` + separate M1/M2).
> - CAISR was trained on **S0001 (MGH) + MESA/MrOS/SHHS** — S0001 is in-distribution; I0002/I0004/I0006/I0007 are CAISR-held-out, with montage + hardware differences (I0004: mixed hardware generations; I0007: lower-bandwidth system).
> - Our model (78% S0001 training weight) is calibrated to the S0001 feature regime → LO-site I0006-holdout (0.562, monopolar + CAISR-OOD) ≈ official scores (0.59–0.62), while I0002-holdout (0.742) transfers far better.
> - **2026-08-09 probe on the real inference domain (C1)**: I0004 arousal_frac 0.304 vs 0.045–0.069 (×5–6), limb_fracs ≈0.0005 (÷100), Wake prob ↑ (0.380 vs 0.24–0.29); **I0007 is the opposite extreme** (arousal 0.012, limb 0).  Event-count features are the most corrupted; the two unseen sites flank the training mean → no fixed correction can fix both, per-site empirical harmonisation is the only feature-side fit.

---

## Approach Overview

**`EpochCRNN_M` + binary-arousal 21-dim CAISR features** on the large set (6,600 records); each night → N × 30-s epochs → epoch-sequence model → single binary CI prediction.  Same-site baseline O0 = 0.762 age-cond; default config = **O8 5-fold focal+LS ensemble** (0.8045 same-site full-train; but see P5 — same-site numbers do not transfer).  *(Unofficial best: sub3, AUROC 0.555.)*

**Why CAISR features**: available for all splits (organisers pre-ran CAISR); 21 floats/epoch vs ~36 M raw samples/night (memory) — but **"site-agnostic" is partially refuted (2026-08-06)**: the canonical space still shifts by site (Data Facts), making cross-site robustness a first-class problem.  Records missing CAISR annotations (no EEG/EOG/EMG) fall back to `(0, 0.5)`.

**Feature layout** (binary-arousal, 21 dims):

| Indices | Content | Dims | Source |
|:------:|---------|:---:|--------|
| [0:6] | Sleep stage one-hot (N3, N2, N1, REM, W, Unknown) | 6 | `stage_caisr` |
| [6:11] | Stage softmax probabilities (re-normalised) | 5 | `caisr_prob_*` ÷ 9 |
| [11] | Arousal fraction | 1 | `arousal_caisr` mean per epoch |
| [12:17] | Respiratory event fractions (OA, CA, MA, HY, RERA) | 5 | `resp_caisr` |
| [17:19] | Limb event fractions (isolated, periodic) | 2 | `limb_caisr` |
| [19:21] | Sin/cos time-position encoding | 2 | Computed |

---

## Official Phase Strategy (post-abstract-acceptance)

> **Priority shift**: the official metric is **age-conditioned AUROC**, which penalises "age → CI" shortcuts; also a prevalence-shift calibration gap (train 50% → test ~6%).

### P5 — Cross-site robustness (current priority, 2026-08-06) 🔴

Official sub1/sub2 (0.617/0.592) show the real task is generalising to **unseen sources** (val = I0004, test = I0007) under CAISR-domain + montage/hardware shift; same-site val (0.76–0.80) does not transfer.  Local proxy = **LO-site I0006-holdout** (≈ I0004's profile; 0.562 ≈ official 0.59–0.62), n_train held constant for comparability.

Candidate interventions (evidence × cost ranking; P5-6 in progress):

1. ~~Small-set training~~ — ❌ **refuted (B1)**: −0.092 on the I0006 proxy.  Data volume beats overfitting risk cross-site.
2. ~~Per-record z-score~~ — ❌ **flat (B2)**: −0.004 on the proxy (+0.048 only on I0002).  Fixes amplitude calibration, not the CAISR-OOD mean shift.
3. **Mixup / feature-noise / epoch-masking augmentation** — sleep evidence (XSleepFormer; BEETL 2021).
4. **Model selection on the LO-site proxy** (early-stop/checkpoint by held-out-site metric) — directly optimises the cross-site objective.
5. **Stronger regularisation** (wd / dropout / smaller capacity) — cross-institutional EHR: simple models + site control ≥ deep nets.
6. ~~ComBat / NeuroHarmonize feature harmonisation~~ — ❌ **refuted (C4)**: per-site mean/var standardisation fit on the 2 training sites; unseen-site effect estimated empirically (10 records mimicking the supplementary examples, or two-pass over the whole eval dir).  All 3 variants negative on the proxy (−0.023 / −0.026 / −0.005); distributional repair on real I0004/I0007 is exact yet the score never gains → feature harmonisation cannot recover the drop.
7. **Multi-architecture soft voting** (CRNN + Transformer) — SOMNUS: soft voting beats every single model in 94.9% of comparisons.
8. ~~Model-side TTA~~ — ❌ **refuted (C3)**: BN-stat −0.011, entropy-min −0.069.  Matches the literature: gradient-based model-side TTA frequently *degrades* EEG models ([NeuroAdapt-Bench 2026](https://huggingface.co/papers/2604.16926)); the positive EEG-TTA result ([PSDNorm 2025](https://ar5iv.labs.arxiv.org/html/2503.04582)) is *input-side* normalisation — precisely our surviving option (item 6).
9. **Site-adversarial GRL** — mixed evidence; O1 (age-adv) failed; last resort.
10. **Raw-PSG spectral features (P3, current priority 08-13)** — ✅ probe (C2): spectra overlap all 5 sites → no site confound, features viable.  **Deep rationale (C1+C2)**: the official-val drop's only identifiable mechanism is CAISR *event-count* drift (arousal ×5–6, limb ÷100 on I0004) — spectral features are the one input channel that is *stable* cross-site, sidestepping the CAISR drift source.  CAISR is a classifier output: no spectral power, no waveform morphology (spindles), no ECG/SpO₂ (HRV/ODI) — the literature's strongest sleep-cognition biomarkers sit in its blind spots.  Official baseline computes physiological stats in `run_model` → raw is available at inference.  **Data reality (08-13)**: small-set full raw (1,103 recs, 214 GB) on `/Data1/physionetchallenge2026data`; supplementary I0004/I0007 raw (20 recs) on `/Data1/cinc2026supplementary`; **large-set raw (6,599 files, ~1.2 TB) not downloaded** (kaggle; ~8–20 h at typical bandwidth; /Data1 1.2 T free → stream extract-then-delete).

**Not recommended**: site-ID input (useless for unseen sites); BBSE/prior-shift (rank-invariant); SAM (no sleep evidence).

---

### P0 — Age-conditioned AUROC monitoring ✅

r(age, prob) = +0.39 (2026-08-02); age-cond AUROC added to the trainer; **age-adversarial head (O1) failed** — GRL pushed age into the FiLM channel, r(age,prob) rose to +0.61; retry would need λ≫1 with mechanism risk.  [ ] age-stratified sampling (open).

### P1 — Night-level aggregation features ❌ closed (O2)

15-dim night-level branch implemented (`build_night_features`, late fusion); **O2: −0.024 age-cond** — redundant with the epoch-sequence backbone (only I0006 gained).  Route closed; details + evidence grades in Phase 9.

### P2 — Calibration ❌ closed (O3)

pos_weight sweep {2,4,8,16}: **no gain over default 12.16**; temperature/Platt: **AUROC rank-invariant** by construction (only Brier/ECE move).

### P3 — Spectral features 🔴 current priority (08-15)

**Why now**: 08-04→08-13 interventions (B/C/D/O7 waves) all reshuffled the same 21-dim CAISR input + CRNN_M skeleton; raw-spectral features are the only untried **new-information** channel and the only cross-site-**stable** one (C2: overlap) where CAISR event counts are most corrupted (C1: ×5–6).  Feasibility gated on large-set raw availability (small-set + supplementary raw are local).

- [x] Cross-site spectra probed (C2, 08-09): overlap → **site-level harmonisation not required**.
- [x] **641-dim Ye-2023-style extractor** (D1, 08-15): staged relative powers/ratios, kurtosis, Hjorth, I-CARE quantiles, coherence, HRV/SpO2, CAISR architecture, time metadata — deterministic; 1,103-record cache extracted.
- [x] Univariate age-cond screening: N1 θ/α 0.63–0.64 across sites; N3 delta / edge95 / REM coherence weak positives.
- [x] **D2 (small)**: LightGBM on the 641-dim bank 0.655 ± 0.019 (15 seeds) vs small-pool CRNN 0.546; the `spec` block alone 0.659 ± 0.007; XGB agrees (0.653), RF weak (0.576).
- [ ] **D2 (large, A1-comparable)**: tabular on the large set — CAISR-derived features are local (no raw); the full spectral bank needs the 1.2 TB large raw.

### P4 — Philosopher's Stone (BDSP pretrained sleep EEG model)

**Status (08-15)**: fully wired — submodule pinned `0b1b49a`; 2.4 GB checkpoint baked into the Docker image with SHA-256 verification (`post_docker_build.py`); `scripts/phi_cache_extract.py` batch extraction (resume / multi-process / deterministic order, I0006 monopolar C4−M1 derivation); cache loader with on-the-fly fallback in `utils/phi_cache.py`.  GPU smoke 3/3 (~78 s/record); CI opt-in test green (~86 s, truncated segment).

Usage plan: full 1,103-record cache on AutoDL 5090 (~4–8 h) → 1024-D latent → PCA-64 → frozen feature extractor, evaluated under the Δ > 0.04 gate.

---

## Official Phase Submission History

| # | ID | Date | Config | Age-cond AUROC | Reward | Notes |
|:--:|:---:|------|-------|:-------------:|:------:|-------|
| 1 | 2372 | 08-02 | `EpochCRNN_M` single, BCE, pos_weight 12.16 | 0.617 | 0.027 | SessionID int/str bug (local-eval artefact only — see §10.5) |
| 2 | 2407 | 08-05 | sub1 training + SessionID fix + sliding-window inference | 0.592 | −0.013 | ≈ sub1 within noise — official scorer uses its own ages |
| 3 | 2471 | 08-08 | 5-fold focal+LS ensemble, early-stop floor (O8) | 0.611 | −0.130 | within noise of sub1 (Δ0.006), +0.019 vs sub2; proxy 0.638 discounted −0.027 |
| 4 | 2620 | 08-14 | sub3 config unchanged (noise anchor) | — | — | submitted 22:33 EDT; processing — score and Δ(sub4, sub3) pending |

> Official val = unseen **I0004**; leaderboard top ≈ 0.847 (08-15).  Full detail in `submissions` (sub1–3 confirmed; sub4 processing).

---

## Unofficial Phase Submission History

| # | ID | Date | Model | AUROC | Feature Set | Notes |
|:--:|:---:|------|-------|:-----:|-------------|-------|
| 1 | 1173 | 04-03 | `EpochTransformer_M` | 0.522 | `binary_arousal` (21d) | |
| 2 | 1192 | 04-04 | `EpochCRNN_resnetNC_BNse_M` | 0.497 | `binary_arousal` (21d) | bs=64 likely caused convergence issues |
| 3 | 1240 | 04-07 | **`EpochCRNN_M`** 🏆 | **0.555** | `binary_arousal` (21d) | wd=1e-2, LS 0.05 |
| 4 | 1270 | 04-08 | `EpochTransformer_L` | 0.491 | `binary_arousal` (21d) | 4.8 M params — overfitted |
| 5 | 1348 | 04-09 | `EpochCRNN_M` | 0.448 | `arousal_prob_stats` (21d) ⚠️ | bundled: prob-stats + z-score + no time-enc |

**Key insight**: parameter count ↔ AUROC inversely correlated (sub3: 437K→0.555 > sub1: 826K→0.522 > sub4: 4.8M→0.491) — small-data regime; capacity constraint beats expressivity.  (Rank 102/244.)

---

## Unofficial Phase Feedback

**1. Richer sub-epoch statistics can *degrade* performance** — sub5 (prob mean/std/max + z-score + no time-enc, all bundled) dropped 0.555 → 0.448.  Prob channels are noisier than binary labels; z-score destroys between-site variation; **confounded changes** make the cause unidentifiable.  *(2026-08-09 B2: single-factor z-score on the official proxy is flat — sub5's failure was mostly the feature-set change.)*  **Principle: single-factor ablations only; add independent information, not re-parameterisations.**

**2. CAISR-to-CI information bottleneck** — missing biomarkers with established links: NREM delta power (strongest), spindle density, spindle–SW coupling, HRV, SpO₂ depth, sleep-architecture summaries (computable from CAISR).  → P3 (spectral), P1 (night features, closed), P4.

**3. Prevalence-shift calibration gap** — train 50% vs hidden ~6%: local val AUROC ≈ 0.70 vs leaderboard 0.555.  (O3: pos_weight no gain; temperature/Platt rank-invariant → closed.)

**4. Smaller models generalise better** — smallest (437K) won in the 780-record regime.  *(2026-08-09 B1: on the cross-site proxy, data volume beats capacity-constraint regularisation.)*

---

## Design Decisions & Literature

- **Why epoch-sequence models**: CI alters sleep *architecture* (SWA front-loading gradient, arousal periodicity/CAP, stage-transition fragmentation) — temporal organisation, not just totals; a CRNN/Transformer over epochs captures this.
- **Why FiLM for demographics**: learned affine modulation by (age, sex, BMI) instead of concatenation — the same N3% can mean different things at 45 vs 85; age is the strongest CI risk factor.

**Spectral features (P3) literature**:

| Biomarker | Mechanism | Key refs |
|-----------|-----------|----------|
| NREM delta power (0.5–4 Hz) | Glymphatic Aβ clearance in SWS; reduced SWA predicts decline | Xie 2013 *Science*; Ju 2017 *Brain*; Mander 2016 *Neuron* |
| Spindle density (12–15 Hz) | Thalamocortical consolidation; reduced fast spindles in MCI/AD | Gorgoni 2016; Winer 2019; Mander 2016 |
| Spindle–SW coupling | Replay precision; impaired in AD | Helfrich 2018 *Neuron*; Winer 2021 *Curr Biol* |
| HRV (SDNN, RMSSD, LF/HF) | Autonomic dysfunction in neurodegeneration | Lanfranchi 1999 *Circulation*; Toledo 2022 *SMR* |
| Theta/alpha ratio | EEG slowing = cortical dysfunction in early AD | Babiloni 2016; Rossini 2020 |

**Night-feature literature (P1, closed)** — grades (web-verified 2026-08-02): **strong** = AHI, sleep efficiency, WASO, PLMI, REM latency, N3%; **mixed** = REM%, arousal index; **negative** = TST; **inferred** = temporal-dynamics features (Borbély physiology).  Per-feature table with cohort details + ORs, caveats, and the full reference list: [Appendix — Night-Level Feature Evidence](#appendix--night-level-feature-evidence).

---

## Completed Phases — Archive

> History of what was built; canonical results live in the [Experiment Log](#experiment-log).  Key lessons kept inline.

**Phase 1 — Data exploration & pipeline** ✅: raw data explored; site heterogeneity documented in `data_reader.py`; **`caisr_prob_*` EDF scale bug found & fixed** (physical-range header 9 instead of 1 → ÷9 + re-normalise); `const.py` / `cfg.py` / `dataset.py` (`build_epoch_features`, stratified split, `collate_fn` with `padding_mask`).

**Phase 2 — Model implementation** ✅: `EpochTransformer` (Pre-LN, sinusoidal PE, masked mean pooling, FiLM; S/M/L = 116K/826K/4.8M).

**Phase 3 — Training loop** ✅: AdamW-AMSGrad lr=3e-4, OneCycleLR, grad_clip 1.0, LS 0.05, early stop patience 20, best-checkpoint saving (hyperparams canonical in the `submissions` file).

**Phase 4 — CAISR-missing fallback** ⏳: 14/780 (unofficial) & 13/1,103 (small) records lack CAISR (no EEG/EOG/EMG).  Current: `(0, 0.5)` fallback in `run_model`; ECG-HRV MLP branch planned, **deferred** (cost ≫ expected gain).

**Phase 5 — Validation & analysis** 🔄: per-site AUROC tracked (gap ~0.05); age-dependence diagnosis → P0; calibration → P2.

**Phase 6 — Alternative models & raw pathway** 🔄: `EpochCRNN` (ResNet-N + BiLSTM, FiLM; S/M/L + backbone variants, all verified).  Foundation models (TimesFM/Chronos/Moirai/MOMENT) **not recommended** (scale mismatch); exceptions: Philosopher's Stone (→ P4), MOMENT-Small.  **Raw pathway reality check**: ~214 GiB for 1,103 records, ~91 M samples/night → recommended path = 30-s epoch spectral augmentation (P3), not raw e2e.

**Phase 7 — Submission pipeline** ✅: config-driven `team_code.py`, official API + `test_docker.py` + `sync_official.py`, data-layout support, mini/reduced subsets, O0 baseline run.

**Phase 8 — Spectral augmentation** 🔴 (→ P3, current priority 08-13): relative delta power, NREM-only, bipolar C3-M2 (derive for I0006); **why relative**: absolute power is hardware-dependent.  Raw-signal access at official inference **confirmed** (the official baseline computes physiological stats in `run_model`).

**Phase 9 — Night-level features** 🔜 (→ P1, closed): `build_night_features` (15 dims) implemented `2658f57`, **O2 failed** → closed.

**Phase 10 — Calibration & robustness**:
- **10.1** Age diagnosis → P0 (done; age-adv failed O1).
- **10.2** Calibration → P2 (closed).
- **10.3** LO-site experiment: held-out age-cond S0001 0.552 / I0006 0.562 / I0002 0.742 (O6, BCE); under sub3 config **A1: 0.594 / 0.638 / 0.703±0.02**.  **"I0002 is the sub1 drop source" refuted** (held-out I0002 ≥ in-distribution OOF).  Caveats: n_train differs across holdouts (N1 control 08-09: S0001's deficit is domain, not volume); models best at ep 2–5 under BCE (floor in sub3 config fixes).  Canonical numbers: log rows **O6/A1/B1/B2/N1**.
- **10.4** 5-fold ensemble: multi-factor stratified split (label × site × sex × age); `load_model` auto-detects and equal-weight averages; sliding-window inference for long nights (768/384, +0.013); OOF age-cond 0.7169 (conservative).  Log row **O5**.
- **10.5** sub1 + SessionID bug: `run_model` demographics silently fell back to defaults (int/str type mismatch) — measured +0.08 local; fix `7c1187a`.  **⚠️ Correction 08-06: local-eval artefact** — the official scorer reads ages from its own labels file; sub2 (fixed) 0.592 ≈ sub1 within noise; the real driver is the unseen val source.  *Lesson: pure-numeric IDs are a classic int-vs-str trap — pandas strict-type masks fail silently.*
- **10.6** O7 loss experiments: O7a pairwise failed; **O7b focal γ=2 + LS 0.05 = new default** (+0.049); sub3 config = 5-fold × focal+LS + early-stop floor (O8, 0.8045 full-train); merged via PR #15, submitted 08-08.  **Empty-batch crash found & fixed** (`4f267fb`): CAISR-missing records → empty epoch matrix → CNN conv-5 crash at small batch sizes; `_filter_missing_caisr()` drops them from train/val.
- **10.7** Research queue (deprioritised by the cross-site pivot): bs 16→32; `_M`→`_L` capacity.  Only re-test if capacity gains remain after the pivot.

---

## Appendix — Night-Level Feature Evidence

> The 15-dim feature list was defined internally (2026-07-30 review).  Every feature was
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
4. The **temporal-dynamics features** (#11–14, and partly #8, #10) are physiologically-motivated *hypotheses*, not validated biomarkers.  O2 (2026-08-02) tested exactly this — age-cond −0.024, no gain: the model already extracts the signal from the epoch sequence.

**Full reference list (verified 2026-08-02)**

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

## Feature Enrichment Backlog

> Lower priority than P0–P5; ⚠️ = tested in sub5 and harmful — single-factor retries only.

- Richer CAISR features: arousal burst pattern, resp event counts, limb run-lengths, stage-transition entropy.  ⚠️ `caisr_prob_arous` mean/std/max was sub5's key change (0.555→0.448) — retry mean-only, single-factor.
- Remove time-position encoding for CRNN ⚠️ (bundled in sub5; the recurrent backbone already tracks position) — retry single-factor.
- Per-record normalization ⚠️ — **2026-08-09 B2: not adopted** (flat on I0006, +0.048 only on I0002).

---

## Experiment Log

> Official-phase baseline reset: all unofficial-phase experiments used 780 records (3–7 y CI window); official uses 1,103/6,600 records (1–6 y).  Old baselines are not comparable.

| ID | Date | Model | Feat | Change | lr / bs / ep | Val AUROC | Age-cond AUROC | Δ vs Baseline | Notes |
|:--:|------|-------|:----:|--------|--------------|:---------:|:--------------:|:------------:|-------|
| **O0** | 08-01 | `EpochCRNN_M` | 21 | (baseline) | 3e-4/16/100 | 0.833 | **0.762** | — | Official baseline (6,600 recs, 5,280/1,320). Best ep51. Per-site S0001 0.840 / I0006 0.786 / I0002 0.791. Run B: 0.836/0.769. ⚠️ saved checkpoint hit best-state-dict snapshot bug (reload 0.824/0.744; CSV trustworthy). |
| **O1** | 08-02 | + age-adv | 21 | GRL age head (α=0.5, λ=1.0, before FiLM) | 3e-4/16/100 | 0.816 | 0.735 | −0.027 | ❌ GRL shoved age into FiLM channel — r(age,prob) rose to +0.61; degrades post-best. P0 route. |
| **O2** | 08-02 | + night-feat | 21+15 | 15-dim night aggregates, late fusion | 3e-4/16/100 | 0.817 | 0.738 | −0.024 | ❌ Redundant with backbone; only I0006 gained (+0.032). P1 closed. |
| **O3** | 08-03 | pw sweep | 21 | pos_weight {2,4,8,16} | 3e-4/16/100 | 0.825 | pw4 0.761 … pw16 0.731 | ≈0 | ❌ No gain over default 12.16. Temp/Platt rank-invariant by construction. P2 closed. |
| **O0repro** | 08-03 | `EpochCRNN_M` | 21 | baseline on multi-factor canonical split (= fold_0) | 3e-4/16/100 | 0.845 | 0.758 | — | Reference for all post-08-03 experiments. Per-site 0.855/0.795/0.822. |
| **O4** | 08-03 | + no-age | 21 | zero the FiLM age channel | 3e-4/16/100 | 0.816 | 0.760 | ≈0 | ❌ Within-stratum ranking was already age-independent; age powered only between-stratum shortcuts. |
| **O5** | 08-03→04 | ×5 | 21 | 5-fold CV ensemble (multi-factor split) | 3e-4/16/100 | 0.822 | 0.717 (OOF) | −0.041 | ✅ Per-fold 0.785/0.704/0.756/0.740/0.747; OOF 0.7169; fold_0 same-split +0.027 vs O0repro (no split artifact). OOF = honest lower bound. |
| **O6** | 08-04 | ×3 | 21 | LO-site (train 2 → eval held-out) | 3e-4/16/100 | — | **0.552 / 0.562 / 0.742** (S0001/I0006/I0002) | −0.186/−0.120/+0.089 | ✅ **"I0002 is the sub1 drop" refuted** (held-out I0002 beats its in-dist OOF). Drop ordering tracks n_train (1461/5458/6281) — confounded with domain; best at ep 2–5. *(08-06: official sub1/sub2 match the I0006-holdout profile — I0006-holdout is the primary proxy; n_train held constant across runs.)* |
| **O7a** | 08-05 | + pairwise | 21 | age-matched pairwise hinge (λ=1, margin=0.5) | 3e-4/16/100 | 0.831 | 0.719 | −0.039 | ❌ Fights the BCE signal; unstable. Closed unless λ≪1. |
| **O7b** | 08-05 | + focal | 21 | focal γ=2 + LS 0.05 | 3e-4/16/100 | 0.867 | 0.784 | **+0.049** | ✅ **New default.** Every site up; mechanism: easy negatives dominate BCE at 7.6% prevalence. Adopted for sub3. |
| **O8** | 08-05 | ×5 focal | 21 | sub3 config: 5-fold × focal+LS, floor min_epochs=30, patience 15 | 3e-4/16/100 | 0.863 | 0.767 fold-mean; 0.8045 full-train | +0.052 | ✅ **Submission config.** Floor fixed fold_4's early spike-cut (best 0.708@ep9→0.7364@ep44). Ensemble > O7b single. |
| **A1** | 08-09 | ×3 (sub3 cfg) | 21 | O6 rerun under sub3 config | 3e-4/16/100 | — | **0.638 / 0.594 / 0.703** (I0006/S0001/I0002; I0002 ×3: 0.704/0.724/0.683) | **+0.076 / +0.042 / ≈0** | ✅ **Config change transfers cross-site.** I0006 proxy 0.562→0.638 → sub3 official should beat 0.592. Noise ±0.02–0.04 → pass line Δ>0.04. |
| **B1** | 08-09 | ×3 (A1 cfg) | 21 | small (1,103) vs large train pool | 3e-4/16/100 | — | 0.546 / 0.558 / 0.700 | **−0.092** / −0.036 / +0.018 | ❌ **Refuted.** Small pool loses on both CAISR-OOD holdouts; data volume beats overfitting risk. |
| **B2** | 08-09 | ×3 (A1 cfg) | 21 | per-record z-score (B2a I0006 / B2b I0002 / B2c +small) | 3e-4/16/100 | — | 0.634 / 0.730 / 0.538 | −0.004 / +0.048 / −0.099 | ❌ **Not adopted.** Only I0002 gains; proxy flat — z-score fixes amplitude, not the CAISR-OOD mean shift. |
| **N1** | 08-09 | (A1 cfg) | 21 | n_train control: I0006 pool capped to 1,461 (= S0001's size) | 3e-4/16/100 | — | 0.618 | −0.019 | ✅ **S0001's deficit is domain shift, not data volume** (matched-n_train I0006 still +0.024 above S0001). |
| **C1** | 08-09 | — | 21 | Feature probe on the real inference domain (supplementary I0004/I0007) | — | — | — | — | ✅ **Diagnostic.** I0004: arousal ×5–6, limb ÷100, Wake prob ↑; I0007 = opposite extreme. Event counts most corrupted; per-site empirical harmonisation is the only feature-side fix. |
| **C2** | 08-09 | — | — | Raw spectral probe (C3(-M2), 12 recs/site + 20 supplementary) | — | — | — | — | ✅ **Diagnostic.** Band powers overlap all 5 sites (within-site σ ≈ between-site Δ) → no spectral site confound; delta-power features viable. Caveat: full-night (Wake unmasked), NREM-only refinement pending. |
| **C3** | 08-09 | (A1 ckpt) | 21 | TTA on 10 unlabelled held-out records: BN-stat / entropy-min / both | — | — | 0.627 / **0.569** / 0.568 (baseline 0.6375) | −0.011 / **−0.069** / −0.070 | ❌ **Refuted.** Matches [NeuroAdapt-Bench 2026](https://huggingface.co/papers/2604.16926) (gradient-based TTA degrades EEG); the positive [PSDNorm 2025](https://ar5iv.labs.arxiv.org/html/2503.04582) is input-side (C4). |
| **C4** | 08-09 | (A1 ckpt) | 21 | ComBat feature harmonisation, 3 variants: empirical site effect from 10 records (cols 6:19 / 11:19) and two-pass transductive over the whole eval set | — | — | 0.6143 / 0.6119 / **0.6326** (baseline 0.6375) | −0.023 / −0.026 / **−0.005** | ❌ **Refuted.** Distributional repair on real I0004/I0007 is exact (both land back in training range) yet the proxy never gains.  Feature harmonisation cannot recover the drop — sub4 = sub3 config unchanged. |
| **D1** | 08-11 | (A1 cfg) | 21 | no-age: zero the FiLM age channel (O4 re-tested on the LO-site proxy) | 3e-4/16/100 | — | 0.6616 / **0.6392** (rerun seed 1) | +0.024 / **+0.002** | ⚠️ **Not reproducible** — the +0.024 was within run-to-run noise (Δ between runs 0.0224 ≈ noise floor).  O4's same-site ≈0 + cross-site ≈0: no evidence the age channel hurts cross-site ranking.  **Not adopted; sub4 decision waits on O7c.** |
| **D2** | 08-11 | (A1 cfg) | 21+15 | 15-dim night-features late fusion (O2 re-tested on the proxy) | 3e-4/16/100 | — | 0.6394 | +0.002 | ❌ Neutral — O2's same-site verdict transfers to cross-site. |
| **D3** | 08-11 | (A1 cfg) | 21 | drop cols 11 (arousal) & 17:19 (limb) — C1's most-shifted blocks | 3e-4/16/100 | — | 0.5874 | **−0.050** | ❌ **Refuted.** The most-shifted columns are still signal cross-site; no per-site feature deletion. |
| **D4** | 08-11 | (A1 cfg) | 21 | two-sided domain-randomization on train features (I0004/I0007 shift dirs, p=0.5 each) | 3e-4/16/100 | — | 0.6535 | +0.016 | ~ Below pass line, not adopted solo; possible no-age+aug ensemble member later. |
| **O7c** | 08-12 | (A1 cfg) | 21 | age-stratified sampler (4 pos + 8 win-neg + 4 mixed-neg per batch) + tanh pairwise on logits, bank_size=0; λ sweep | 3e-4/16/100 | — | 0.6664 / 0.6065 / 0.6154 / 0.6335 (λ=0.05/0.1/0.3/0.1×no-age); **rerun λ=0.05 seed 1: 0.6199** | **+0.029** / −0.031 / −0.022 / −0.004 / **rerun −0.018** | ⚠️ **Refuted by rerun (Δ0.047 ≈ 2× noise floor)** — the +0.029 was seed noise; λ=0.05 not adopted.  Sampler's age-mixing (n_mixed_neg) preserves FiLM conditioning; pairwise family (O7a/O7c) closed. |
| **S1** | 08-12 | (A1 cfg) | 21 | lr_scheduler swap: OneCycle (max 1e-3, pct 0.3) → warmup 5% + cosine (peak base lr 3e-4) | 3e-4/16/100 | — | 0.6469 | +0.009 | ~ Noise-adjacent; plain AUROC 0.7036 vs baseline 0.7171.  Not adopted alone; could stack with O7c. |
| **D2-tabular** | 08-15 | ×15 seeds (LGBM) | 641 | GBDT/XGB/RF on the small spectral bank, I0006-holdout (911/192) | — | — | **0.655 ± 0.019** (LGBM n=15, min 0.626) / 0.653 ± 0.012 (XGB) / 0.576 ± 0.020 (RF) | **+0.109** vs small-pool CRNN 0.546 | ✅ **First method to clear the pass line with margin.** `spec` block alone 0.659 ± 0.007; `arch` (CAISR) 0.576; coh/tp/hrv/spo2 ≈ 0.52/0.51/0.40/0.46.  Small regime only — the large-pool (A1-comparable) run is the next gate. |

**B-wave verdict (08-09)**: no intervention adopted — the sub3 config remains the best cross-site baseline; the CAISR-OOD mean shift is untargeted until the ComBat result.

**C-wave verdict (08-09)**: the feature side is now exhausted (z-score / small pool / TTA / ComBat all ✗).  The drop is driven by the stage-prob shift + montage/hardware domain effect — outside the 21-dim feature space.  Surviving input-side option: **P3 raw spectral features** (delta power); surviving model-side options: multi-arch soft voting, proxy-based selection.

**D-wave verdict (08-13)**: no D/O7/S-wave intervention survived rerun — D1 (no-age) 0.6616→0.6392, **O7c λ=0.05 0.6664→0.6199** (Δ0.047 ≈ 2× noise floor); every proxy gain ±0.02–0.03 to date has been seed noise.  The proxy's run-to-run noise under the sub3 config is larger than any candidate's effect size.  Pairwise-loss family (O7a/O7c) closed; warmup-cosine +0.009 not adopted alone.  **sub4 = sub3 config unchanged** — re-submit to measure official-side noise and anchor the proxy→official discount.

### Ablation protocol

1. Start from the locked sub3 config.  2. Exactly **one** change.  3. 100 epochs; record val + age-cond AUROC.  4. Δ > 0 accept / Δ ≤ 0 reject, document why, move on.

---

## Key Dates & Deadlines

| Date | Event |
|------|-------|
| 2026-06-03 | Official phase begins |
| 2026-08-20 | **Official phase final submission deadline** |
| 2026-09-01 | 4-page preprint deadline |
| 2026-09-20–23 | **CinC 2026, Madrid** |
| 2026-10-10 | Final 4-page paper deadline |
