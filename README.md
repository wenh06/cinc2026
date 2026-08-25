# CinC2026

[![docker-ci-and-test](https://github.com/wenh06/cinc2026/actions/workflows/docker-test.yml/badge.svg?branch=docker-test)](https://github.com/wenh06/cinc2026/actions/workflows/docker-test.yml)
[![format-check](https://github.com/wenh06/cinc2026/actions/workflows/check-formatting.yml/badge.svg)](https://github.com/wenh06/cinc2026/actions/workflows/check-formatting.yml)

<p align="left">
  <img src="images/cinc2026-banner.svg" width="40%" />
</p>

Screening for Cognitive Impairment During Sleep Studies: The George B. Moody PhysioNet Challenge 2026

[Challenge Website](https://moody-challenge.physionet.org/2026/)

<!-- toc -->

- [The Conference](#the-conference)
- [Submission Status](#submission-status)
- [Description of the files/folders(modules)](#description-of-the-filesfoldersmodules)

<!-- tocstop -->

## The Conference

[Conference Website](https://cinc2026.org/) |
[Unofficial Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vSseLxrUufQX34SPCjF1rN_0Zaew6Jvvree2KYxZ17AF0BqPpEGAtvEPcdmCpLhn3j3neDFhRrhCXWE/pubhtml) |
[Official Phase Leaderboard](https://docs.google.com/spreadsheets/d/e/2PACX-1vQPPM17Qj6d1JCzn3fPhGE1CE0QI45z-KYkNVs5mainy7nQNEQV2FAfYNsvGq0y2P5aMVZ_Y7rNs070/pubhtml)

## Submission Status

The official phase is **closed** (final submission deadline 2026-08-20 23:59 GMT).  Nine of the ten
entries were evaluated successfully; the best age-conditioned AUROC on the official validation set is
**0.627**, achieved by the sub5 tabular XGBoost entry (**#2693**) on the 390-dim spectral block.  Our
final entry, the fusion ranker (**#2877** — PCA-64 Philosopher's Stone latent concatenated with the
390-dim spectral block, single XGBoost, with the sub5 tabular XGB as the per-record fallback), scored
0.600; the frozen-embedding variants all scored below the pure tabular baseline (0.583 / 0.557 / 0.600
vs 0.627).  One entry (#2750) failed to evaluate and did not count toward the ten.

Next milestones: choose the test-set algorithm (recommended: **#2693**) and submit the challenge 4-page
preprint by **2026-08-27**; the final 4-page paper is due in early October.  The full submission
history — configs, scores and commit hashes — lives in [submissions](submissions).

## Description of the files/folders(modules)

### Files

<details>
<summary>Click to view the details</summary>

- [README.md](README.md): this file, serves as the documentation of the project.
- [cfg.py](cfg.py): the configuration file for the whole project, including the D2 tabular entry
  (`TrainCfg.tabular`) and the model-component registry (`TrainCfg.components`; the default is the
  `phi_pca64` Phi ranker with `sub5_tabular_xgb` as the per-record fallback — see `TrainCfg.phi`).
  `TrainCfg.phi.features` selects the ranker input: `"fusion"` (default — PCA-64 latent concatenated
  with the 390-dim spectral block, single XGBoost) or `"latent"` (PCA-64 only, `model` =
  xgboost/logistic/ensemble).
- [const.py](const.py): constant definitions.
- [Dockerfile](Dockerfile): docker file for building the docker image for submissions.  The image bakes the
  vendored Philosopher's Stone source (`third_party/philosophers-stone/src`) and the D1 spectral feature cache
  (`data/spectral_features`) directly from the repository — the build needs no `.git`, no MEGA and no
  HuggingFace access.  The 2.4 GB Phi checkpoint is downloaded at build time (`PHI_MODEL_DOWNLOAD=1`)
  from huggingface.co → hf-mirror.com → the baked MEGA link (`PHI_MEGA_URL`), size + SHA-256 verified.
- [requirements.txt](requirements.txt), [requirements-docker.txt](requirements-docker.txt), [requirements-no-torch.txt](requirements-no-torch.txt):
  requirements files for different purposes.
- [create_labels.py](create_labels.py), [evaluate_model.py](evaluate_model.py), [helper_code.py](helper_code.py),
  [run_model.py](run_model.py), [train_model.py](train_model.py): scripts inherited from the
  [official baseline](https://github.com/physionetchallenges/python-example-2026.git).
  Modifications on these files are invalid and are immediately overwritten after being pulled by the organizers (or the submission system).
- [sync_official.py](sync_official.py): script for synchronizing data from the official baseline and official scoring code.
- [team_code.py](team_code.py): entry file for the submissions — `train_model` / `load_model` / `run_model`, including the
  5-fold CV ensemble mode (`TrainCfg.folds = [0..4]`: one model per fold, equal-weight probability averaging at inference),
  and the routing to the D2 tabular path when `TrainCfg.tabular.enable` is set.  With `TrainCfg.components` configured,
  each component trains into `model_folder/components/<name>/`, a manifest (`model_manifest.json`) drives loading, and
  `run_model` walks the components in priority order with per-record fallback (e.g. Phi → sub5 tabular XGB).
- [component_registry.py](component_registry.py): model-component registry — manifest format, per-type artifact names,
  component enumeration (priority-ordered) and the baked spectral-feature-cache resolution
  (`data/spectral_features`, vendored in the repository, no network at runtime).
- [phi_component.py](phi_component.py): the Phi primary component — frozen Philosopher's Stone 1024-d
  latents (cache-first from the vendored `data/phi_cache`, on-the-fly extraction for misses) → PCA-64,
  optionally fused with the 390-dim spectral block (`TrainCfg.phi.features="fusion"`), ranked by
  XGBoost / logistic regression / their probability-averaged ensemble.  Records without a usable C4-M1
  fall through to the next component at run time.
- [tabular_pipeline.py](tabular_pipeline.py): the D2 tabular submission path — 641-dim spectral/physiological feature bank
  (cache-first, on-the-fly extraction from raw PSG + CAISR on misses) + XGBoost/LightGBM.  The fitted feature-column list is
  serialised with the model so training and inference always see identical columns.
- [trainer.py](trainer.py): training loop (`CINC2026Trainer`) with AUROC monitoring, per-site evaluation, and early stopping.
- [dataset.py](dataset.py): dataset classes (`CINC2026Dataset`, `FastDataReader`) and CAISR epoch-feature construction
  (`build_epoch_features`).  Supports 5-fold CV via `train_config.fold`; train/val splits use multi-factor stratification
  (label × site × sex × age band) via torch_ecg's `stratified_train_test_split`.
- [data_reader.py](data_reader.py): database reader (`CINC2026`) for loading PSG recordings, CAISR and human-expert annotations.
- [outputs.py](outputs.py): model output dataclass (`CINC2026Outputs`) handling logits → probability → binary prediction conversion.
- [test_docker.py](test_docker.py): tests for the Docker submission pipeline (dataset, models, trainer, entry),
  including the tabular path and the component registry / per-record fallback chain.
- [post_docker_build.py](post_docker_build.py): post-Docker-build setup script — environment sanity check,
  Philosopher's Stone checkpoint download (default on; huggingface.co → hf-mirror.com → optional MEGA via
  megadl, every attempt size + SHA-256 pinned), and verification of the vendored Phi source and the D1
  spectral feature cache (`data/spectral_features`).
- [channel_table.csv](channel_table.csv): channel name standardization mapping across recording sites.
- [submissions](submissions): log file for the submissions, including the key hyperparameters, the scores received,
  commit hash, etc. The log file is updated after each submission.

</details>

### Folders(Modules)

<details>
<summary>Click to view the details</summary>

- [data](data): the only exception to the "no data in git" rule.  Two precomputed caches are tracked and must
  ship with the build context (the official image build has no `.git` and no guaranteed MEGA/HF access):
  `data/spectral_features` (features.csv, record_meta.csv, manifest.json; ~12 MB) — the D1 641-dim spectral
  cache used cache-first by the tabular path; and `data/phi_cache` (1,090 npz, ~8.6 MB) — the frozen
  Philosopher's Stone 1024-d latents used cache-first by the Phi component.  All other `data/` contents
  (local dataset staging) stay ignored.
- [official_baseline](official_baseline): the official baseline code, included as a submodule.
- [model_configs](model_configs): modular per-model configurations (separate config files for `EpochTransformer`, `EpochCRNN`).
- [models](models): model definitions (`EpochTransformer`, `EpochCRNN` with multiple CNN backbone variants, `ChannelTransformer`, `MultiBranchNet`).
- [utils](utils): utility scripts, including [custom scoring metrics](utils/scoring_metrics.py), [hyperparameter search](utils/run_search.py),
  [log analysis](utils/analyze_logs.py), [feature shift analysis](utils/analyze_feature_shift.py),
  the [Phi latent cache loader + on-the-fly fallback](utils/phi_cache.py), the [hybrid low-memory wavelet stage](utils/phi_preprocess.py),
  the [fixed train/val split](utils/cinc2026-data-split.json) (alias of the 5-fold split's fold_0),
  the [5-fold split generator](utils/make_5fold_split.py) + [5-fold split](utils/cinc2026-5fold-split.json),
  and [out-of-fold evaluation](utils/evaluate_oof.py) for the ensemble.
- [results](results): experiment log files and analysis notes.
- [scripts](scripts): local-validation scripts (not part of the submission pipeline) — [spectral feature extraction](scripts/extract_spectral_features.py),
  [Phi latent cache extraction](scripts/phi_cache_extract.py), [D2 tabular baselines](scripts/d2_tabular_baseline.py) /
  [robustness](scripts/d2_tabular_robustness.py) / [large-pool gate](scripts/d2_tabular_large.py), and the
  [Phi PCA-rankers](scripts/phi_pca_ranker.py) (per-site `--holdout-site`).
- [third_party](third_party): vendored third-party code — the [Philosopher's Stone](third_party/philosophers-stone)
  brain-health model, pinned to commit `0b1b49a8` (CC BY-NC 4.0); only `src/` and `LICENSE` are tracked.  It is
  no longer a git submodule because the official build context has no `.git`.
- [.github/workflows](.github/workflows): CI pipelines — the [docker test](.github/workflows/docker-test.yml) builds
  from a `git archive` context without `.git` (mimicking the official harness) and runs the full pipeline suite
  with `--network none`.

</details>

:point_right: [Back to TOC](#cinc2026)

### Miscellaneous

[CinC2020](https://github.com/DeepPSP/cinc2020) | [CinC2021](https://github.com/DeepPSP/cinc2021) | [CinC2022](https://github.com/DeepPSP/cinc2022) | [CinC2023](https://github.com/wenh06/cinc2023) | [CinC2024](https://github.com/wenh06/cinc2024) | [CinC2025](https://github.com/wenh06/cinc2025)
