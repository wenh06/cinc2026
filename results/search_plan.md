# CinC 2026 — Model & Hyperparameter Search Plan

## Context

- **Task**: Binary classification (future cognitive impairment) from overnight PSG
- **Input**: sequence of 30-s-epoch CAISR feature vectors (21-dim), T ≈ 720–1100 epochs/night
- **Training set**: 780 labelled records; 80/20 train/val split
- **Primary metric**: validation AUROC (averaged over 3 sites: I0002, I0006, S0001)
- **Compute**: local RTX 5060 Ti 16 GB; challenge GPU: A30 24 GB / RTX 6000 Ada 48 GB
- **Time budget per run**: ~30 min for 100 epochs (EpochTransformer_M on 624 training records)

---

## Available Model Presets

### EpochTransformer

| Preset | d_model | heads | layers | ff | ~params |
|---|---|---|---|---|---|
| `epoch_transformer_S` | 64 | 2 | 2 | 256 | ~110 K |
| `epoch_transformer_M` | 128 | 4 | 4 | 512 | ~825 K |
| `epoch_transformer_L` | 256 | 8 | 6 | 1024 | ~5.3 M |

### EpochCRNN — resnetN backbone (3-stage basic blocks)

| Preset | CNN channels | LSTM hidden | ~params |
|---|---|---|---|
| `epoch_crnn_S` | 16→32→64 | 64 | ~101 K |
| `epoch_crnn_M` | 32→64→128 | 128 | ~437 K |
| `epoch_crnn_L` | 64→128→256 | 256 | ~1.6 M |

### EpochCRNN — resnetNC_BNse backbone (4-stage bottleneck+SE, Nature-Comm style)

| Preset | CNN num_filters | CNN out ch | LSTM hidden | ~params |
|---|---|---|---|---|
| `epoch_crnn_resnetNC_BNse_S` | 16→32→48→64 | 256 | 64 | ~384 K |
| `epoch_crnn_resnetNC_BNse_M` | 32→64→96→128 | 512 | 128 | ~1.5 M |
| `epoch_crnn_resnetNC_BNse_L` | 48→96→192→256 | 1024 | 256 | ~5.5 M |

### EpochCRNN — tresnetE backbone (4-stage mixed basic_se+bottleneck_se, TResNet-style)

| Preset | CNN num_filters | CNN out ch | LSTM hidden | ~params |
|---|---|---|---|---|
| `epoch_crnn_tresnetE_S` | 16→32→64→128 | 512 | 64 | ~669 K |
| `epoch_crnn_tresnetE_M` | 32→64→128→256 | 1024 | 128 | ~2.6 M |
| `epoch_crnn_tresnetE_L` | 48→96→192→384 | 1536 | 256 | ~6.9 M |

---

## Search Strategy

### Phase 1: Architecture Screening (M-size, fixed HPs)

**Goal**: Quickly rank the 4 backbone families.
**Fixed HP**: `lr=3e-4`, `max_lr=1e-3`, `decay=1e-2`, `n_epochs=100`, `batch_size=16`, `optimizer=adamw_amsgrad`, `lr_scheduler=one_cycle`.
**Runs** (change `TrainCfg.model_name` in `cfg.py`):

| # | model_name | Expected params | Priority |
|---|---|---|---|
| 1 | `epoch_transformer_M` | 825 K | ✅ baseline (already run) |
| 2 | `epoch_crnn_M` | 437 K | highest — smallest, fast |
| 3 | `epoch_crnn_resnetNC_BNse_M` | 1.5 M | SE attention effect |
| 4 | `epoch_crnn_tresnetE_M` | 2.6 M | mixed-block effect |

**Decision**: take the top-1 or top-2 families (by best val AUROC) into Phase 2.
**Analysis**: `python utils/analyze_logs.py --log-dir saved_models/run/working_dir/log`

---

### Phase 2: Size Scaling (best family)

**Goal**: Find the sweet spot between capacity and overfitting on ~624 training records.

Run S / M / L for the winner(s) from Phase 1.
Watch for:
- val AUROC plateau: if M ≈ L → M is sufficient
- strong overfitting (train loss << val loss at same epoch): reduce size or add regularisation

Expected additional runs: 2–4.

---

### Phase 3: Hyperparameter Search (best arch + size)

**Dimensions to sweep** (one-at-a-time or small grid):

| HP | Values to try | Notes |
|---|---|---|
| `max_lr` (OneCycle peak) | 5e-4, **1e-3**, 2e-3 | current=1e-3 |
| `decay` (AdamW weight decay) | 1e-3, **1e-2**, 5e-2 | current=1e-2 |
| `dropout` (model) | 0.0, **0.1**, 0.2 | current=0.1 |
| `early_stopping.patience` | 10, **20**, 30 | |
| `batch_size` | 8, **16**, 32 | smaller = noisier gradients, often regularises |

Recommended grid (8 runs, 2-level factorial on the 3 most impactful HPs):

```
max_lr ∈ {5e-4, 2e-3}  ×  decay ∈ {1e-3, 5e-2}  ×  dropout ∈ {0.0, 0.2}
```

---

### Phase 4: Ensemble / Multi-model

If time allows, train the top-2 architectures with their best HPs and ensemble predictions:
```
final_prob = 0.5 * prob_model_A + 0.5 * prob_model_B
```
Or use a calibrated weighted average (optimise weights on val set).

---

## Key Tracking Commands

```bash
# Run a training experiment (edit TrainCfg.model_name in cfg.py first)
python train_model.py /Data1/wenh06/physionetchallenge2026data/training_set \
    saved_models/run -v

# Analyse all logs and generate plots
python utils/analyze_logs.py \
    --log-dir saved_models/run/working_dir/log \
    --out-dir images/analysis

# Quick summary table only
python utils/analyze_logs.py \
    --log-dir saved_models/run/working_dir/log --no-plot
```

## Switching Models

Edit **one line** in `cfg.py`:
```python
TrainCfg.model_name = "epoch_crnn_resnetNC_BNse_M"   # example
```
Everything else (trainer, team_code, checkpoint saving) is config-driven.

## Notes

- **Site imbalance**: I0006 (in-lab) consistently reaches AUROC ~0.80+; S0001 (home-sleep)
  ~0.55. If no architectural change closes this gap, consider per-site normalisation or
  a site-conditioning approach (site as an additional embedding).
- **Label imbalance**: unknown, but AUPRC > AUROC throughout → positive class is a minority.
  Consider `pos_weight` in `BCEWithLogitsLoss` if imbalance is severe.
- **Data augmentation**: not yet implemented. Possible: temporal jitter (shift crop window),
  epoch dropout (randomly mask a fraction of CAISR features), feature noise.
- **CAISR feature extension** (Phase 6.3a in ROADMAP): adding spectral features would
  increase the 21-dim CAISR vector to ~34-dim, possibly lifting all models uniformly.
