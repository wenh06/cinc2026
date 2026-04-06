# CinC 2026 — Submission Analysis & Improvement Plan

_Updated: 2026-04-06_

---

## Submission Results So Far

| # | ID | Model | Commit | AUROC | AUPRC | Acc | F |
|---|---|---|---|---|---|---|---|
| 1 | 1173 | EpochTransformer_M | 5a77988 | 0.522 | 0.059 | 0.201 | 0.065 |
| 2 | TBD | EpochCRNN resnetNC_BNse_M | 765a4b7 | ~0.49x | ? | ? | ? |

---

## Root Cause Analysis

### 1. Massive local-to-leaderboard gap

| Metric | Local val | Leaderboard |
|---|---|---|
| AUROC | 0.6955 | 0.522 |
| AUPRC | 0.710 | 0.059 |

This ≈ 0.17 AUROC gap (and 0.65 AUPRC gap) is **not noise** — it points to structural problems.

**Why the gap exists:**

**a) Positive-class prevalence shift**
The training set has 50% positive rate (392/780). AUPRC=0.059 on the hidden validation set
implies the positive rate there is ~6%.  Random AUPRC ≈ prevalence, so a model performing
at random on a 6%-positive dataset would get exactly this score. The challenge team likely
uses a cohort with realistic cognitive-impairment prevalence (~5–10%), while the training set
was constructed with deliberate 50/50 balancing. The model calibrated on 50% prevalence will
output probabilities in the wrong range for 6% prevalence.

**b) Site-specific overfitting**
Training set site distribution: **S0001 73%, I0006 20%, I0002 7%**.
A random 80/20 split gives a val set also dominated by S0001. If the model learns S0001-
specific patterns (equipment artifacts, demographic biases) it can reach local AUROC≈0.70
while completely failing on other sites or temporal held-out cohorts.

**c) Local val set too small (156 samples)**
AUROC from 156 balanced samples has ±0.05–0.08 standard error — large enough to be
misleading. A model achieving 0.695 locally could realistically be 0.55–0.75 on the true test.

**d) Possible binary-threshold miscalibration**
`binary_output = (ci_prob_pos >= 0.5).long()`. With a 6%-positive test set, the optimal
threshold is ~0.06, not 0.5. Using 0.5 causes the model to predict almost all negative
→ Accuracy≈1-0.06=0.94, but that contradicts Accuracy=0.201. The Accuracy=0.201 with
F=0.065 suggests the model is predicting positive for a large fraction of cases but is
mostly wrong → the model is actually overconfident in the positive direction, OR the
hidden validation set labels are structured differently (e.g. multi-session per patient).

### 2. EpochCRNN trained on half the data

The CRNN log shows **20 gradient steps/epoch** vs 40 for the Transformer.
20 × batch_size(16) = 320 training samples vs the expected ~620 (80% of 780 CAISR records).
The CRNN was effectively trained on half the dataset — this alone reduces val AUROC by
~0.02–0.05 and makes generalization worse. **Verify the training command** (see §Execution).

### 3. Slow CRNN convergence with OneCycleLR

Training loss for CRNN was still ~0.69 (≈ uninformative log-loss) at epoch 9. The OneCycleLR
default warmup (30% of total steps) is too long for a CRNN that already has inductive bias.
The Transformer needs more warmup because its attention weights start random; the CRNN's
convolutions learn faster.

---

## Improvement Plan

### Priority 0: Immediate fixes (before next submission)

| Fix | File | Action |
|---|---|---|
| Verify CRNN training command | — | Confirm `db_dir = /Data1/.../physionetchallenge2026data` (parent, not `training_set/`), print n_train to log |
| Print training set size at epoch 0 | trainer.py | Log `len(train_dataset)` at setup |
| Add pos_weight for leaderboard prevalence | cfg.py | Try `pos_weight=8.0` (≈ 0.94/0.06) to calibrate toward low-prevalence test |
| Lower binary threshold | team_code.py | Use threshold≈0.2 (geometric mean of 0.5 and prevalence≈0.06) |

### Priority 1: Better validation strategy

The single 80/20 random split is misleading. Replace with **site-stratified split**:

```python
# In dataset.py __init__ / trainer _setup_dataloaders:
from sklearn.model_selection import StratifiedGroupKFold
# groups = SiteID, stratify = label
```

This ensures each site appears proportionally in both train and val, preventing the model
from exploiting site-specific biases to inflate local val AUROC.

Even better: **5-fold stratified CV** locally, submit the best-checkpoint ensemble.

### Priority 2: Regularization

The model has strong local val performance (0.695) but fails on the hidden test — classic
overfitting signature with 780 training samples.

| Technique | Config change |
|---|---|
| Label smoothing (ε=0.1) | `criterion_kw = {"label_smoothing": 0.1}` in model config |
| Higher weight decay | `TrainCfg.decay = 5e-2` (up from 1e-2) |
| Dropout 0.2→0.3 | model config `dropout=0.3` |
| Dropout on CAISR feat at input | Add `nn.Dropout(0.1)` before first layer in dataset/model |
| CAISR feature noise | Add Gaussian noise (σ=0.05) to CAISR features during training |
| Epoch masking | Randomly zero 10% of 30-s epochs in the input sequence |

### Priority 3: Model ranking (arch screening)

Still need to run `epoch_crnn_M` (simplest CRNN) and `epoch_crnn_tresnetE_M` locally:

| Preset | Status | Expected advantage |
|---|---|---|
| `epoch_transformer_M` | ✅ done (best AUROC=0.695) | attention over long sequence |
| `epoch_crnn_resnetNC_BNse_M` | ✅ done (0.667) | deeper CNN features |
| `epoch_crnn_M` | ⬜ pending | simplest, least overfitting |
| `epoch_crnn_tresnetE_M` | ⬜ pending | mixed-block depth |

Run them with the **fixed training command and site-stratified val split** so the
comparison is apples-to-apples.

### Priority 4: Learning rate tuning for CRNN

CRNN does not need a long OneCycleLR warmup. Try:

```python
# For CRNN runs specifically:
TrainCfg.pct_start = 0.1   # warmup only 10% of total steps (default is 0.3)
TrainCfg.max_lr = 3e-3      # CRNN tolerates higher peak LR than Transformer
```

### Priority 5: Ensemble

After identifying top-2 architectures, ensemble their probability outputs:

```python
final_prob = 0.6 * prob_transformer + 0.4 * prob_crnn
```

Optimise weights on site-stratified val set.

---

## Execution: How to Run the Search

### Step 1 — Fix training command

Always pass the **parent directory** to ensure both layout branches are tested:

```bash
python train_model.py /Data1/wenh06/physionetchallenge2026data saved_models/run -v
```

Watch the first log line to confirm n_train ≈ 611 (80% of 766 CAISR-available records).

### Step 2 — Run arch screening (local, overnight)

Each run ≈ 30 min × 4 architectures = 2 h.  Run sequentially:

```bash
# In cfg.py: change TrainCfg.model_name, then:
for model in epoch_crnn_M epoch_crnn_tresnetE_M; do
    sed -i "s/^TrainCfg.model_name = .*/TrainCfg.model_name = \"${model}\"/" cfg.py
    python train_model.py /Data1/wenh06/physionetchallenge2026data saved_models/run_$model -v
done
```

Compare with:
```bash
python utils/analyze_logs.py --log-dir saved_models/run/working_dir/log --no-plot
```

### Step 3 — HP search (best arch, local)

Fix architecture to the Phase 1 winner. Run the following 9-point grid (each ~30 min):

```
max_lr  ∈ {5e-4, 1e-3, 2e-3}
decay   ∈ {5e-3, 1e-2, 5e-2}
```

One-at-a-time variation from the best single-run:
```bash
# Example: vary max_lr
for lr in 5e-4 2e-3; do
    # Edit cfg.py max_lr, then train
    python train_model.py /Data1/... saved_models/hp_lr_$lr -v
done
```

Total: ~9 runs × 30 min = 4.5 h.

### Step 4 — Submission cadence

5 submissions per phase. Spend them strategically:
- Sub 2: `epoch_crnn_resnetNC_BNse_M` ✅ already done
- Sub 3: `epoch_crnn_M` (simpler, less overfitting) — or best HP sweep result
- Sub 4: Best model + label smoothing + pos_weight calibration
- Sub 5: Ensemble of top-2 architectures

---

## Key Diagnostic Commands

```bash
# Quick best-checkpoint table
python utils/analyze_logs.py --log-dir saved_models/run/working_dir/log --no-plot

# Full plots
python utils/analyze_logs.py --log-dir saved_models/run/working_dir/log --out-dir results/plots

# Check training set size
python -c "
from data_reader import CINC2026
dr = CINC2026('/Data1/wenh06/physionetchallenge2026data')
df = dr._df_records[dr._df_records['partition']=='training_set']
import os; df['has_caisr'] = df['algo_ann_path'].apply(os.path.exists)
print(f'Total: {len(df)}, has CAISR: {df.has_caisr.sum()}')
print(df['SiteID'].value_counts())
"
```

---

## Data Facts (for reference)

| Fact | Value |
|---|---|
| Training records | 780 |
| Records with CAISR | 766 (14 missing: 8 I0006, 6 S0001) |
| CI positive rate (training) | 50.3% |
| Estimated CI rate (hidden test) | ~6% (inferred from AUPRC=0.059) |
| Site distribution | S0001: 572 (73%), I0006: 154 (20%), I0002: 54 (7%) |
| Local val size (80/20 split) | ~156 samples |
| AUROC std error at n=156, balanced | ±0.05–0.08 |
