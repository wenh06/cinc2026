# Post-Challenge Study: What the Top Teams Did, and What We Did Differently

This document studies the 20 highest-ranked teams of the
[George B. Moody PhysioNet Challenge 2026](https://moody-challenge.physionet.org/2026/)
**as listed in the preliminary results table of 25 September 2026**, and compares their designs
with the experiments recorded in [ROADMAP.md](ROADMAP.md).  The corrected table of 1 October 2026
promoted one team (OUS_IVS) into the ranking at 16th place, which shifted every entry below 15th
down by one — including us, 19th → 20th of 42; the affected numbers are updated below, and the
newcomer is covered in section 3.20.
It was written after [Computing in Cardiology 2026](https://cinc2026.org/)
(Madrid, 20-23 September 2026) to inform a possible future attempt, and it is the reference
for the post-conference discussion in [README.md](README.md).

**Sources.** The official test-set results in [_results/challenge_2026_results.tsv_](results/challenge_2026_results.tsv)
(mirror of <https://moody-challenge.physionet.org/2026/results/challenge_2026_results.tsv>;
the post-correction version of 1 October 2026), and the
4-page conference papers linked from the
[preliminary program](https://cinc.org/prelim_program_2026/) at
`https://cinc.org/2026/Program/accepted/<CinC Submission ID>_Preprint.pdf`.
Local copies of the papers are kept outside the repository (they are third-party material);
the tables below quote only published numbers.

## 1. Overview

| Rank | Team | val | test | Δ (test − val) | Approach in one line | Training set |
|---:|---|---:|---:|---:|---|---|
| 1 | NeuroUC3M | 0.755 | **0.749** | −0.006 | 1031 descriptors × 5 views × 3 heads × 15 seeds (225 models) + 6 self-supervised EEG encoders, rank-averaged | large (6,582) |
| 2 | Matcha | 0.737 | 0.727 | −0.010 | Multimodal 30-s epoch representation (human + CAISR + self-supervision) → local and full-night Transformers | large |
| 3 | CLECLINIC | 0.695 | 0.708 | +0.013 | 105 per-epoch features + 33-d whole-night vector → InceptionTime with a rank loss | large (6,600) |
| 4 | CAU_KU | 0.746 | 0.705 | −0.041 | EEG only: time-domain and time-frequency branches, per-subject calibration, probability averaging | large |
| 5 | NeuroAI | 0.708 | 0.704 | −0.004 | "Feature-stable" multi-scale descriptors; features kept only if their age-conditioned association agrees in direction across all training sites | large |
| 6 | Better Call Sandman | 0.636 | 0.703 | +0.067 | Engineered features + XGBoost vs. transfer-learned CNN; XGBoost was selected | large (+ extra HSP data) |
| 6 | NLICA | 0.736 | 0.703 | −0.033 | Age-residualized 272-d feature vector + 30-member tree ensemble, fused with a 1-D CNN over N2/N3/REM | large |
| 8 | Koalalition | 0.669 | 0.683 | +0.014 | 136 sleep-stage-stratified physiological features + 23 age-residual features, bagged XGBoost | large |
| 9 | REMedy | 0.641 | 0.681 | +0.040 | Attention/GradCAM-driven feature selection + ECG-derived biological age, seven modality-combination XGBoost models with dynamic routing | large |
| 10 | bashlab | 0.528 | 0.678 | +0.150 | 151 named features (including 2 recording-date features) + CORAL alignment + TabFM/tree rank blend | large |
| 11 | Momochi-SleepAI | 0.601 | 0.673 | +0.072 | 10 CAISR summaries + 3 EEG spectral descriptors from 16 fixed windows → class-balanced logistic regression | large |
| 12 | Biosignal Pilots | 0.636 | 0.671 | +0.035 | Sleep depth (SWA, ORP, delta entropy) + fragmentation (CAP, arousal timing, respiratory events) → XGBoost + SHAP | large |
| 12 | MeDSP | 0.708 | 0.671 | −0.037 | Multi-domain features (architecture, EEG spectra, SpO₂, respiration, limb movements, hypnogram complexity), top-20 → 300-tree Random Forest | large |
| 14 | BAPORLab | 0.706 | 0.670 | −0.036 | Stage-aware hierarchical ensemble: 5-s staging CNN + 128 selected epochs per stage + per-modality Set Transformer experts | large (+ external HSP pretraining) |
| 15 | OCA-CENTINEL | 0.658 | 0.666 | +0.008 | Frozen CBraMod EEG embeddings + YASA/EMG features, mutual-information removal of site-dominated dimensions, LightGBM with age-matched pairwise ranking | large |
| 16 | OUS_IVS | 0.708 | 0.664 | −0.044 | Frozen multimodal SleepFM embeddings (512-d) + clinically derived sleep/respiratory/oxygenation/age-relative features for ranking, with a separate restricted ExtraTrees head for the binary decision | large |
| 17 | PhysioWinn | 0.748 | 0.658 | −0.090 | Lightweight CNN (79,457 parameters) on six EEG channels; F3/F4 and N3 windows dominate | large |
| 18 | Med_YNNU | 0.646 | 0.636 | −0.010 | DFHANet: dense multi-scale time- and frequency-domain CNNs over five modalities with hierarchical attention | large |
| 19 | UPV Maths | 0.681 | 0.635 | −0.046 | Frozen SleepFM backbone with a lightweight head, studying input combinations and temporal pooling | large |
| 20 | **Revenger (us)** | 0.627 | **0.632** | **+0.005** | CAISR CRNN / XGBoost on a 390-d spectral block / frozen brain-health embeddings, selected with a leave-one-site proxy | small |
| 21 | FuneLab | 0.729 | 0.630 | −0.099 | Multi-rate channel-wise 1-D CNN over raw EEG/EOG/EMG/ECG/respiration/SpO₂ + demographics and annotations | small |

`val` and `test` are the official `Age-conditioned AUROC` values from the corrected table
(42 ranked entries); Δ is our own arithmetic.  Ranks 16–21 are the post-correction ones: OUS_IVS
was unranked in the preliminary table and entered at 16th on 1 October 2026.  The team sections
below keep the preliminary order, with OUS_IVS appended as section 3.20.

## 2. Where we stood

Our own experiments are recorded in [ROADMAP.md](ROADMAP.md) with experiment identifiers
(O/B/C/D/S waves, D2-tabular, P1-P5):

| Our experiment | What it was | Result |
|---|---|---|
| `EpochCRNN_M`, 21-d CAISR (O0/O0repro) | Epoch-sequence CRNN on CAISR features, large set | same-site age-conditioned AUROC 0.758-0.762; leave-one-site (A1) I0006 **0.638** / S0001 0.594 / I0002 0.703 |
| D2-tabular | 641-d spectral bank (staged relative powers, ratios, kurtosis, Hjorth, I-CARE quantiles, coherence, HRV, SpO₂, CAISR architecture, time metadata) + GBDT, small set | LightGBM **0.655 ± 0.019**, XGBoost 0.653, Random Forest 0.576; spectral block alone 0.659; architecture block 0.576; coherence 0.52 / temporal quantiles 0.51 / HRV 0.40 / SpO₂ 0.46 |
| P1 / O2 / D2 | 15-d whole-night aggregate branch (TST, SE, N3%, REM%, WASO, AHI, PLMI, REM/N3 latency, transitions/h, first-half N3%, cycle count/length, N3 decay slope), late-fused into the CRNN | same-site −0.024, proxy +0.002 → closed as redundant |
| P4 | Frozen Philosopher's Stone 1024-d latent → PCA-64, ranked by XGBoost/LR | validation 0.557-0.583; fusion with the spectral block 0.600 |
| Age interventions | FiLM age conditioning, O1 age-adversarial head (−0.027), O4/D1 zeroing the age channel (≈0), O7c age-stratified sampler + pairwise loss (refuted on rerun) | none adopted |
| Cross-site interventions | B2 per-record z-score (flat on I0006, +0.048 on I0002), C3 test-time adaptation (refuted), C4 ComBat harmonization (refuted), D3 deleting the most-shifted columns (−0.050), D4 domain randomization (+0.016) | none adopted |
| B1 | Small (1,103) vs. large (6,600) training pool under the same configuration | **−0.092** on the I0006 holdout for the small pool |

Our selected entry was the 390-d spectral XGBoost trained on the small set (official
validation 0.627, test 0.632).

## 3. Team-by-team comparison

Each entry lists what the team did, what we did that is comparable, where the designs differ,
and a hypothesis for the score difference.

### 3.1. NeuroUC3M — SIREN (test 0.749)

**They did**: 1031 hand-built descriptors split into five feature views, each fitted by logistic
regression, histogram gradient boosting and a LambdaRank head over 15 seeds (225 models); a second
branch reads four EEG derivations at 64 Hz with six self-supervised convolutional encoders
(pretrained on augmented copies of this year's data) whose frozen embeddings feed a 258-parameter
attention head; the branches are combined by **rank averaging** (0.6/0.4). Site invariance is
enforced by construction (within-recording percentiles, within-site rank normalization,
contrastive augmentations that imitate site discrepancies) and quantified with a site probe that
drops from 0.97 AUC on raw descriptors to 0.33 after normalization.

**We did**: engineered features plus trees (the 641-d bank with LightGBM/XGBoost, D2-tabular) and
frozen external embeddings (P4, Philosopher's Stone).

**Differences**: (i) they ensemble **heterogeneous** heads by rank; our roadmap lists
"multi-arch soft voting" as a surviving option in the C-wave verdict, but we never implemented
it — we submitted a single family; (ii) their self-supervision is trained **on this year's data**,
ours is a frozen external checkpoint; (iii) all of our site-invariance attempts were **post hoc**
(B2 z-scoring, C4 ComBat, D3 dropping shifted columns), theirs act at feature definition.

**Why they may score higher**: rank averaging across genuinely different views reduces variance,
and preventing site information from entering the features attacks the I0004/I0007 covariate
shift directly — our C1 probe showed the shift (arousal ×5-6, limb ÷100) and C4 showed that
repairing the distribution afterwards does not recover the score.

**Worth copying**: rank averaging instead of probability averaging; a site probe as a design
criterion.

### 3.2. Matcha (test 0.727)

**They did**: modality-specific convolutional stems over 20 canonical channels produce a 192-d
30-s epoch embedding; supervision combines **human** and CAISR staging/arousal/respiratory/limb
labels (weights 1.0/0.5/0.4/0.25) with cross-view self-supervision; local and full-night
Transformers aggregate to the patient level, plus a low-capacity record-wise residual that carries
whole-night summaries and outcome-observation timing.

**We did**: epoch-sequence models (EpochCRNN/Transformer) and three representation families, all
built from CAISR features, the 390/641-d spectral block, or frozen embeddings.

**Differences**: we never used raw waveforms or human annotations — our sequence model consumed
the 21-d CAISR vector alone; they train with self-supervision plus multi-task supervision and use
the large set.

**Why they may score higher**: the quality of the per-epoch representation bounds what a sequence
model can learn, and the feature family we fed it (CAISR event rates) is precisely the one our own
probes identified as most corrupted across sites. Their outcome-timing residual additionally
consumes date-like information, which the organizers later removed — consistent with their
validation score dropping from 0.847 in the paper to 0.737 in the official table.

### 3.3. CLECLINIC (test 0.708)

**They did**: 105 features per 30-s epoch — 72 EEG features (six bipolar derivations × 12:
relative power in six bands, slowing ratio, alpha/theta ratio, Hjorth mobility and complexity,
kurtosis, line length), plus EMG, SpO₂, spindle/slow-oscillation microstructure, CAISR events and
HRV, a 5-d stage encoding, 7 validity indicators and 3 time-of-night channels — and a separate
33-d whole-night static vector that includes a **brain-age gap** (bias-corrected residual of an
out-of-fold ridge regression fitted on cognitively normal subjects). Cross-family attention feeds
InceptionTime blocks whose kernels span 9/19/39 epochs (4.5/9.5/19.5 minutes, matching sleep-stage
cycling), pooled by stage; the loss includes a **rank term (weight 0.43)**; three networks are
averaged. Subject-level median/IQR normalization and random modality blanking during training
provide cross-site robustness. Final training used all 6,600 recordings.

**We did**: the 641-d spectral bank with LightGBM (D2-tabular, proxy 0.655) and the 15-d
whole-night branch (P1/O2/D2) — the same feature families, and the counterpart of their static
vector.

**Differences**: (i) **granularity**: they keep the per-epoch time axis (~800-1024 epochs) while we
aggregate the same features into 390/641-d vectors and lose it; (ii) data: they train on 6,600
while our 641-d bank exists only for the 1,103-record small set (the large set needs 1.2 TB of raw
data), and our own B1 experiment shows the small pool costs 0.092 on the I0006 holdout; (iii) model:
InceptionTime with stage pooling and a 3-network average vs. our single GBDT; (iv) objective: their
rank loss vs. our pairwise attempts (O7a/O7c) that were swamped by noise in a 21-d feature space;
(v) their static branch carries a brain-age gap, whereas our age work only removed age
(O4/D1) rather than distilling it.

**Why they may score higher**: they combined the two halves we tested **separately** — rich
features and the time axis. The kernel scales match sleep-cycle dynamics, so per-epoch EEG
microstructure (spindles, slow waves, CAP-like dynamics) can be expressed, and a rank loss only
pays off when the input carries enough information.

**Worth copying**: per-epoch features into a sequence model, a static whole-night vector, a rank
loss, multi-network averaging, subject-level median/IQR normalization and modality dropout.

### 3.4. CAU_KU (test 0.705)

**They did**: EEG only, with the six derivations harmonized across sites; time-domain and
time-frequency branches are trained separately, calibrated per subject, and combined by
probability averaging; nothing is pretrained; the large set is used throughout.

**We did**: a fusion family that concatenates PCA-64 frozen embeddings with the 390-d spectral
block and feeds a single XGBoost.

**Differences**: ours is **early** fusion (concatenation), theirs is independent training plus
**late** fusion. Our fusion scored 0.600 against 0.627 for the spectral model alone — we explained
it as the concatenation diluting the stable features.

**Why they may score higher**: late fusion preserves each branch's decision surface, whereas
concatenating heterogeneous scales mixes them; their EEG-only design also matches our C2 probe,
which found band powers to overlap across sites and therefore need no site harmonization.

**Worth copying**: re-run fusion as two independently trained branches combined by ranks or
probabilities — the cheapest fix available to us.

### 3.5. NeuroAI (test 0.704)

**They did**: multi-scale "feature-stable" descriptors — whole-night architecture and
cardiorespiratory summaries, EEG sleep depth at 3 s, interhemispheric EEG contrasts at 5 s, event
timing at 1 s — computed within a recording or as ratios between homologous channels so that
amplifier gain and montage scale cancel before modelling; a feature is retained only if its
age-conditioned association agrees in direction across all three training sites. Their
leave-one-site-out score is 0.823, the reported official-phase validation score 0.838, and the
final entry scores 0.708 / 0.704 in the official table.

**We did**: feature-side stability work too — the C1 domain probe (to locate the most-shifted
blocks), D3 deleting the most-shifted columns (−0.050), D4 domain randomization (+0.016).

**Differences**: our remedies were **post hoc distribution corrections** (B2, C4, D3); theirs act
at feature definition. We also never used a cross-site consistency criterion for selection.

**Why they may score higher**: C4's lesson was that the distribution can be repaired exactly while
the score does not move, which suggests the limiting factor is how much usable signal survives,
not how well the distributions are aligned. Cancelling units inside the feature keeps the
physiology while removing the site.

**Extra value**: their LOSO 0.823 → official 0.704 gap is external evidence for our paper's claim
that leave-one-site proxies are optimistic.

### 3.6. Better Call Sandman (test 0.703)

**They did**: two routes — engineered physiological/sleep features with XGBoost (training-set CV
0.870, official validation 0.636) and a pretrained CNN with transfer learning (validation 0.574,
using additional HSP data); XGBoost was selected and scored 0.703 on the test set.

**We did**: the closest match to us — engineered features with gradient-boosted trees
(390-d spectral XGBoost) — and we also tried deep models (CRNN, frozen embeddings).

**Differences**: their feature set is broader and they use the large set; our spectral model has
390 features and was trained on 1,103 records.

**Why they may score higher**: data volume and feature coverage. Their CV 0.870 → official 0.703
gap mirrors our own 0.645-0.749 → 0.632 discount, and their CNN losing to the tree model shows
that depth alone is not the answer.

### 3.7. NLICA (test 0.703)

**They did**: after within-site location-scale harmonization they remove age as a direct
predictor and subtract per-feature linear age trends estimated on negative-class records
(age residualization) from 272 numeric features; a 30-member tree ensemble and a compact 1-D CNN
over artifact-screened N2/N3/REM epochs from four EEG derivations are combined by equal-weight late
fusion. Their leave-one-site-out mean is 0.694 with a worst site of 0.654.

**We did**: heavy age interventions, but all **model-side** — FiLM age conditioning, O1
age-adversarial training (−0.027), O4/D1 zeroing the age channel (≈0), O7c age-stratified sampling
with a pairwise loss (refuted on rerun).

**Differences**: they residualize age in the **features**, and they perform within-site
normalization, which corresponds to our B2/C4 attempts that we did not adopt.

**Why they may score higher**: the age-conditioned AUROC only compares patients of similar age,
so subtracting the age trend from the features is more direct than asking a model not to use age,
and it costs no samples. Our O4 result only showed that removing the age input from the model is
neutral; feature-side residualization was never tested.

**Worth copying**: age residualization as a feature-engineering step — cheap and compatible with
our existing spectral block.

### 3.8. Koalalition (test 0.683)

**They did**: 136 sleep-stage-stratified, literature-motivated physiological features plus 23
age-residual features, classified by a bagging ensemble of 15 XGBoost models with randomized
hyperparameter search under stratified cross-validation; large set.

**We did**: essentially the same family — D2-tabular's 641-d bank with LightGBM/XGBoost on the
small set, and the submitted 390-d spectral model.

**Differences**: (i) age-residual features (we have none); (ii) large vs. small; (iii) a 15-model
bagging ensemble vs. our single model; (iv) their features are organized per sleep stage and
selected, whereas we feed the whole bank.

**Why they may score higher**: within the same feature family, training pool + ensembling +
age handling plausibly account for the ~0.05 difference. Our proxy 0.655 and their LOSO 0.694 are
the same order of magnitude, so our feature engineering is competitive; our training regime is not.

### 3.9. REMedy (test 0.681)

**They did**: explainable-AI-driven feature selection (network attention plus GradCAM) over EEG,
ECG and respiratory signals; ECG-derived biological age and the biological-chronological age gap;
seven XGBoost classifiers, one per non-empty modality combination, with each recording routed
dynamically; the threshold is tuned for F1; large set.

**We did**: XGBoost on a multi-domain feature bank that includes HRV and SpO₂ (D2-tabular), with a
constant fallback when a modality is missing (the CRNN fills (0, 0.5) when CAISR is unavailable).

**Differences**: no modality routing, no interpretability-driven feature selection (we feed all
641 features, and our Random Forest reached only 0.576 — a sign of noisy features), no
ECG-derived biological age, and again the small set.

**Why they may score higher**: modality availability differs by site (our C1 probe found the
event features corrupted on I0004 and a different extreme on I0007), so modelling missing
modalities explicitly is more robust than constant filling.

**Worth copying**: modality availability indicators with combination routing; attention/GradCAM
feature selection before fitting the tree model.

### 3.10. bashlab (test 0.678)

**They did**: 151 named features (86 classical PSG summaries, 63 describing the distribution of
30-s epoch band power rather than its whole-night mean, and **2 recording-date features**),
aligned to the scored site with CORAL using unlabelled target recordings, then scored by a rank
blend of TabFM (a tabular foundation model used in context, weight 0.7) and a tree (0.3). Their
own ablation attributes the largest single increment, 0.092, to the recording date.

**We did**: engineered features with trees and rank blending (the spectral family), and
distribution alignment attempts (B2 z-scoring, C4 ComBat).

**Differences**: they used recording dates; we never use date-like features and were therefore
unaffected when the organizers removed them. They also summarize the *distribution* of band power
across epochs, whereas our spectral block stores stage-wise means.

**Why they previously scored higher**: the date features leaked follow-up-window information —
after 2020 the set holds positives with no negatives, which inflated their validation score to
0.751. With dates removed their validation falls to 0.528 (below our 0.627), although their test
score of 0.678 still exceeds ours, so their remaining features and the TabFM blend have real value.

**Lesson**: any feature coupled to the follow-up window or acquisition time must be ablated
single-factor. Our C1 probe had the right instinct but never covered dates.

### 3.11. Momochi-SleepAI (test 0.673)

**They did**: ten CAISR-derived sleep and event summaries plus three spectral descriptors computed
from sixteen deterministic 30-s EEG windows per recording, fitted by a class-balanced logistic
regression; leave-one-site-out 0.604-0.775, official validation 0.601, test 0.673; large set.

**We did**: the minimal-route comparison is our CRNN on CAISR features (large set) and our
spectral XGBoost (small set) — they use a model three orders of magnitude smaller.

**Differences**: (i) they include a handful of EEG spectral descriptors, our CRNN is pure CAISR;
(ii) model complexity; (iii) they train on the large set while our spectral model is small-only.

**Why they may score higher**: a very small model on a handful of robust features, trained on
6,600 recordings, transfers better than our large model trained on 1,103 (B1: the small pool costs
0.092). Adding just three EEG descriptors improved their AUROC by 0.019-0.043, which underlines
that even the crudest EEG information is valuable.

### 3.12. Biosignal Pilots (test 0.671)

**They did**: features organized around two themes — sleep depth (slow-wave activity, odds ratio
product, delta power entropy) and sleep fragmentation (cyclic alternating pattern, arousal timing
distribution, respiratory event indices); ANOVA plus mutual-information selection; XGBoost with
class weighting; SHAP analysis; large set. CV 0.661 ± 0.037, official validation 0.636.

**We did**: our 15-d whole-night branch covers the same clinical concepts (N3%, arousal index,
AHI, transitions/h, REM/N3 latency), and our 641-d bank includes a CAISR architecture block
(0.576 on its own).

**Differences**: their feature set is thematically concentrated and selected, while ours is broad
(coherence, temporal quantiles, HRV and SpO₂ each scored 0.40-0.52 on their own — effectively
noise); we have no CAP, ORP or entropy features; and we trained on the small set.

**Why they may score higher**: a themed, selected feature set plus the large set. A broad bank on
1,103 records overfits (our Random Forest reached only 0.576).

### 3.13. MeDSP (test 0.671)

**They did**: multi-domain features (sleep architecture, EEG spectra, SpO₂, respiratory events,
arousals, limb movements, hypnogram complexity) reduced to the top 20 and passed to a 300-tree
Random Forest; leave-one-site-out 0.68 ± 0.04, official validation 0.708, test 0.671.

**We did**: the same family (multi-domain features with a tree model under LOSO), but we feed all
641 features rather than a selected subset, and our Random Forest lagged at 0.576.

**Why they may score higher**: feature selection and data volume. Their validation-to-test drop
(−0.037) also contrasts with our +0.005, suggesting their validation-based selection was more
optimistic than our proxy protocol.

### 3.14. BAPORLab (test 0.670)

**They did**: a first-stage CNN trained on human staging produces five-stage probabilities from 5-s
epochs; per stage, up to 128 high-probability epochs are selected across eight temporal bins to
preserve coverage; stage-specific experts (temporal convolutional encoders with Set Transformers)
are trained separately for central EEG, frontal EEG and EOG; a handcrafted EEG branch is fused
hierarchically. External Human Sleep Project recordings (excluding Challenge participants) were
used for pretraining; only physiological signals enter the final predictor. Official validation
0.706.

**We did**: epoch-sequence modelling (30-s epochs with CAISR features) and multi-scale/multi-branch
architectures in the repository, but never 5-s granularity, never human staging, and never
external pretraining.

**Why they may score higher**: finer temporal resolution, deliberate coverage-preserving epoch
selection, and external data as a volume multiplier. The lesson for us is that the time axis must
be paired with a representation fine and informative enough to exploit it.

### 3.15. OCA-CENTINEL (test 0.666)

**They did**: frozen CBraMod EEG embeddings plus automated staging and chin/leg EMG features;
embedding dimensions dominated by site information are removed using mutual information; bagged
LightGBM and linear heads are trained with age-matched pairwise ranking and channel dropout.
Leave-one-site-out 0.69 (worst site 0.67), official validation 0.658, test 0.666. They conclude
that increasing model or representation capacity improves within-cohort validation but not
held-out-site performance.

**We did**: both of their ingredients, separately — the frozen Philosopher's Stone latent reduced
by PCA (P4: 0.557-0.583; fusion 0.600) and pairwise/ranking losses (O7a −0.039; O7c refuted on
rerun).

**Differences**: (i) they remove site-dominated embedding dimensions with mutual information, we
did not (our D3 deleted raw feature columns instead and lost 0.050); (ii) they combine the frozen
representation with low-dimensional robust features (EMG, staging), we only used PCA components;
(iii) they use channel dropout and the large set.

**Why they may score higher**: their changes are capacity-neutral — dimension selection plus a
ranking-aligned objective — rather than "bigger model". Our PCA-64 keeps site-dominated variance,
which matches our own explanation that the frozen latent encodes site-specific structure; their
mutual-information filter removes exactly that.

**Worth copying**: filtering site information out of a frozen representation before ranking it.

### 3.16. PhysioWinn (test 0.658)

**They did**: a lightweight CNN with 79,457 parameters over six EEG channels; interpretation shows
frontal F3/F4 and N3 windows contributing most. Validation 0.748 (9th of 100 teams at the time),
test 0.658 (−0.090).

**We did**: compact sequence models, but always on CAISR features rather than raw EEG; the only
raw-signal representation we ever used was the frozen brain-health embedding (P4).

**Why they previously scored higher**: raw EEG carries more usable information than CAISR event
rates (our C1 conclusion). Their large validation-to-test drop nevertheless shows the limits of a
single-modality raw-signal model.

**Implication**: a small model over raw EEG already beats our 0.632, so the gap is about input
representation rather than model size.

### 3.17. Med_YNNU (test 0.636)

**They did**: DFHANet — five independent temporal dense multi-scale CNNs and five
frequency-domain CNNs over five PSG modalities, fused by branch-level multi-head self-attention,
a two-branch 1:1 connection and two further attention layers, then mask-based segment pooling.
Official validation 0.646, test 0.636.

**We did**: the multi-branch/multi-scale family (EpochTransformer, MultiBranchNet), but on CAISR
features.

**Why they are only 0.004 above us**: a large network over frequency-domain representations does
not by itself solve cross-site transfer (their Δ = −0.010), which supports our paper's claim that
representation matters more than capacity.

### 3.18. UPV Maths (test 0.635)

**They did**: a lightweight head on a frozen SleepFM backbone, with a systematic study of input
combinations and temporal pooling; official validation 0.681, test 0.635; large set.

**We did**: the same recipe with a different backbone (Philosopher's Stone), but only PCA-64 with
XGBoost/LR and no input/pooling study.

**Why they may score higher**: a more systematic input/pooling exploration and the large set —
although both entries sit far below the non-frozen approaches (they rank 19th, we rank 20th),
which jointly supports our negative-transfer finding.

### 3.19. FuneLab (test 0.630)

**They did**: a multi-rate, channel-wise 1-D CNN consuming raw EEG, EOG, EMG, ECG, respiration and
SpO₂, plus demographics, algorithmic annotations and SpO₂ features; explicit availability
indicators mark missing channels; predictions from multiple two-minute windows are aggregated to
the patient level. Validation 0.729, test 0.630 (−0.099); their best entry was trained on the
small set.

**We did**: simplified versions of the same ideas (missing-modality fallback, multi-modal
aggregation), but never on raw waveforms and without window-level aggregation.

**Why they previously scored higher**: again, the information content of raw waveforms. Their
large validation-to-test drop shows how strongly this design can overfit the validation site.

**Implication**: end-to-end raw-signal pipelines were not uniformly rewarded, which supports our
decision not to abandon feature engineering — but it does not contradict the value of
information-rich per-epoch representations (see CLECLINIC and Matcha).

### 3.20. OUS_IVS (test 0.664)

*Added after the corrected results table of 1 October 2026 promoted this entry to 16th; it was
unranked in the preliminary table used for sections 3.1–3.19.*

**They did**: a hybrid representation combining frozen multimodal SleepFM embeddings (512-d from
four modality groups encoded in three five-minute windows) with clinically derived features
(160 CAISR descriptors, SpO₂, channel-availability indicators, and 10 age-relative physiological
features, with the age reference estimated inside each leave-one-site fold). The ranking pathway
(`Fusion730`) produced the continuous risk score used for Age-conditioned AUROC; because the
thresholded decision side collapsed across sites (Reward −0.293), they deliberately **decoupled**
ranking from classification and used a restricted ExtraTrees head over 182 clinical features
without the embeddings for the binary decision (Reward 0.156). Validation 0.708, test 0.664
(−0.044); Reward 0.156 → 0.069.

**We did**: the same frozen-embedding ingredient (Philosopher's Stone) with PCA-64 and XGBoost/LR,
but as a *ranking* model only, with no separate decision head, and we kept the embedding branch
only when it beat the tabular baseline locally (it did not: 0.583 / 0.557 / 0.600 vs 0.627).

**Differences**: they used the embeddings inside a much richer hybrid (their static block already
contains 160 CAISR descriptors plus oxygen saturation and age-relative features), trained on the
large pool, and — most importantly — treated ranking and thresholded decisions as two tasks with
different failure modes.

**Why they may score higher**: the embedding branch is not asked to do the work alone; the
clinical feature block carries the cross-site signal, and the frozen encoder mostly contributes
within-site structure. Their explicit ranking-vs-decision split is also the strongest external
support for our paper's metric analysis: our own entry with the *worst* Age-conditioned AUROC
(0.557) was the only one with a positive Reward, and the cohort-level pattern is the same — across
the 42 ranked entries the Spearman correlation between test Age-conditioned AUROC and test Reward
is only 0.64, i.e. the two official metrics disagree substantially.

**Implication for us**: (i) a hybrid embedding + clinical block with the *decision* head separated
from the ranking head is worth one experiment; (ii) our negative result on frozen embeddings
should be stated as "no gain as the primary ranking pathway", not "frozen embeddings do not
transfer", which is what their Reward column actually shows.

## 4. Cross-cutting patterns

1. **Three representation × time-axis regimes explain most of the spread.** (a) Aggregated features
   with trees, no time axis: us 0.632, bashlab 0.678, REMedy 0.681, Koalalition 0.683, MeDSP and
   Biosignal Pilots 0.671 → a 0.63-0.68 ceiling. (b) A time axis with impoverished features (pure
   CAISR event rates): our CRNN at 0.592-0.617 validation → 0.59-0.62. (c) A time axis with rich
   per-epoch representations, mostly EEG: CLECLINIC 0.708, Matcha 0.727, SIREN 0.749 → 0.70+.
2. **Site invariance works either by construction or by selection.** Construction: within-site
   rank normalization (SIREN), within-subject normalization (CLECLINIC), ratios between homologous
   channels (NeuroAI). Selection: cross-site directional consistency of feature-outcome
   associations (NeuroAI), mutual-information removal of site-dominated dimensions
   (OCA-CENTINEL). Every remedy we tried was a post hoc distribution correction (B2/C4/D3/D4) and
   most were rejected by our own protocol.
3. **We treated age on the model side only.** O1, O4/D1 and O7c all modified the model or loss;
   NLICA and Koalalition residualize age in the features, REMedy adds an ECG-derived biological
   age, OCA-CENTINEL uses age-matched pairwise ranking. This is the cheapest untried direction.
4. **Our fusion was the wrong kind.** Concatenation (early fusion) gave 0.600, below the 0.627 of
   the spectral model alone, whereas the high-ranked teams all use late fusion or rank averaging
   (CAU_KU, NLICA, SIREN, bashlab).
5. **Training pool is nearly unanimous.** Eighteen of these twenty teams trained on the large set;
   only FuneLab's selected entry and our submitted spectral model used the small set. Our richest
   features (641-d) exist only for 1,103 records because the large raw set is 1.2 TB, and B1 shows
   the small pool costs 0.092 on the I0006 holdout.
6. **Cross-validated and leave-one-site scores are systematically optimistic.** Matcha 0.804 →
   0.727, NeuroAI 0.823 → 0.704, Better Call Sandman 0.870 → 0.703, OCA-CENTINEL 0.69 → 0.666,
   ours 0.645-0.749 → 0.632. This is external evidence for our paper's claim that the fidelity of
   a held-out-site proxy is family-dependent.
7. **Frozen foundation models did not pay off.** UPV (SleepFM) 0.681/0.635, OCA-CENTINEL
   (CBraMod, used as one of several feature families) 0.658/0.666, ours (Philosopher's Stone)
   0.557-0.583. The only self-supervised success, SIREN, pretrained on this year's data and
   combined the result with rank averaging.
8. **Date leakage mattered.** bashlab is the only top-20 team that explicitly used recording
   dates, and removing them dropped its validation score from 0.751 to 0.528. We never used
   date-like features, but the episode argues for covering acquisition-time features in single
   factor ablations.

## 5. What we would try next

Ordered by expected value per unit of effort:

1. **Per-epoch EEG features into a sequence model.** Compute six-band relative power, slowing
   ratio, Hjorth mobility/complexity, kurtosis and line length per 30-s epoch over the six bipolar
   derivations and feed them to a TCN/InceptionTime with kernels matched to sleep-cycle scales.
   This is the single change that separates regime (c) from regimes (a) and (b).
2. **Move to the large training set** for whichever model we take forward (B1 shows the small pool
   costs 0.092 on the hard holdout).
3. **Replace concatenation with late fusion or rank averaging** across families.
4. **Add age residualization** to the feature pipeline.
5. **Re-test a rank loss** on top of the richer representation; our O7a/O7c negative results were
   obtained in a 21-d feature space with a noise floor larger than the effect sizes.
6. **Subject-level normalization plus modality dropout** for cross-site robustness.
7. **Filter site-dominated dimensions** out of frozen representations before ranking them.

## 6. Open questions

- Several papers report validation scores that differ from the official table (Matcha 0.847 →
  0.737, NeuroAI 0.838 → 0.708, bashlab 0.751 → 0.528). The differences may come from switching
  the final submission or from removing the sleep-study dates; the official table update after
  1 October 2026 will help separate the two.
- We read the 4-page papers only; the site probe in SIREN, the cross-site consistency filter in
  NeuroAI, the capacity-neutral analysis in OCA-CENTINEL and the InceptionTime details in
  CLECLINIC deserve a closer read.
- The oral sessions (for example Session S54) and the poster discussions are not covered here.

## References

1. Official results and rankings, hidden test set:
   <https://moody-challenge.physionet.org/2026/results/> (table mirrored at
   [results/challenge_2026_results.tsv](results/challenge_2026_results.tsv)).
2. Computing in Cardiology 2026 preliminary program (links to the accepted papers):
   <https://cinc.org/prelim_program_2026/>.
3. Challenge description: Reyna MA, Weigle A, Li Q, Koscova Z, Sun HS, Wen S, Katwa U, Thomas R,
   Hwang D, Trotti LM, Mignot E, Sameni R, Nasiri S, Westover MB, Clifford GD. Screening for
   Cognitive Impairment During Sleep Studies: The George B. Moody PhysioNet Challenge 2026.
   In Computing in Cardiology 2026, volume 53, p. 1-4.
4. Challenge data: Li Q, Wen S, Sun H, Ganglberger W, Tripathi A, Turley N, et al. The Human Sleep
   Project: A Multi-Center Clinical Polysomnography Dataset Across the Human Lifespan. Sleep,
   2026;zsag215.
5. Our own experiment log: [ROADMAP.md](ROADMAP.md).
