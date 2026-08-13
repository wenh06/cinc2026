"""Comprehensive O7c tests: sampler statistics + loss numerical verification
+ no_age/pairwise combination via a real model forward.

Run: python scripts/test_o7c.py
"""

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset as dataset_mod
from cfg import ModelCfg, TrainCfg
from dataset import AgeStratifiedBatchSampler, CINC2026Dataset, collate_fn
from models import EpochCRNN
from models.epoch_crnn import AgeMatchedPairwiseLossTanh


def main():
    SPLIT_FILE = Path(__file__).resolve().parent / "split_I0006.json"
    with open(SPLIT_FILE) as f:
        dataset_mod.FIXED_DATA_SPLIT_FILE = str(SPLIT_FILE)

    cfg = deepcopy(TrainCfg)
    cfg.db_dir = str(Path(__file__).resolve().parents[1] / "data" / "official-phase-large")
    cfg.folds = None
    ds = CINC2026Dataset(cfg, training=True, lazy=True)
    df = ds.reader._df_records
    ages = df.loc[ds.records, "Age"].astype(float).values
    labels = df.loc[ds.records, "Cognitive_Impairment"].astype(int).values

    TOL = 2.0
    PASS = []

    def check(name, cond, detail=""):
        PASS.append((name, bool(cond)))
        print(f"[{'OK ' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    # ══════════════════════════════════════════════════════════════════════
    # A. Sampler: full-epoch statistics (n_mixed_neg=4)
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== A. Sampler full-epoch statistics ===")
    bs = AgeStratifiedBatchSampler(ds, batch_size=16, n_pos=4, n_mixed_neg=4, tolerance=TOL, seed=0)

    n_batches = 0
    pos_counts, same_age_pairs, age_spans = [], [], []
    for b in bs:
        n_batches += 1
        b = np.asarray(b)
        pos_counts.append(labels[b].sum())
        p_idx = b[labels[b] == 1]
        n_idx = b[labels[b] == 0]
        pairs = 0
        for p in p_idx:
            pairs += int((np.abs(ages[n_idx] - ages[p]) <= TOL).sum())
        same_age_pairs.append(pairs)
        age_spans.append(ages[b].max() - ages[b].min())

    pos_counts = np.array(pos_counts)
    same_age_pairs = np.array(same_age_pairs)
    age_spans = np.array(age_spans)
    print(f"batches: {n_batches} (= ceil({len(ds)}/16))")
    print(f"n_pos/batch: min={pos_counts.min()} mean={pos_counts.mean():.2f} max={pos_counts.max()}")
    print(f"age-matched pairs/batch: min={same_age_pairs.min()} mean={same_age_pairs.mean():.1f}")
    print(f"batch age span: mean={age_spans.mean():.1f} max={age_spans.max():.1f} y")
    check("A1: every batch has >= 1 positive", pos_counts.min() >= 1)
    check("A2: every batch has >= 1 age-matched (pos,neg) pair", same_age_pairs.min() >= 1)
    check("A3: mean age-matched pairs >= 24", same_age_pairs.mean() >= 24)
    check("A4: mixed negatives widen age span (> 2*tolerance)", age_spans.max() > 2 * TOL)

    # positive coverage / over-sampling factor
    pos_seen = {}
    for b in bs:
        b = np.asarray(b)
        for p in b[labels[b] == 1]:
            pos_seen[int(p)] = pos_seen.get(int(p), 0) + 1
    n_pos_total = int(labels.sum())
    print(
        f"positives covered in one epoch: {len(pos_seen)}/{n_pos_total} "
        f"({len(pos_seen) / n_pos_total * 100:.1f}%), mean visits {np.mean(list(pos_seen.values())):.1f}x"
    )
    check("A5: most positives visited at least once per epoch", len(pos_seen) >= 0.8 * n_pos_total)

    # ══════════════════════════════════════════════════════════════════════
    # B. Loss: numerical check against hand-computed values
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== B. Loss numerical verification ===")
    K, M = 2.0, 0.25
    loss_fn = AgeMatchedPairwiseLossTanh(margin=M, tolerance=TOL, bank_size=0, k=K)

    def L(z, lab, age100):
        z = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        lab = torch.tensor(lab, dtype=torch.float64)
        age100 = torch.tensor(age100, dtype=torch.float64)
        return loss_fn(z, lab, age100).item()

    # B1: hand-computed — pos z=0.5 (60y); neg z=0.1 (61y, d=0.1−0.5+0.25=−0.15→0),
    #     z=0.9 (61y, d=0.65 → tanh(1.3)); z=1.5 (63y, gap 3 > 2 → dropped)
    expected_b1 = np.tanh(2.0 * 0.65) / 2.0
    got_b1 = L([0.5, 0.1, 0.9, 1.5], [1.0, 0, 0, 0], [0.60, 0.61, 0.61, 0.63])
    check(
        "B1: hand-computed case matches math",
        abs(got_b1 - expected_b1) < 1e-12,
        f"got {got_b1:.6f}, expected {expected_b1:.6f}",
    )

    # B2: all correctly ranked (margin satisfied) → loss == 0
    got_b2 = L([1.0, 0.0, 0.0], [1.0, 0, 0], [0.60, 0.61, 0.61])
    check("B2: all pairs satisfy margin → loss = 0", got_b2 == 0.0, f"got {got_b2}")

    # B3: saturation — grossly inverted pair → loss ≈ 1 (bounded, not hinge-linear)
    got_b3 = L([0.0, 5.0], [1.0, 0], [0.60, 0.61])
    check("B3: grossly inverted pair saturates near 1", got_b3 > 0.99, f"got {got_b3:.6f}")

    # B4: gradient directions — ∂L/∂z_pos < 0, ∂L/∂z_neg > 0 for an inverted pair
    zz = torch.tensor([0.5, 1.0], dtype=torch.float64, requires_grad=True)
    l4 = loss_fn(zz, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.60, 0.61], dtype=torch.float64))
    g = torch.autograd.grad(l4, zz)[0]
    check("B4: dL/dz_pos < 0 (push pos up)", g[0] < 0, f"g_pos={g[0]:.4f}")
    check("B4: dL/dz_neg > 0 (push neg down)", g[1] > 0, f"g_neg={g[1]:.4f}")

    # B5: margin boundary — d = 0 exactly → loss = 0
    got_b5 = L([0.5, 0.25], [1.0, 0], [0.60, 0.61])  # z_neg − z_pos = −0.25 = −margin
    check("B5: pair at exactly margin boundary → loss = 0", got_b5 == 0.0, f"got {got_b5}")

    # B6: just below margin → small loss AND dense gradient (unlike hinge)
    zz6 = torch.tensor([0.5, 0.3], dtype=torch.float64, requires_grad=True)  # d = 0.05
    l6 = loss_fn(zz6, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.60, 0.61], dtype=torch.float64))
    g6 = torch.autograd.grad(l6, zz6)[0]
    check(
        "B6: just-below-margin pair: small loss + dense gradient",
        abs(l6.item() - np.tanh(2.0 * 0.05)) < 1e-12 and (g6 != 0).any(),
        f"loss={l6.item():.6f}, grad={g6.tolist()}",
    )

    # B7: tanh vs hinge gradient shape — grossly-inverted pair gets *weaker* gradient
    zz7a = torch.tensor([0.5, 0.6], dtype=torch.float64, requires_grad=True)  # d = 0.35
    zz7b = torch.tensor([0.5, 4.0], dtype=torch.float64, requires_grad=True)  # d = 3.75 (saturated)
    l7a = loss_fn(zz7a, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.60, 0.61], dtype=torch.float64))
    l7b = loss_fn(zz7b, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.60, 0.61], dtype=torch.float64))
    ga = torch.autograd.grad(l7a, zz7a)[0]
    gb = torch.autograd.grad(l7b, zz7b)[0]
    check(
        "B7: saturated pair gets weaker gradient than near-boundary pair",
        float(ga.abs().sum()) > float(gb.abs().sum()),
        f"|grad| near-boundary {ga.abs().sum():.4f} vs grossly-inverted {gb.abs().sum():.4f}",
    )

    # B8: label-smoothing targets (0.95/0.05) pair exactly like raw labels
    l8a = L([0.5, 0.1, 0.9], [1.0, 0, 0], [0.60, 0.61, 0.61])
    l8b = L([0.5, 0.1, 0.9], [0.95, 0.05, 0.05], [0.60, 0.61, 0.61])
    check("B8: smoothed labels pair identically to raw", abs(l8a - l8b) < 1e-12, f"raw {l8a:.6f} vs smoothed {l8b:.6f}")

    # B9: age-unit contract — the loss REQUIRES age/100 units (demographics
    # convention, same as AgeMatchedPairwiseLossHinge).  Feeding years must
    # break pairing (0.01 vs 0.02 tolerance), documenting the contract.
    l9a = L([0.5, 0.1, 0.9], [1.0, 0, 0], [0.60, 0.61, 0.61])  # age/100 → pairs
    l9b = L([0.5, 0.1, 0.9], [1.0, 0, 0], [60.0, 61.0, 61.0])  # years → no pairs
    check(
        "B9: age/100 contract — years input breaks pairing as documented",
        l9a > 0.3 and l9b == 0.0,
        f"age/100 {l9a:.6f} (pairs), years {l9b:.6f} (no pairs)",
    )

    # B10: no valid pair → loss 0, still differentiable
    zz10 = torch.tensor([0.5, 0.5], dtype=torch.float64, requires_grad=True)
    l10 = loss_fn(zz10, torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([0.60, 0.70], dtype=torch.float64))
    g10 = torch.autograd.grad(l10, zz10)[0]
    check("B10: no age-matched pair → loss 0, zero grads, no crash", l10.item() == 0.0 and g10.sum().item() == 0.0)

    # ══════════════════════════════════════════════════════════════════════
    # C. no_age + pairwise combination — real model forward
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== C. no_age + pairwise on a real model forward ===")
    for no_age in (False, True):
        mc = deepcopy(ModelCfg.epoch_crnn_M)
        mc.no_age = no_age
        # λ = 10 only for this wiring check: the pairwise branch is gated on
        # self.training (model.forward:693), so train() mode is required, and a
        # large λ makes λ·pairwise dominate dropout noise — the comparison
        # becomes deterministic.  Real training λ lives in cfg/lo_site scripts.
        mc.age_pairwise = {
            "enable": True,
            "variant": "tanh",
            "k": K,
            "margin": M,
            "tolerance": TOL,
            "bank_size": 0,
            "lambda_": 10.0,
        }
        # identical init for both models → the only difference is the
        # pairwise branch (init noise eliminated)
        torch.manual_seed(0)
        model = EpochCRNN(**mc)
        mc2 = deepcopy(mc)
        mc2.age_pairwise = None
        torch.manual_seed(0)
        model2 = EpochCRNN(**mc2)
        model.train()
        model2.train()
        # one real batch from the sampler (rename label → labels as run_one_step does)
        batch_idx = next(iter(AgeStratifiedBatchSampler(ds, batch_size=16, n_pos=4, n_mixed_neg=4, tolerance=TOL, seed=2)))
        loader = DataLoader(ds, batch_sampler=[batch_idx], num_workers=0, collate_fn=collate_fn)
        batch = next(iter(loader))
        batch["labels"] = batch.pop("label").to(torch.float32)
        out = model(batch)
        loss_val = out["ci_loss"]
        check(
            f"C{int(no_age)}: forward ok, ci_loss finite & non-zero (no_age={no_age})",
            torch.isfinite(loss_val) and loss_val.item() > 0,
            f"loss={loss_val.item():.4f}",
        )
        # verify the pairwise branch fires: lambda-weighted, so total > plain focal alone
        batch2 = dict(batch)
        out2 = model2(batch2)
        check(
            f"C{int(no_age)}: pairwise term contributes (total > focal-only)",
            out["ci_loss"].item() > out2["ci_loss"].item(),
            f"with-pair {out['ci_loss'].item():.4f} vs focal-only {out2['ci_loss'].item():.4f}",
        )

    # ══════════════════════════════════════════════════════════════════════
    # D. Multi-worker DataLoader integration
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== D. DataLoader(num_workers=4) integration ===")
    bs2 = AgeStratifiedBatchSampler(ds, batch_size=16, n_pos=4, n_mixed_neg=4, tolerance=TOL, seed=3)
    loader = DataLoader(ds, batch_sampler=bs2, num_workers=4, pin_memory=False, collate_fn=collate_fn)
    ok_d = True
    n_seen = 0
    for batch in loader:
        b = batch["label"]
        if b.dim() == 2:
            b = b.squeeze(-1)
        if int(b.sum()) < 1:
            ok_d = False
            break
        n_seen += 1
        if n_seen >= 5:
            break
    check("D: 4-worker loader produces batches with positives", ok_d, f"{n_seen} batches")

    print("\n" + "=" * 60)
    n_fail = sum(1 for _, c in PASS if not c)
    print(f"RESULT: {len(PASS) - n_fail}/{len(PASS)} passed" + ("  ALL OK" if n_fail == 0 else f"  {n_fail} FAILED"))


if __name__ == "__main__":
    main()
