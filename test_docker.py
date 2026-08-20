""" """

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from dataset import CINC2026Dataset, collate_fn
from evaluate_model import evaluate_model as _evaluate_model
from evaluate_model import run as model_evaluator_func
from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients
from models import EpochCRNN, EpochTransformer
from run_model import run as model_runner_func
from team_code import _MODEL_CLASS_MAP, _resolve_db_dir, train_model
from trainer import CINC2026Trainer
from utils.misc import func_indicator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("mount_data_dir", _BASE_DIR / "tmp" / "CINC2026")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")

tmp_model_dir = Path(os.environ.get("revenger_model_dir", TrainCfg.model_dir)).resolve()
tmp_model_dir.mkdir(parents=True, exist_ok=True)
tmp_output_dir = Path(os.environ.get("revenger_output_dir", _BASE_DIR / "tmp" / "output")).resolve()
tmp_output_dir.mkdir(parents=True, exist_ok=True)


def echo_write_permission(folder: Union[str, Path]) -> None:
    is_writeable = "is writable" if os.access(str(folder), os.W_OK) else "is not writable"
    print(f"{str(folder)} {is_writeable}")


echo_write_permission(tmp_data_dir)
echo_write_permission(tmp_model_dir)
echo_write_permission(tmp_output_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """Test CINC2026Dataset instantiation and basic data access."""
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = _resolve_db_dir(str(tmp_data_dir))
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(ds_config.working_dir)

    train_ds = CINC2026Dataset(config=ds_config, training=True, lazy=True)
    val_ds = CINC2026Dataset(config=ds_config, training=False, lazy=True)

    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"
    print(f"  Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    expected_fields = train_ds.data_fields - {"padding_mask"}  # padding_mask is added by collate_fn
    sample = train_ds[0]
    missing = expected_fields - set(sample.keys())
    assert not missing, f"Missing fields in sample: {missing}"

    ef = sample["epoch_features"]
    expected_dim = getattr(ModelCfg, ds_config.model_name).caisr_feat_dim
    assert ef.ndim == 2, f"epoch_features should be 2-D, got shape {ef.shape}"
    assert ef.shape[1] == expected_dim, f"epoch_features last dim should be {expected_dim}, got {ef.shape[1]}"

    demo = sample["demographics"]
    assert demo.shape == (3,), f"demographics should be shape (3,), got {demo.shape}"

    assert "label" in sample and sample["label"] in (0, 1, np.int64(0), np.int64(1))
    assert "record_id" in sample and isinstance(sample["record_id"], str)

    # Test collate_fn with a 2-sample mini-batch
    batch = collate_fn([train_ds[0], train_ds[1]])
    assert batch["epoch_features"].shape[0] == 2
    assert batch["padding_mask"].dtype == torch.bool
    print("  collate_fn OK")


@func_indicator("testing models")
def test_models() -> None:
    """Test EpochTransformer and EpochCRNN forward pass and inference.

    Always tests both model families (each at M-size).  The default model
    is determined by ``TrainCfg.model_name``; we also run a quick sanity check
    on the other family so that regressions in either are caught in CI.
    """
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    B, T = 2, 50

    for model_name, cfg_attr, model_cls in [
        ("epoch_transformer_M", ModelCfg.epoch_transformer_M, EpochTransformer),
        ("epoch_crnn_M", ModelCfg.epoch_crnn_M, EpochCRNN),
    ]:
        D = cfg_attr.caisr_feat_dim  # 23 for Transformer, 21 for CRNN
        model = model_cls(config=deepcopy(cfg_attr)).to(DEVICE)
        model.eval()

        epoch_features = torch.randn(B, T, D, device=DEVICE)
        demographics = torch.randn(B, 3, device=DEVICE)
        padding_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
        padding_mask[0, 40:] = True  # simulate a shorter record in slot 0

        input_dict = {
            "epoch_features": epoch_features,
            "demographics": demographics,
            "padding_mask": padding_mask,
        }
        with torch.no_grad():
            out = model(input_dict)

        assert "ci_logits" in out, f"{model_name}: ci_logits missing from output"
        assert out["ci_logits"].shape == (
            B,
            1,
        ), f"{model_name}: expected ci_logits (B,1), got {out['ci_logits'].shape}"

        # Inference (single sample, numpy)
        feat_np = np.random.randn(100, D).astype(np.float32)
        demo_np = np.array([0.65, 1.0, 0.5], dtype=np.float32)
        outputs = model.inference(epoch_features=feat_np, demographics=demo_np)
        assert outputs.ci_prob.shape == (
            1,
            2,
        ), f"{model_name}: expected ci_prob (1,2), got {outputs.ci_prob.shape}"
        prob = outputs.ci_prob[0, 1].item()
        assert 0.0 <= prob <= 1.0, f"{model_name}: probability out of [0,1]: {prob}"

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {model_name}: params={n_params:,}  CI prob={prob:.4f}  OK")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test evaluate_model with synthetic predictions (official phase API)."""
    rng = np.random.default_rng(42)
    n = 20
    patient_ids = [f"sub-{i:04d}" for i in range(n)]
    site_ids = ["S0001"] * n
    ages = rng.integers(40, 90, size=n).astype(float)
    labels = rng.integers(0, 2, size=n)
    probs = rng.uniform(0.1, 0.9, size=n)
    binary_preds = (probs >= 0.5).astype(int)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_file = os.path.join(tmpdir, "labels.csv")
        preds_file = os.path.join(tmpdir, "predictions.csv")
        # Prevalence file uses the same format as labels; for testing we
        # reuse the labels file itself.
        prev_file = labels_file

        pd.DataFrame(
            {
                "SiteID": site_ids,
                "BDSPPatientID": patient_ids,
                "Cognitive_Impairment": labels,
                "Age": ages,
            }
        ).to_csv(labels_file, index=False)

        pd.DataFrame(
            {
                "SiteID": site_ids,
                "BDSPPatientID": patient_ids,
                "Cognitive_Impairment": binary_preds,
                "Cognitive_Impairment_Probability": probs,
            }
        ).to_csv(preds_file, index=False)

        reward, auroc_age, auroc_weighted, auroc, auprc, accuracy, f_measure, table = _evaluate_model(
            [labels_file], [preds_file], [prev_file]
        )

    for name, val in [
        ("auroc", auroc),
        ("auprc", auprc),
        ("auroc_age", auroc_age),
        ("auroc_weighted", auroc_weighted),
        ("accuracy", accuracy),
        ("f_measure", f_measure),
        ("reward", reward),
    ]:
        assert isinstance(val, float), f"{name} should be float, got {type(val)}"
        if name != "reward":
            assert 0.0 <= val <= 1.0, f"{name} out of [0,1]: {val}"
    assert isinstance(table, list) and len(table) > 0, "table should be non-empty list"
    print(
        f"  AUROC={auroc:.3f}  AUROC_age={auroc_age:.3f}  AUROC_weighted={auroc_weighted:.3f}  AUPRC={auprc:.3f}  Reward={reward:.3f}"
    )


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test CINC2026Trainer for 1 epoch (quick smoke test)."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = _resolve_db_dir(str(tmp_data_dir))
    train_config.n_epochs = 3
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "test_trainer_working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)
    train_config.model_dir = train_config.working_dir / "checkpoints"
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.log_dir = train_config.working_dir / "log"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    # In CI (CINC2026_REVENGER_TEST set) the runner has limited RAM; shrink batch.
    if os.environ.get("CINC2026_REVENGER_TEST"):
        train_config.batch_size = 4

    model_config = deepcopy(getattr(ModelCfg, train_config.model_name))
    model_cls = _MODEL_CLASS_MAP[train_config.model_name]
    model = model_cls(config=model_config).to(DEVICE)

    trainer = CINC2026Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )
    trainer.train()

    # Verify that at least one checkpoint was written
    ckpts = list(train_config.model_dir.glob("*.pth*")) + list(train_config.model_dir.glob("*.safetensors"))
    print(f"  Trainer completed 1 epoch. Checkpoints found: {len(ckpts)}")


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Full pipeline test via the official challenge entry scripts.

    Mirrors how PhysioNet evaluates submissions:

    1. ``train_model.py``  →  ``team_code.train_model``
    2. ``run_model.py``    →  ``team_code.load_model`` + ``team_code.run_model``
    3. ``evaluate_model.py`` →  scoring
    """
    # cfg default now routes submissions through the tabular path (sub5);
    # this test exercises the CRNN entry, so force the CRNN branch for the
    # whole train → load → run sequence (load_model reads the global TrainCfg).
    TrainCfg.tabular.enable = False
    TrainCfg.phi.enable = False
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    db_dir = _resolve_db_dir(str(tmp_data_dir))
    train_data_dir = None
    for part in ["training_set_small", "training_set_large", "training_set"]:
        candidate = db_dir / part
        if candidate.exists():
            train_data_dir = candidate
            break
    if train_data_dir is None:
        train_data_dir = tmp_data_dir

    if not (train_data_dir / DEMOGRAPHICS_FILE).exists():
        print(f"  No demographics.csv found at {train_data_dir}; skipping test_entry.")
        return

    entry_model_dir = tmp_model_dir / "test_entry_model"
    entry_model_dir.mkdir(parents=True, exist_ok=True)
    entry_output_dir = tmp_output_dir / "test_entry_output"
    entry_output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Train (short run for speed; CI shrinks batch to avoid OOM)
    # ------------------------------------------------------------------
    print("   Train model   ".center(100, "#"))

    # Build CI-specific config overrides and inject via CINC2026_OVERRIDE_JSON.
    # n_epochs=3 keeps the test fast; batch_size=4 prevents OOM in CI.
    _ci_cfg: dict = {"n_epochs": 3}
    if os.environ.get("CINC2026_REVENGER_TEST"):
        _ci_cfg["batch_size"] = 4

    prev_override = os.environ.get("CINC2026_OVERRIDE_JSON")
    _tmp_override = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(_ci_cfg, _tmp_override)
    _tmp_override.close()
    os.environ["CINC2026_OVERRIDE_JSON"] = _tmp_override.name
    try:
        train_model(str(train_data_dir), str(entry_model_dir), verbose=True)
    finally:
        os.unlink(_tmp_override.name)
        if prev_override is None:
            os.environ.pop("CINC2026_OVERRIDE_JSON", None)
        else:
            os.environ["CINC2026_OVERRIDE_JSON"] = prev_override

    # ------------------------------------------------------------------
    # 2. Run inference via run_model.py entry point
    # ------------------------------------------------------------------
    print("   Run model (run_model.py)   ".center(100, "#"))
    model_runner_args = CFG(
        data_folder=str(train_data_dir),
        model_folder=str(entry_model_dir),
        output_folder=str(entry_output_dir),
        allow_failures=False,
        verbose=True,
    )
    model_runner_func(model_runner_args)

    predictions_file = entry_output_dir / DEMOGRAPHICS_FILE
    assert predictions_file.exists(), f"Predictions file not created: {predictions_file}"
    print(f"  Predictions written to {predictions_file}")

    # ------------------------------------------------------------------
    # 3. Evaluate via evaluate_model.py entry point
    # ------------------------------------------------------------------
    print("   Evaluate model (evaluate_model.py)   ".center(100, "#"))
    score_file = entry_output_dir / "score.txt"
    table_file = entry_output_dir / "table.csv"
    # Official phase: labels_files, predictions_files, prevalence_files are all
    # lists; -p defines the population used to compute age-specific prevalence.
    model_evaluator_args = CFG(
        labels_files=[str(train_data_dir / DEMOGRAPHICS_FILE)],
        predictions_files=[str(predictions_file)],
        prevalence_files=[str(train_data_dir / DEMOGRAPHICS_FILE)],
        score_file=str(score_file),
        table_file=str(table_file),
    )
    model_evaluator_func(model_evaluator_args)

    if score_file.exists():
        print("Score file contents:")
        print(score_file.read_text())
    if table_file.exists():
        print(f"Age-breakdown table written to {table_file}")

    print("test_entry passed ✓")


# Allow test_entry to be referenced as test_team_code for compatibility
test_team_code = test_entry


@func_indicator("testing raw spectral features")
def test_spectral_features() -> None:
    """Run the spectral feature extractor on the raw PSG subset (3 records)."""
    import subprocess
    import sys

    raw_dir = tmp_data_dir / "physiological_data"
    edfs = sorted(raw_dir.glob("*/*.edf")) if raw_dir.exists() else []
    if not edfs:
        print("  No raw PSG subset present — skipping (set MEGA_RAW_ACTION_TEST_URL to enable).")
        return
    out_dir = tmp_model_dir / "spectral_features"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts" / "extract_spectral_features.py"),
            "--data-root",
            str(tmp_data_dir),
            "--out-dir",
            str(out_dir),
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    feat = pd.read_csv(out_dir / "features.csv", index_col=0)
    assert len(feat) == len(edfs), f"expected {len(edfs)} rows, got {len(feat)}"
    assert not feat.isna().all(axis=1).any(), "all-NaN feature row for a raw record"
    print(f"  spectral features: {feat.shape[0]} records x {feat.shape[1]} features")
    print("test_spectral_features passed ✓")


@func_indicator("testing Philosopher's Stone cache utilities")
def test_phi_cache() -> None:
    """Test C4-M1 resolution and cache loading without running the model."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.phi_cache import PHI_LATENT_DIM, PHI_SCORE_KEYS, _resolve_c4m1, load_phi_cache

    raw_dir = tmp_data_dir / "physiological_data"
    edfs = sorted(raw_dir.glob("*/*.edf")) if raw_dir.exists() else []
    if not edfs:
        print("  No raw PSG subset present — skipping (set MEGA_RAW_ACTION_TEST_URL to enable).")
        return
    for p in edfs:
        resolved = _resolve_c4m1(p)
        assert resolved is not None, f"failed to resolve C4-M1 for {p}"
        sig, fs = resolved
        assert fs > 0 and len(sig) > 0, f"empty C4-M1 for {p}"
    demo = pd.read_csv(tmp_data_dir / DEMOGRAPHICS_FILE)
    cache = load_phi_cache(tmp_model_dir / "phi_cache_nonexistent", demo.head(10))
    assert cache.shape == (10, PHI_LATENT_DIM + len(PHI_SCORE_KEYS))
    assert cache.isna().all().all(), "empty cache dir should yield all-NaN frame"
    print("test_phi_cache passed ✓")


@func_indicator("testing Phi demographics lookup")
def test_phi_demographics_lookup() -> None:
    """Regression: SessionID must reach ``load_demographics`` with its raw type.

    ``phi_component`` used to cast SessionID to str before the lookup, which
    never matches the int64 CSV column and silently yields an empty
    demographics dict -> NaN age -> NaN latent -> constant XGB output on
    cache-miss records (the official hidden set).  Guard the fix at the
    helper seam without touching the GPU path.
    """
    from phi_component import _phi_patient_data

    demo_file = tmp_data_dir / DEMOGRAPHICS_FILE
    records = find_patients(str(demo_file))
    assert records, "no records in the CI data subset"
    patient_data = _phi_patient_data(demo_file, records[0])
    assert patient_data, "empty demographics — SessionID type mismatch"
    age = patient_data.get(HEADERS["age"])
    assert age is not None and float(age) == float(age), f"invalid age {age!r}"
    print("test_phi_demographics_lookup passed ✓")


@func_indicator("testing Philosopher's Stone inference")
def test_phi_inference() -> None:
    """Run Phi latent extraction on ONE raw record (CPU; opt-in).

    Gated by ``CINC2026_TEST_PHI=1`` because it adds several minutes on the
    2-core / 7 GB CI runner.  The record is truncated to 10 minutes for the
    wavelet stage: the full-night transform alone needs ~9-18 GB of intermediate
    arrays, which OOMs the runner, and the 30-minute variant sat close enough to
    the 7 GB limit to be OOM-killed under runner contention.  The spectrogram is
    then padded back to the
    canonical 11 h so the model forward, head weights, etc. still run end to end
    exactly as in production.  The checkpoint is baked into the image at build
    time by post_docker_build.py (MODEL_CACHE_DIR/philosophers-stone/model_files/).
    """
    import subprocess
    import sys

    if not str2bool(os.environ.get("CINC2026_TEST_PHI", "0")):
        print("  CINC2026_TEST_PHI not set — skipping.")
        return
    raw_dir = tmp_data_dir / "physiological_data"
    edfs = sorted(raw_dir.glob("*/*.edf")) if raw_dir.exists() else []
    if not edfs:
        print("  No raw PSG subset present — skipping.")
        return
    model_cache = Path(os.environ.get("MODEL_CACHE_DIR", "/challenge/cache/revenger_model_dir"))
    checkpoint = model_cache / "philosophers-stone" / "model_files" / "SleepPhilosophersStone.ckpt"
    assert checkpoint.exists(), f"checkpoint missing at {checkpoint}"
    cache_dir = tmp_model_dir / "phi_cache"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "scripts" / "phi_cache_extract.py"),
            "--data-root",
            str(tmp_data_dir),
            "--checkpoint",
            str(checkpoint),
            "--cache-dir",
            str(cache_dir),
            "--workers",
            "1",
            "--limit",
            "1",
            "--max-seconds",
            "600",
            "--no-collect-heads",
        ],
        capture_output=True,
        text=True,
        timeout=5400,
    )
    if result.returncode != 0:
        print(f"--- phi extract returncode: {result.returncode} ---")
        print("--- phi extract stdout ---")
        print(result.stdout)
        print("--- phi extract stderr ---")
        print(result.stderr)
    assert result.returncode == 0, result.stderr[-2000:]
    npzs = sorted(cache_dir.glob("*/*.npz"))
    if len(npzs) != 1:
        print("--- phi extract stdout ---")
        print(result.stdout)
        print("--- phi extract stderr ---")
        print(result.stderr)
        failures = cache_dir / "failures.csv"
        if failures.exists():
            print("--- failures.csv ---")
            print(failures.read_text())
    assert len(npzs) == 1, f"expected 1 npz, got {npzs}"
    print(f"  phi latent cached: {npzs[0]}")
    print("test_phi_inference passed ✓")


@func_indicator("testing tabular pipeline")
def test_tabular() -> None:
    """End-to-end smoke of the D2 tabular path on the raw PSG subset.

    Part A drives the on-the-fly branch (extraction from raw EDFs — the
    official-environment path); part B drives the cache-first branch and
    regression-tests that metadata columns are filled from demographics at
    inference (guards against the silent all-NaN metadata bug).
    """
    from helper_code import load_demographics
    from tabular_pipeline import (
        assemble_record_features,
        load_tabular_model,
        run_tabular_model,
        train_tabular,
    )

    raw_dir = tmp_data_dir / "physiological_data"
    if not raw_dir.exists() or not any(raw_dir.glob("*/*.edf")):
        print("  No raw PSG subset present — skipping.")
        return

    records = find_patients(str(tmp_data_dir / DEMOGRAPHICS_FILE))
    hits = []
    for r in records:
        base = f"{r[HEADERS['bids_folder']]}_ses-{r[HEADERS['session_id']]}"
        if (raw_dir / r[HEADERS["site_id"]] / f"{base}.edf").exists():
            hits.append(r)
    assert hits, "no raw record found"
    rec = hits[0]

    cache_csv = tmp_model_dir / "spectral_features" / "features.csv"
    assert cache_csv.exists(), "test_spectral_features should have produced the feature cache"

    cfg = deepcopy(TrainCfg)
    cfg.db_dir = _resolve_db_dir(str(tmp_data_dir))
    cfg.tabular.enable = True
    cfg.tabular.model = "xgboost"
    cfg.tabular.xgb_params.n_estimators = 30
    cfg.tabular.feature_groups = ["spec"]
    cfg.tabular.workers = 1

    # Part A — on-the-fly extraction branch
    cfg.tabular.feature_cache = ""
    model_folder_a = tmp_model_dir / "tabular_test_onthefly"
    train_tabular(cfg, model_folder_a, verbose=True)
    model_dict_a = load_tabular_model(model_folder_a, cfg, verbose=True)
    binary, prob = run_tabular_model(model_dict_a, rec, str(tmp_data_dir), verbose=True)
    assert binary in (0, 1) and 0.0 <= prob <= 1.0
    print(f"  on-the-fly output for {rec[HEADERS['bids_folder']]}: binary={binary}, prob={prob:.4f}")

    # Part B — cache-first branch + metadata regression guard
    cfg.tabular.include_meta = True  # sub5 default is False; re-enable for the meta-fill regression
    cfg.tabular.feature_cache = str(cache_csv)
    model_folder_b = tmp_model_dir / "tabular_test_cache"
    train_tabular(cfg, model_folder_b, verbose=True)
    model_dict_b = load_tabular_model(model_folder_b, cfg, verbose=True)
    config = model_dict_b["config"]
    meta_cols = ["meta_age", "meta_sex_male", "meta_bmi", "meta_rec_year"]
    assert config["feature_list"][-len(meta_cols) :] == meta_cols, "metadata columns missing from the trained feature list"
    binary, prob = run_tabular_model(model_dict_b, rec, str(tmp_data_dir), verbose=True)
    assert binary in (0, 1) and 0.0 <= prob <= 1.0
    patient_data = load_demographics(
        str(tmp_data_dir / DEMOGRAPHICS_FILE),
        rec[HEADERS["bids_folder"]],
        rec[HEADERS["session_id"]],
    )
    assembled = assemble_record_features(pd.Series(dtype=float), patient_data, config)
    assert pd.notna(assembled["meta_age"]) and pd.notna(
        assembled["meta_sex_male"]
    ), "metadata columns are NaN at inference — demographics were not filled"
    print(f"  cache-first output for {rec[HEADERS['bids_folder']]}: binary={binary}, prob={prob:.4f}")
    print("test_tabular passed ✓")


@func_indicator("testing model components")
def test_model_components() -> None:
    """Component registry: manifest round-trip, e2e train/load/run, fallback chain."""
    from component_registry import MANIFEST_NAME, read_manifest, write_manifest
    from phi_component import PHI_SCORE_KEYS
    from tabular_pipeline import load_tabular_model, run_tabular_model
    from team_code import _run_components, load_model, run_model

    raw_dir = tmp_data_dir / "physiological_data"
    records = find_patients(str(tmp_data_dir / DEMOGRAPHICS_FILE))
    hits = [
        r
        for r in records
        if (raw_dir / r[HEADERS["site_id"]] / f"{r[HEADERS['bids_folder']]}_ses-{r[HEADERS['session_id']]}.edf").exists()
    ]
    assert hits, "no raw record found for the component test"
    rec = hits[0]

    # 1. manifest round-trip
    roundtrip_dir = tmp_model_dir / "manifest_roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        roundtrip_dir,
        [
            CFG(name="first", type="tabular", enable=True, priority=0),
            CFG(name="second", type="tabular", enable=False, priority=1),
        ],
    )
    manifest = read_manifest(roundtrip_dir)
    assert [c["name"] for c in manifest["components"]] == ["first", "second"]
    assert [c["enable"] for c in manifest["components"]] == [True, False]

    # 2. end-to-end through team_code.train_model / load_model / run_model.
    # The CI raw subset has no Phi latents baked in (the mounted /challenge/data
    # shadows the vendored cache), so write synthetic latents first; the phi
    # component then trains from cache and the tabular component still extracts
    # on the fly.  The real extraction path is covered by test_phi_inference.
    TrainCfg.phi.cache = str(tmp_model_dir / "phi_cache")
    rng = np.random.default_rng(2026)
    for r in records:
        site = str(r[HEADERS["site_id"]])
        key = f"{r[HEADERS['bids_folder']]}__{r[HEADERS['session_id']]}"
        out = tmp_model_dir / "phi_cache" / site / f"{key}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out,
            latent=rng.standard_normal(1024).astype(np.float32),
            **{k: float(rng.random()) for k in PHI_SCORE_KEYS},
        )

    comp_dir = tmp_model_dir / "components_e2e"
    old_n = TrainCfg.tabular.xgb_params.n_estimators
    old_phi_n = TrainCfg.phi.xgb_params.n_estimators
    try:
        TrainCfg.tabular.xgb_params.n_estimators = 30
        TrainCfg.phi.xgb_params.n_estimators = 200
        train_model(str(tmp_data_dir), str(comp_dir), True)
        assert (comp_dir / MANIFEST_NAME).is_file(), "component manifest not written by train_model"
        model_dict = load_model(str(comp_dir), True)
        assert set(model_dict["components"]) == {"phi_pca64", "sub5_tabular_xgb"}
        phi_config = model_dict["components"]["phi_pca64"]["config"]
        assert phi_config.get("features") == "fusion"
        assert len(phi_config.get("spec_feature_list") or []) > 0
        binary, prob = run_model(model_dict, rec, str(tmp_data_dir), True)
        assert binary in (0, 1) and 0.0 <= prob <= 1.0
        print(f"  component e2e output for {rec[HEADERS['bids_folder']]}: binary={binary}, prob={prob:.4f}")
    finally:
        TrainCfg.tabular.xgb_params.n_estimators = old_n
        TrainCfg.phi.xgb_params.n_estimators = old_phi_n

    # 3. per-record fallback chain: a failing primary falls through to the next
    good_payload = load_tabular_model(comp_dir / "components" / "sub5_tabular_xgb", TrainCfg, False)
    bad_payload = dict(good_payload)
    bad_payload["config"] = dict(good_payload["config"])
    bad_payload["config"]["constant"] = None  # force the booster predict path
    bad_payload["booster"] = None  # run_tabular_model raises on predict
    chain = {
        "components": {"primary_bad": bad_payload, "fallback_good": good_payload},
        "manifest": {
            "components": [
                {"name": "primary_bad", "type": "tabular", "priority": 0},
                {"name": "fallback_good", "type": "tabular", "priority": 1},
            ]
        },
    }
    binary_chain, prob_chain = _run_components(chain, rec, str(tmp_data_dir), False)
    binary_good, prob_good = run_tabular_model(good_payload, rec, str(tmp_data_dir), False)
    assert (binary_chain, prob_chain) == (binary_good, prob_good), "fallback chain did not recover the fallback output"

    # 4. all components failing raises (the run_model wrapper turns it into (0, 0.5))
    all_bad = {
        "components": {"bad": bad_payload},
        "manifest": {"components": [{"name": "bad", "type": "tabular", "priority": 0}]},
    }
    try:
        _run_components(all_bad, rec, str(tmp_data_dir), False)
        raise AssertionError("expected RuntimeError when every component fails")
    except RuntimeError:
        pass

    print("test_model_components passed ✓")


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2026_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        print("Test is skipped.")
        print("Please set CINC2026_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test:")
        print("CINC2026_REVENGER_TEST=1 python test_docker.py")
        print("Other environment variables:")
        print("  mount_data_dir: the data directory, e.g.")
        print("    CINC2026_REVENGER_TEST=1 mount_data_dir=/path/to/data python test_docker.py")
        exit(0)

    # CINC2026_REVENGER_TEST being set also activates strict error propagation
    # in team_code.py (_is_strict_test checks this same variable).

    print("#" * 100)
    print("testing team code")
    print("#" * 100)
    print(f"tmp_data_dir:   {str(tmp_data_dir)}")
    print(f"tmp_model_dir:  {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 100)

    test_dataset()
    test_models()
    test_challenge_metrics()
    test_spectral_features()
    test_phi_cache()
    test_phi_demographics_lookup()
    test_phi_inference()
    test_tabular()
    test_model_components()
    # test_trainer()  # passed, and overriden by test_entry
    test_entry()
