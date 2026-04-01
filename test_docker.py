""" """

import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch_ecg.utils.misc import str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from dataset import CINC2026Dataset, collate_fn
from evaluate_model import evaluate_model as _evaluate_model
from evaluate_model import run as model_evaluator_func  # noqa: F401
from models import EpochTransformer
from run_model import run as model_runner_func  # noqa: F401
from team_code import _resolve_db_dir, load_model, run_model, train_model  # noqa: F401
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
    assert ef.ndim == 2, f"epoch_features should be 2-D, got shape {ef.shape}"
    assert ef.shape[1] == 21, f"epoch_features last dim should be 21, got {ef.shape[1]}"

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
    """Test EpochTransformer forward pass and inference."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    model_config = deepcopy(ModelCfg.epoch_transformer)
    model = EpochTransformer(config=model_config).to(DEVICE)
    model.eval()

    B, T, D = 2, 50, 21
    epoch_features = torch.randn(B, T, D, device=DEVICE)
    demographics = torch.randn(B, 3, device=DEVICE)
    padding_mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
    padding_mask[0, 40:] = True  # simulate a shorter record in slot 0

    # Forward pass
    input_dict = {
        "epoch_features": epoch_features,
        "demographics": demographics,
        "padding_mask": padding_mask,
    }
    with torch.no_grad():
        out = model(input_dict)

    assert "ci_logits" in out, "ci_logits missing from output"
    assert out["ci_logits"].shape == (B, 1), f"Expected ci_logits (B,1), got {out['ci_logits'].shape}"
    print(f"  Forward pass OK: ci_logits shape = {out['ci_logits'].shape}")

    # Inference (single sample, no padding)
    feat_np = np.random.randn(100, D).astype(np.float32)
    demo_np = np.array([0.65, 1.0, 0.5], dtype=np.float32)
    outputs = model.inference(epoch_features=feat_np, demographics=demo_np)
    assert outputs.ci_prob.shape == (1, 2), f"Expected ci_prob (1,2), got {outputs.ci_prob.shape}"
    prob = outputs.ci_prob[0, 1].item()
    assert 0.0 <= prob <= 1.0, f"Probability out of [0,1]: {prob}"
    print(f"  Inference OK: CI probability = {prob:.4f}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  EpochTransformer parameters: {n_params:,}")


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test evaluate_model with synthetic predictions."""
    rng = np.random.default_rng(42)
    n = 20
    patient_ids = [f"sub-{i:04d}" for i in range(n)]
    labels = rng.integers(0, 2, size=n)
    probs = rng.uniform(0.1, 0.9, size=n)
    binary_preds = (probs >= 0.5).astype(int)

    with tempfile.TemporaryDirectory() as tmpdir:
        labels_file = os.path.join(tmpdir, "labels.csv")
        preds_file = os.path.join(tmpdir, "predictions.csv")

        pd.DataFrame(
            {
                "BDSPPatientID": patient_ids,
                "Cognitive_Impairment": labels,
            }
        ).to_csv(labels_file, index=False)

        pd.DataFrame(
            {
                "BDSPPatientID": patient_ids,
                "Cognitive_Impairment": binary_preds,
                "Cognitive_Impairment_Probability": probs,
            }
        ).to_csv(preds_file, index=False)

        auroc, auprc, accuracy, f_measure = _evaluate_model(labels_file, preds_file)

    assert isinstance(auroc, float), f"auroc should be float, got {type(auroc)}"
    assert 0.0 <= auroc <= 1.0, f"auroc out of [0,1]: {auroc}"
    assert 0.0 <= auprc <= 1.0, f"auprc out of [0,1]: {auprc}"
    assert 0.0 <= accuracy <= 1.0, f"accuracy out of [0,1]: {accuracy}"
    assert 0.0 <= f_measure <= 1.0, f"f_measure out of [0,1]: {f_measure}"
    print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Acc={accuracy:.3f}  F1={f_measure:.3f}")


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test CINC2026Trainer for 1 epoch (quick smoke test)."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    from trainer import CINC2026Trainer

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = _resolve_db_dir(str(tmp_data_dir))
    train_config.n_epochs = 1
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "test_trainer_working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)
    train_config.model_dir = train_config.working_dir / "checkpoints"
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.log_dir = train_config.working_dir / "log"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)

    model_config = deepcopy(ModelCfg.epoch_transformer)
    model = EpochTransformer(config=model_config).to(DEVICE)

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
    """Full pipeline: train → load → run → evaluate."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    db_dir = _resolve_db_dir(str(tmp_data_dir))
    train_data_dir = db_dir / "training_set"
    val_data_dir = db_dir / "validation_set"

    if not train_data_dir.exists():
        # tmp_data_dir might itself be the training_set partition
        train_data_dir = tmp_data_dir
        val_data_dir = tmp_data_dir

    entry_model_dir = tmp_model_dir / "test_entry_model"
    entry_model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Train (1 epoch for speed)
    # ------------------------------------------------------------------
    print("   Run model training function   ".center(100, "#"))
    prev_epochs_env = os.environ.get("CINC2026_REVENGER_TRAIN_EPOCHS")
    os.environ["CINC2026_REVENGER_TRAIN_EPOCHS"] = "1"
    try:
        train_model(str(train_data_dir), str(entry_model_dir), verbose=True)
    finally:
        if prev_epochs_env is None:
            os.environ.pop("CINC2026_REVENGER_TRAIN_EPOCHS", None)
        else:
            os.environ["CINC2026_REVENGER_TRAIN_EPOCHS"] = prev_epochs_env

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    model_dict = load_model(str(entry_model_dir), verbose=True)
    assert "model" in model_dict

    # ------------------------------------------------------------------
    # 3. Run inference on validation set
    # ------------------------------------------------------------------
    print("   Run model inference function   ".center(100, "#"))
    from helper_code import DEMOGRAPHICS_FILE, find_patients

    val_demo_file = val_data_dir / DEMOGRAPHICS_FILE
    if not val_demo_file.exists():
        print(f"  Validation demographics not found at {val_demo_file}; skipping inference.")
        return

    records = find_patients(str(val_demo_file))
    print(f"  Found {len(records)} validation records.")

    all_results = {}
    for record in records[:5]:  # limit to 5 for speed
        bids_folder = record["BidsFolder"]
        binary_out, prob_out = run_model(model_dict, record, str(val_data_dir), verbose=True)
        all_results[bids_folder] = (binary_out, prob_out)
        print(f"    {bids_folder}: binary={binary_out}, prob={prob_out:.4f}")

    # ------------------------------------------------------------------
    # 4. Save predictions and evaluate
    # ------------------------------------------------------------------
    from helper_code import update_demographics_table

    output_table = update_demographics_table(str(val_demo_file), str(tmp_output_dir), all_results)
    print(f"  Predictions saved to: {output_table}")

    # Evaluate (only records we actually predicted on)
    preds_df = pd.read_csv(output_table)
    labels_df = pd.read_csv(str(val_demo_file))
    predicted_bids = list(all_results.keys())
    preds_sub = preds_df[preds_df["BidsFolder"].isin(predicted_bids)]
    labels_sub = labels_df[labels_df["BidsFolder"].isin(predicted_bids)]

    if len(labels_sub) >= 2 and labels_sub["Cognitive_Impairment"].nunique() > 1:
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_file = os.path.join(tmpdir, "labels.csv")
            preds_file = os.path.join(tmpdir, "preds.csv")
            labels_sub.to_csv(labels_file, index=False)
            preds_sub.to_csv(preds_file, index=False)
            auroc, auprc, accuracy, f_measure = _evaluate_model(labels_file, preds_file)
        print(f"  Eval on {len(labels_sub)} records: AUROC={auroc:.3f} AUPRC={auprc:.3f}")
    else:
        print("  Skipping metric evaluation (too few / homogeneous labels in subset).")


# Allow test_entry to be referenced as test_team_code for compatibility
test_team_code = test_entry


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
    test_trainer()
    test_entry()
