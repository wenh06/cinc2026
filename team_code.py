#!/usr/bin/env python
"""
CinC 2026 Challenge entry: train_model, load_model, run_model.

Primary model and CAISR feature pipeline are configurable via ``TrainCfg``.
The default reproduces the best unofficial submission
(``epoch_crnn_M`` + the binary-arousal 21-dim CAISR feature set), while the
later arousal-probability-statistics pipeline remains available through
``TrainCfg.feature_set``.

All physiological signals are summarised via CAISR (pre-computed by the
challenge organisers) into a compact per-epoch feature vector, so the
representation is robust to inter-site signal heterogeneity.

Data-folder conventions
-----------------------
Both train_model.py and run_model.py pass a *partition* folder (e.g.
``training_set/`` or ``validation_set/``) as ``data_folder``.  The internal
:class:`data_reader.CINC2026` reader expects ``db_dir`` to be the *parent*
of the partition subfolders.  :func:`_resolve_db_dir` handles this mapping
transparently so that either convention works.

Environment variables
---------------------
This module reads two environment variables:

``CINC2026_OVERRIDE_JSON``
    Path to a JSON file whose key-value pairs override the corresponding
    ``TrainCfg`` attributes before training starts.  Used by the
    hyperparameter search script (``utils/run_search.py``) and by
    ``test_docker.py`` for CI-specific settings (e.g. ``n_epochs``,
    ``batch_size``).  Leave unset for a normal submission run.

``CINC2026_REVENGER_TEST``
    Set to ``"1"`` (or any truthy string) to activate test/CI mode:
    all ``except`` blocks re-raise instead of returning a safe fallback,
    and ``test_docker.py`` injects CI-friendly training settings
    (short run, small batch).  Set automatically by ``test_docker.py``.
    Leave unset for submission.

Typical usage per mode
----------------------
* **Submission** (PhysioNet evaluator): no env vars — cfg.py drives everything.
* **Search** (local): ``CINC2026_OVERRIDE_JSON=/path/exp.json python train_model.py …``
* **CI** (GitHub Actions / local test): ``CINC2026_REVENGER_TEST=1`` triggers strict
  error reporting; ``test_docker.py`` injects ``{"n_epochs": 3, "batch_size": 4}``
  via ``CINC2026_OVERRIDE_JSON``.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from cfg import ModelCfg, TrainCfg, sync_feature_config
from const import BINARY_AROUSAL_FEATURE_SET, resolve_feature_pipeline
from dataset import build_epoch_features, normalize_epoch_features
from helper_code import (
    DEMOGRAPHICS_FILE,
    HEADERS,
    load_demographics,
)
from models import EpochCRNN, EpochTransformer

# Map TrainCfg.model_name → model class.  Both plain names ("epoch_transformer")
# and size-suffixed names ("epoch_transformer_M", "epoch_crnn_L") are handled.
_MODEL_CLASS_MAP: Dict[str, Any] = {
    "epoch_transformer": EpochTransformer,
    "epoch_transformer_S": EpochTransformer,
    "epoch_transformer_M": EpochTransformer,
    "epoch_transformer_L": EpochTransformer,
    "epoch_crnn": EpochCRNN,
    "epoch_crnn_S": EpochCRNN,
    "epoch_crnn_M": EpochCRNN,
    "epoch_crnn_L": EpochCRNN,
    # resnetNC_BNse backbone (4-stage bottleneck + SE)
    "epoch_crnn_resnetNC_BNse_S": EpochCRNN,
    "epoch_crnn_resnetNC_BNse_M": EpochCRNN,
    "epoch_crnn_resnetNC_BNse_L": EpochCRNN,
    # tresnetE backbone (4-stage mixed basic+bottleneck+SE, TResNet-style)
    "epoch_crnn_tresnetE_S": EpochCRNN,
    "epoch_crnn_tresnetE_M": EpochCRNN,
    "epoch_crnn_tresnetE_L": EpochCRNN,
}
from outputs import CINC2026Outputs
from trainer import CINC2026Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_MODEL_NAME = "final_model.pth.tar"


def _is_strict_test() -> bool:
    """Return True when CINC2026_REVENGER_TEST is set to a truthy value.

    When active, all ``except`` blocks re-raise instead of swallowing errors,
    surfacing hidden bugs during CI.  In production (flag unset) errors are
    caught and replaced with a safe fallback ``(0, 0.5)`` so the challenge
    scorer always receives a valid prediction.
    """
    return os.environ.get("CINC2026_REVENGER_TEST", "0") not in ("0", "", "false", "False", "no", "No")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_db_dir(data_folder: str) -> Path:
    """Return the resolved db_dir path to pass to :class:`CINC2026`.

    :class:`CINC2026._ls_rec` handles two layouts automatically:

    * **Standard (nested)** — db_dir contains ``training_set/`` subdirectory.
    * **Flat** — db_dir itself contains ``demographics.csv`` at its root
      (the layout used by the PhysioNet challenge evaluator).

    This function therefore simply resolves the path without any parent-hopping;
    the reader discovers the correct layout on its own.
    """
    return Path(data_folder).resolve()


def _load_caisr_ann(caisr_path: str) -> Dict[str, np.ndarray]:
    """Load CAISR annotation signals from an EDF file into a label→array dict."""
    try:
        from pyedflib import EdfReader

        reader = EdfReader(caisr_path)
        annotations: Dict[str, np.ndarray] = {}
        for i, label in enumerate(reader.getSignalLabels()):
            annotations[label.lower().strip()] = reader.readSignal(i)
        reader._close()
        return annotations
    except Exception:
        if _is_strict_test():
            raise
        return {}


def _extract_demographics(patient_data: Dict) -> np.ndarray:
    """Return normalised ``[age/100, is_male, bmi/50]`` from a demographics dict.

    Mirrors the logic in :meth:`FastDataReader._extract_demographics` so that
    training and inference see identical demographic vectors:

    * **age** : ``age_years / 100.0``.  Defaults to ``0.6`` (≈ 60 years) when
      missing or non-numeric.
    * **sex** : ``1.0`` if the raw string starts with ``'m'``, ``0.0`` otherwise.
    * **bmi** : ``bmi_kg_m2 / 50.0``.  Defaults to ``0.5`` (≈ 25 kg/m²) when
      missing or non-numeric.
    """
    age_raw = patient_data.get(HEADERS["age"])
    try:
        age_f = float(age_raw)
        age = age_f / 100.0 if not np.isnan(age_f) else 0.6
    except (TypeError, ValueError):
        age = 0.6

    sex_raw = str(patient_data.get(HEADERS["sex"], "")).strip().lower()
    sex = 1.0 if sex_raw.startswith("m") else 0.0

    bmi_raw = patient_data.get(HEADERS["bmi"])
    try:
        bmi_f = float(bmi_raw)
        bmi = bmi_f / 50.0 if not np.isnan(bmi_f) else 0.5
    except (TypeError, ValueError):
        bmi = 0.5

    return np.array([age, sex, bmi], dtype=np.float32)


# ---------------------------------------------------------------------------
# Challenge entry points
# ---------------------------------------------------------------------------


def train_model(data_folder: str, model_folder: str, verbose: bool) -> None:
    """Train the model selected by ``TrainCfg.model_name`` and save the checkpoint.

    Called by ``train_model.py``.  *data_folder* may be a partition subfolder
    (e.g. ``training_set/``) or the data root; both are handled correctly.
    Change the active model by setting ``TrainCfg.model_name`` in ``cfg.py``
    (e.g. ``"epoch_crnn_M"`` or ``"epoch_transformer_L"``).
    """
    if verbose:
        print(f"[CinC2026] Training on {DEVICE} — data: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = _resolve_db_dir(data_folder)

    # Apply config overrides from a JSON file (used by search script and CI).
    # Key-value pairs override the corresponding TrainCfg attributes.
    # Example: {"model_name": "epoch_crnn_M", "n_epochs": 3, "batch_size": 4}
    override_json = os.environ.get("CINC2026_OVERRIDE_JSON", "")
    if override_json and Path(override_json).exists():
        with open(override_json) as f:
            overrides = json.load(f)
        _skip = {"db_dir", "model_folder"}
        for k, v in overrides.items():
            if k not in _skip:
                setattr(train_config, k, v)
    sync_feature_config(train_config, ModelCfg)

    # Route trainer logs and checkpoints inside model_folder
    working_dir = Path(model_folder) / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    train_config.working_dir = working_dir
    train_config.model_dir = working_dir / "checkpoints"
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.log_dir = working_dir / "log"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    train_config.debug = False

    model_name = train_config.model_name
    model_config = deepcopy(getattr(ModelCfg, model_name))

    # Bridge TrainCfg.pos_weight → model criterion (BCEWithLogitsLoss).
    # Official phase CI prevalence is 7.6% — without this, the loss is overwhelmed
    # by negatives and the model can degenerate into an all-negative predictor.
    pw = train_config.get("pos_weight", None)
    if pw is not None:
        pw_tensor = torch.tensor([float(pw)], device=DEVICE, dtype=torch.float32)
        model_config.criterion_kw = {"pos_weight": pw_tensor}

    # Bridge TrainCfg.age_adv → model config (age-adversarial branch toggle)
    age_adv_cfg = train_config.get("age_adv", None)
    if age_adv_cfg is not None:
        model_config.age_adv = deepcopy(age_adv_cfg)
    model_cls = _MODEL_CLASS_MAP[model_name]
    model = model_cls(config=model_config)
    model.to(DEVICE)

    trainer = CINC2026Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )
    best_state_dict = trainer.train()

    # Save the best model checkpoint for load_model().  trainer.train()
    # returns the state dict of the best epoch (per `monitor`); load it into
    # the model so that `model.save()` (which serialises the *current* weights)
    # writes the best-epoch weights, not the final-epoch ones.
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    save_path = Path(model_folder) / FINAL_MODEL_NAME
    model.save(str(save_path), train_config=train_config)

    if verbose:
        print(f"[CinC2026] Model saved to {save_path}")


def load_model(model_folder: str, verbose: bool) -> Dict[str, Any]:
    """Load the trained model from *model_folder*.

    The model class is inferred from ``TrainCfg.model_name``.
    Called by ``run_model.py``.  Falls back to a randomly initialised model
    if the checkpoint file is not found (useful for dry runs).
    """
    if verbose:
        print("[CinC2026] Loading model ...")

    model_name = TrainCfg.model_name
    model_cls = _MODEL_CLASS_MAP[model_name]

    model_path = Path(model_folder) / FINAL_MODEL_NAME
    if model_path.exists():
        model, train_config = model_cls.from_checkpoint(str(model_path), weights_only=False)
    else:
        if verbose:
            print(f"  Warning: {model_path} not found — using random weights.")
        model = model_cls(config=getattr(ModelCfg, model_name))
        train_config = TrainCfg

    model.to(DEVICE)
    model.eval()
    return {"model": model, "train_config": train_config}


@torch.no_grad()
def run_model(
    model_dict: Dict[str, Any],
    record: Dict[str, str],
    data_folder: str,
    verbose: bool,
) -> Tuple[int, float]:
    """Run inference for one record and return ``(binary_prediction, probability)``.

    Called by ``run_model.py``.

    Parameters
    ----------
    model_dict:
        Output of :func:`load_model` — contains ``"model"`` and ``"train_config"``.
    record:
        One entry from :func:`helper_code.find_patients`.
        Expected keys: ``BidsFolder``, ``SiteID``, ``SessionID``.
    data_folder:
        Path to the data partition folder (e.g. ``validation_set/`` or
        ``test_set/``).  Must contain ``demographics.csv`` and
        ``algorithmic_annotations/`` at its root.
    verbose:
        Print progress messages when ``True``.

    Returns
    -------
    binary_output : int
        ``1`` if cognitive impairment is predicted, ``0`` otherwise.
        Falls back to ``0`` on any unhandled error (unless strict-test mode is
        active — see :func:`_is_strict_test`).
    probability_output : float
        Positive-class probability in ``[0, 1]``.  Falls back to ``0.5``.
    """
    try:
        return _run_model_impl(model_dict, record, data_folder, verbose)
    except Exception as exc:
        if _is_strict_test():
            raise
        if verbose:
            print(f"  [run_model] ERROR for record {record}: {exc!r}; returning fallback (0, 0.5).")
        return 0, 0.5


def _run_model_impl(
    model_dict: Dict[str, Any],
    record: Dict[str, str],
    data_folder: str,
    verbose: bool,
) -> Tuple[int, float]:
    """Inner implementation of :func:`run_model` (may raise)."""
    model: Any = model_dict["model"]

    bids_folder = str(record[HEADERS["bids_folder"]])
    site_id = str(record[HEADERS["site_id"]])
    session_id = str(record[HEADERS["session_id"]])

    # ------------------------------------------------------------------
    # 1. Demographics
    # ------------------------------------------------------------------
    demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
    patient_data = load_demographics(str(demo_file), bids_folder, session_id)
    demographics = _extract_demographics(patient_data)

    # ------------------------------------------------------------------
    # 2. CAISR annotations → epoch feature matrix
    # ------------------------------------------------------------------
    caisr_filename = f"{bids_folder}_ses-{session_id}_caisr_annotations.edf"
    caisr_path = Path(data_folder) / "algorithmic_annotations" / site_id / caisr_filename

    if not caisr_path.exists():
        if verbose:
            print(f"  CAISR not found for {bids_folder}; returning fallback (0, 0.5).")
        return 0, 0.5

    ann = _load_caisr_ann(str(caisr_path))
    train_config = model_dict.get("train_config", None)
    if train_config is not None:
        feature_set = getattr(train_config, "feature_set", BINARY_AROUSAL_FEATURE_SET)
        include_time = getattr(train_config, "include_time_encoding", None)
        if include_time is None:
            include_time = resolve_feature_pipeline(feature_set, getattr(train_config, "model_name", ""))[
                "include_time_encoding"
            ]
    else:
        feature_set = BINARY_AROUSAL_FEATURE_SET
        include_time = True
    epoch_features = build_epoch_features(
        ann,
        feature_set=feature_set,
        include_time_encoding=include_time,
    )

    # Apply the same per-record normalization used during training
    norm_cfg = getattr(train_config, "normalize", None) if train_config is not None else None
    if norm_cfg and getattr(norm_cfg, "method", "") == "per_record_zscore":
        epoch_features = normalize_epoch_features(epoch_features, norm_cfg)

    if len(epoch_features) == 0:
        if verbose:
            print(f"  Empty CAISR features for {bids_folder}; returning fallback.")
        return 0, 0.5

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    outputs: CINC2026Outputs = model.inference(
        epoch_features=epoch_features,
        demographics=demographics,
    )

    binary_output = int(outputs.cognitive_impairment[0])
    probability_output = float(outputs.ci_prob[0, 1])
    return binary_output, probability_output
