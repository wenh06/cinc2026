#!/usr/bin/env python
"""
CinC 2026 Challenge entry: train_model, load_model, run_model.

Primary model: EpochTransformer operating on CAISR epoch-feature sequences.
All physiological signals are summarised via CAISR (pre-computed by the
challenge organisers) into a fixed 21-dim feature vector per 30-second epoch,
so the representation is robust to inter-site signal heterogeneity.

Data-folder conventions
-----------------------
Both train_model.py and run_model.py pass a *partition* folder (e.g.
``training_set/`` or ``validation_set/``) as ``data_folder``.  The internal
:class:`data_reader.CINC2026` reader expects ``db_dir`` to be the *parent*
of the partition subfolders.  :func:`_resolve_db_dir` handles this mapping
transparently so that either convention works.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from cfg import ModelCfg, TrainCfg
from dataset import build_epoch_features
from helper_code import (
    DEMOGRAPHICS_FILE,
    HEADERS,
    load_demographics,
)
from models import EpochTransformer
from outputs import CINC2026Outputs
from trainer import CINC2026Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_MODEL_NAME = "final_model.pth.tar"


def _is_strict_test() -> bool:
    """Return True when CINC2026_REVENGER_STRICT_TEST is set to a truthy value.

    When strict-test mode is active, all ``except`` blocks in this module
    re-raise instead of swallowing errors.  This surfaces hidden bugs during
    CI (``test_docker.py`` sets the flag to ``"1"`` before running tests).
    In production (flag unset or ``"0"``) errors are caught and replaced with
    a safe fallback so that the challenge scorer always receives a valid
    ``(binary_output, probability_output)`` pair.
    """
    return os.environ.get("CINC2026_REVENGER_STRICT_TEST", "0") not in ("0", "", "false", "False", "no", "No")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_db_dir(data_folder: str) -> Path:
    """Return the CinC2026 db_dir (parent of partition subfolders).

    If *data_folder* is itself a partition (contains ``demographics.csv`` at
    its root), the parent directory is returned.  If it is already the data
    root (contains ``training_set/`` as a subdirectory), it is returned
    unchanged.
    """
    p = Path(data_folder).resolve()
    if (p / DEMOGRAPHICS_FILE).exists():
        # data_folder is a partition subfolder (e.g. training_set/)
        return p.parent
    # data_folder is already the root containing partition subdirectories
    return p


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
    """Train :class:`EpochTransformer` on CAISR features and save the checkpoint.

    Called by ``train_model.py``.  *data_folder* may be a partition subfolder
    (e.g. ``training_set/``) or the data root; both are handled correctly.
    """
    if verbose:
        print(f"[CinC2026] Training on {DEVICE} — data: {data_folder}")

    os.makedirs(model_folder, exist_ok=True)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = _resolve_db_dir(data_folder)

    # Allow fast debug / CI runs via env variable (0 = use TrainCfg default)
    debug_epochs = int(os.environ.get("CINC2026_REVENGER_TRAIN_EPOCHS", "0"))
    if debug_epochs > 0:
        train_config.n_epochs = debug_epochs

    # Route trainer logs and checkpoints inside model_folder
    working_dir = Path(model_folder) / "working_dir"
    working_dir.mkdir(parents=True, exist_ok=True)
    train_config.working_dir = working_dir
    train_config.model_dir = working_dir / "checkpoints"
    train_config.model_dir.mkdir(parents=True, exist_ok=True)
    train_config.log_dir = working_dir / "log"
    train_config.log_dir.mkdir(parents=True, exist_ok=True)
    train_config.debug = False

    model_config = deepcopy(ModelCfg.epoch_transformer)
    model = EpochTransformer(config=model_config)
    model.to(DEVICE)

    trainer = CINC2026Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=False,
    )
    trainer.train()

    # Save the final model checkpoint for load_model()
    save_path = Path(model_folder) / FINAL_MODEL_NAME
    model.save(str(save_path), train_config=train_config)

    if verbose:
        print(f"[CinC2026] Model saved to {save_path}")


def load_model(model_folder: str, verbose: bool) -> Dict[str, Any]:
    """Load the trained :class:`EpochTransformer` from *model_folder*.

    Called by ``run_model.py``.  Falls back to a randomly initialised model
    if the checkpoint file is not found (useful for dry runs).
    """
    if verbose:
        print("[CinC2026] Loading model ...")

    model_path = Path(model_folder) / FINAL_MODEL_NAME
    if model_path.exists():
        model, train_config = EpochTransformer.from_checkpoint(str(model_path), weights_only=False)
    else:
        if verbose:
            print(f"  Warning: {model_path} not found — using random weights.")
        model = EpochTransformer(config=ModelCfg.epoch_transformer)
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
    model: EpochTransformer = model_dict["model"]

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
    epoch_features = build_epoch_features(ann)

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
