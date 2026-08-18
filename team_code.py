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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from pyedflib import EdfReader

from cfg import ModelCfg, TrainCfg, sync_feature_config
from component_registry import (
    ARTIFACT_NAMES,
    COMPONENT_TYPES,
    component_dir,
    enabled_components,
    read_manifest,
    resolve_feature_cache,
    write_manifest,
)
from const import BINARY_AROUSAL_FEATURE_SET, resolve_feature_pipeline
from dataset import CINC2026Dataset, build_epoch_features, build_night_features, normalize_epoch_features
from helper_code import (
    DEMOGRAPHICS_FILE,
    HEADERS,
    load_demographics,
)
from models import EpochCRNN, EpochTransformer
from phi_component import load_phi_model, resolve_phi_cache, run_phi_model, train_phi_model
from tabular_pipeline import load_tabular_model, run_tabular_model, tabular_enabled, train_tabular
from utils.scoring_metrics import tune_binary_threshold

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
from trainer import CINC2026Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINAL_MODEL_NAME = ARTIFACT_NAMES["crnn"]


def _is_strict_test() -> bool:
    """Return True when CINC2026_REVENGER_TEST is set to a truthy value.

    When active, all ``except`` blocks re-raise instead of swallowing errors,
    surfacing hidden bugs during CI.  In production (flag unset) errors are
    caught and replaced with a safe fallback ``(0, 0.5)`` so the challenge
    scorer always receives a valid prediction.
    """
    return os.environ.get("CINC2026_REVENGER_TEST", "0") not in (
        "0",
        "",
        "false",
        "False",
        "no",
        "No",
    )


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

    Two training modes, controlled by ``TrainCfg.folds``:

    * ``None`` (default) — train a single model on the canonical split and
      save it to ``model_folder/final_model.pth.tar``.
    * list of fold indices (e.g. ``[0, 1, 2, 3, 4]``) — 5-fold CV ensemble:
      each fold trains on its own train split (``train_config.fold = k``,
      see :class:`CINC2026Dataset`) and is saved to
      ``model_folder/fold_{k}/final_model.pth.tar``.  :func:`load_model`
      detects this layout and loads all folds for averaged inference.
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

    resolve_feature_cache(train_config)
    resolve_phi_cache(train_config)
    components = enabled_components(train_config)
    if components:
        for comp in components:
            _train_component(comp, train_config, Path(model_folder), verbose)
        write_manifest(Path(model_folder), components)
        if verbose:
            names = ", ".join(str(c.get("name")) for c in components)
            print(f"[CinC2026] trained components: {names}")
        return

    if tabular_enabled(train_config):
        train_tabular(train_config, Path(model_folder), verbose)
        return

    folds = train_config.get("folds", None)
    if folds is None:
        _train_single_fold(train_config, Path(model_folder), verbose)
        return

    # 5-fold ensemble mode: one model per fold, each on its own data split
    if verbose:
        print(f"[CinC2026] 5-fold ensemble mode — folds: {list(folds)}")
    for k in folds:
        fold_config = deepcopy(train_config)
        fold_config.fold = k
        _train_single_fold(fold_config, Path(model_folder) / f"fold_{k}", verbose)
        if verbose:
            print(f"[CinC2026] fold_{k} saved")


def _train_component(comp: Any, train_config: Any, model_folder: Path, verbose: bool) -> None:
    """Train one component into ``model_folder/components/<name>/``."""
    name = str(comp.get("name"))
    ctype = str(comp.get("type"))
    out_dir = component_dir(model_folder, name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[CinC2026] training component '{name}' ({ctype})")
    if ctype == "tabular":
        train_tabular(train_config, out_dir, verbose)
    elif ctype == "crnn":
        folds = train_config.get("folds", None)
        if folds is None:
            _train_single_fold(train_config, out_dir, verbose)
            return
        if verbose:
            print(f"[CinC2026] 5-fold ensemble mode — folds: {list(folds)}")
        for k in folds:
            fold_config = deepcopy(train_config)
            fold_config.fold = k
            _train_single_fold(fold_config, out_dir / f"fold_{k}", verbose)
            if verbose:
                print(f"[CinC2026] fold_{k} saved")
    elif ctype == "phi":
        train_phi_model(train_config, out_dir, verbose)
    else:
        raise ValueError(f"unknown component type: {ctype!r} (expected one of {COMPONENT_TYPES})")


def _train_single_fold(train_config: Any, out_folder: Path, verbose: bool) -> None:
    """Train one model (single fold, or the canonical split) into *out_folder*."""
    out_folder.mkdir(parents=True, exist_ok=True)
    sync_feature_config(train_config, ModelCfg)

    # Route trainer logs and checkpoints inside the output folder
    working_dir = out_folder / "working_dir"
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

    # Bridge TrainCfg.focal → model criterion (O7: focal loss on top of BCE).
    # Must run after the pos_weight bridge above — criterion_kw then carries
    # both pos_weight (imbalance) and gamma (easy-sample down-weighting).
    # Merge rather than assign blindly: criterion_kw only exists when
    # pos_weight was set, so focal-on-without-pos_weight must not KeyError.
    focal_cfg = train_config.get("focal", None)
    if focal_cfg is not None and focal_cfg.get("enable", False):
        model_config.criterion = "FocalBCEWithLogitsLoss"
        model_config.criterion_kw = dict(model_config.get("criterion_kw") or {})
        model_config.criterion_kw["gamma"] = float(focal_cfg.get("gamma", 2.0))

    # Bridge TrainCfg.age_pairwise → model config (age-matched pairwise loss)
    ap_cfg = train_config.get("age_pairwise", None)
    if ap_cfg is not None:
        model_config.age_pairwise = deepcopy(ap_cfg)

    # Bridge TrainCfg.age_adv → model config (age-adversarial branch toggle)
    age_adv_cfg = train_config.get("age_adv", None)
    if age_adv_cfg is not None:
        model_config.age_adv = deepcopy(age_adv_cfg)

    # Bridge TrainCfg.night_features → model config (night-level aggregation branch)
    night_cfg = train_config.get("night_features", None)
    if night_cfg is not None:
        model_config.night_features = deepcopy(night_cfg)
    # Bridge TrainCfg.no_age → model config (O4: drop the age channel from FiLM)
    model_config.no_age = bool(train_config.get("no_age", False))
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

    # Tune the binary threshold on the best checkpoint over the val split and
    # persist it in the *model* config (serialised into the checkpoint, so
    # run_model reproduces it).  Full-night features (max_seq_len=None) match
    # the run_model inference path.  Affects Reward/Accuracy/F1 only.
    tune_cfg = deepcopy(train_config)
    tune_cfg.lazy = False
    tune_cfg.max_seq_len = None
    tune_ds = CINC2026Dataset(tune_cfg, training=False)
    best_thr = tune_binary_threshold(model, tune_ds, DEVICE)
    model.config.binary_threshold = best_thr
    model_config.binary_threshold = best_thr  # keep the local copy in sync

    save_path = out_folder / FINAL_MODEL_NAME
    model.save(str(save_path), train_config=train_config)

    if verbose:
        print(f"[CinC2026] Model saved to {save_path}")


def load_model(model_folder: str, verbose: bool) -> Dict[str, Any]:
    """Load the trained model(s) from *model_folder*.

    The model class is inferred from ``TrainCfg.model_name``.
    Called by ``run_model.py``.  Falls back to a randomly initialised model
    if the checkpoint file is not found (useful for dry runs).

    Three layouts are supported:

    * component manifest — ``model_folder/model_manifest.json`` written by
      :func:`train_model` when ``TrainCfg.components`` is configured, returned
      as ``{"components": {name: payload}, "manifest": ...}``; the routing
      does NOT depend on the current ``TrainCfg``;
    * single model — ``model_folder/final_model.pth.tar``, returned as
      ``{"model": ..., "train_config": ...}``;
    * 5-fold ensemble — ``model_folder/fold_{k}/final_model.pth.tar``
      (trained with ``TrainCfg.folds`` set), returned as
      ``{"models": [...], "train_config": ..., "ensemble": True}``.
      :func:`run_model` averages the fold probabilities.
    * legacy tabular — ``{"tabular": ...}`` when ``TrainCfg.tabular.enable``
      is set and no manifest is present.
    """
    if verbose:
        print("[CinC2026] Loading model ...")

    folder = Path(model_folder)
    manifest = read_manifest(folder)
    if manifest is not None:
        load_cfg = deepcopy(TrainCfg)
        resolve_feature_cache(load_cfg)
        resolve_phi_cache(load_cfg)
        components = {}
        for comp in manifest["components"]:
            name = str(comp["name"])
            if verbose:
                print(f"  Loading component '{name}' ({comp.get('type')})")
            components[name] = _load_component(comp, folder, load_cfg, verbose)
        return {"components": components, "manifest": manifest}

    if tabular_enabled(TrainCfg):
        load_cfg = deepcopy(TrainCfg)
        resolve_feature_cache(load_cfg)
        return {"tabular": load_tabular_model(folder, load_cfg, verbose)}

    model_name = TrainCfg.model_name
    model_cls = _MODEL_CLASS_MAP[model_name]

    # 5-fold ensemble layout: model_folder/fold_{k}/final_model.pth.tar
    fold_dirs = sorted(
        Path(model_folder).glob("fold_*/" + FINAL_MODEL_NAME),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    if fold_dirs:
        if verbose:
            print(f"  Loading {len(fold_dirs)}-fold ensemble: {[str(p) for p in fold_dirs]}")
        models = []
        train_configs = []
        for ckpt in fold_dirs:
            model, tc = model_cls.from_checkpoint(str(ckpt), weights_only=False)
            model.to(DEVICE)
            model.eval()
            models.append(model)
            train_configs.append(tc)
        return {"models": models, "train_config": train_configs[0], "ensemble": True}

    model_path = folder / FINAL_MODEL_NAME
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


def _load_component(comp: Dict[str, Any], model_folder: Path, train_config: Any, verbose: bool) -> Any:
    """Load one component's payload from ``model_folder/components/<name>/``."""
    name = str(comp.get("name"))
    ctype = str(comp.get("type"))
    out_dir = component_dir(model_folder, name)
    if ctype == "tabular":
        return load_tabular_model(out_dir, train_config, verbose)
    if ctype == "crnn":
        model_name = train_config.model_name
        model_cls = _MODEL_CLASS_MAP[model_name]
        fold_dirs = sorted(
            out_dir.glob("fold_*/" + FINAL_MODEL_NAME),
            key=lambda p: int(p.parent.name.split("_")[1]),
        )
        if fold_dirs:
            if verbose:
                print(f"  Loading {len(fold_dirs)}-fold ensemble: {[str(p) for p in fold_dirs]}")
            models = []
            train_configs = []
            for ckpt in fold_dirs:
                model, tc = model_cls.from_checkpoint(str(ckpt), weights_only=False)
                model.to(DEVICE)
                model.eval()
                models.append(model)
                train_configs.append(tc)
            return {"models": models, "train_config": train_configs[0], "ensemble": True}
        model_path = out_dir / FINAL_MODEL_NAME
        if model_path.exists():
            model, tc = model_cls.from_checkpoint(str(model_path), weights_only=False)
        else:
            if verbose:
                print(f"  Warning: {model_path} not found — using random weights.")
            model = model_cls(config=getattr(ModelCfg, model_name))
            tc = train_config
        model.to(DEVICE)
        model.eval()
        return {"model": model, "train_config": tc}
    if ctype == "phi":
        return load_phi_model(out_dir, train_config, verbose)
    raise ValueError(f"unknown component type: {ctype!r} (expected one of {COMPONENT_TYPES})")


@torch.no_grad()
def _sliding_window_inference(
    model: Any,
    epoch_features: np.ndarray,
    demographics: np.ndarray,
    night_features: Optional[np.ndarray],
    window: int,
    stride: int,
) -> Tuple[int, float]:
    """Infer one record over overlapping windows; return ``(binary, probability)``.

    Training crops long nights to ``max_seq_len`` (768) but the inference
    features here are the full night, so a full-length forward sees sequences
    the model never trained on.  Covering the night with overlapping windows of
    the training length and averaging the per-window probabilities removes the
    mismatch (measured +0.013 age-cond on fold_0 val vs full-night inference).
    """
    T = len(epoch_features)
    if T <= window:
        out = model.inference(
            epoch_features=epoch_features,
            demographics=demographics,
            night_features=night_features,
        )
        return int(out.cognitive_impairment[0]), float(out.ci_prob[0, 1])

    starts = list(range(0, T - window + 1, stride))
    starts.append(T - window)  # include the tail window
    win_probs = []
    for s in starts:
        w = epoch_features[s : s + window]
        pad = window - len(w)
        if pad > 0:
            w = np.pad(w, ((0, pad), (0, 0)), mode="constant")
            padding_mask = np.concatenate([np.zeros(window - pad, dtype=bool), np.ones(pad, dtype=bool)])
        else:
            padding_mask = None
        out = model.inference(
            epoch_features=w,
            demographics=demographics,
            padding_mask=padding_mask,
            night_features=night_features,
        )
        win_probs.append(float(out.ci_prob[0, 1]))

    p = float(np.mean(win_probs))
    threshold = float(getattr(model.config, "binary_threshold", 0.5))
    return int(p >= threshold), p


def _run_component(
    ctype: str,
    payload: Any,
    record: Dict[str, str],
    data_folder: str,
    verbose: bool,
) -> Tuple[int, float]:
    """Run one component payload; component types may raise → fallback chain."""
    if ctype == "tabular":
        return run_tabular_model(payload, record, data_folder, verbose)
    if ctype == "crnn":
        return _run_crnn_model(payload, record, data_folder, verbose)
    if ctype == "phi":
        return run_phi_model(payload, record, data_folder, verbose)
    raise NotImplementedError(f"component type {ctype!r} cannot run yet")


def _run_components(
    model_dict: Dict[str, Any],
    record: Dict[str, str],
    data_folder: str,
    verbose: bool,
) -> Tuple[int, float]:
    """Run the component chain in priority order; failures fall through.

    Each record goes to the highest-priority component that succeeds.  If a
    component raises (e.g. a Phi ranker facing a montage without C4), the
    next component in priority order gets the record — the sub5 tabular XGB
    is the terminal fallback because its features are montage-agnostic.
    """
    ordered = sorted(
        model_dict["manifest"]["components"],
        key=lambda c: (int(c.get("priority", 0)), str(c.get("name", ""))),
    )
    last_exc: Optional[Exception] = None
    for comp in ordered:
        name = str(comp["name"])
        ctype = str(comp["type"])
        try:
            return _run_component(ctype, model_dict["components"][name], record, data_folder, verbose)
        except Exception as exc:  # noqa: BLE001 — per-record fallback to the next component
            last_exc = exc
            if verbose:
                print(f"  [component '{name}'] failed on {record}: {exc!r}; falling through")
    raise RuntimeError(f"all components failed on {record}: {last_exc!r}")


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
    if model_dict.get("components", None) is not None:
        return _run_components(model_dict, record, data_folder, verbose)
    if model_dict.get("tabular", None) is not None:
        return run_tabular_model(model_dict["tabular"], record, data_folder, verbose)
    return _run_crnn_model(model_dict, record, data_folder, verbose)


def _run_crnn_model(
    model_dict: Dict[str, Any],
    record: Dict[str, str],
    data_folder: str,
    verbose: bool,
) -> Tuple[int, float]:
    """CRNN inference for one record (single model or 5-fold ensemble)."""
    model: Any = model_dict.get("model", None)

    bids_folder = str(record[HEADERS["bids_folder"]])
    site_id = str(record[HEADERS["site_id"]])
    # Keep the original SessionID type for load_demographics: demographics.csv
    # stores it as an int column, so find_patients returns an int; casting to
    # str here silently makes the mask never match and every demographics
    # vector falls back to the defaults (measured: ~0.09 AUROC / ~0.10 age-cond
    # loss on the O5 fold_0 val).  The filename still needs the string form.
    session_id_raw = record[HEADERS["session_id"]]
    session_id = str(session_id_raw)

    # ------------------------------------------------------------------
    # 1. Demographics
    # ------------------------------------------------------------------
    demo_file = Path(data_folder) / DEMOGRAPHICS_FILE
    patient_data = load_demographics(str(demo_file), bids_folder, session_id_raw)
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

    # Night-level aggregation features (P1, Phase 9) — computed from the full
    # annotation dict; the model only consumes them when its night branch is
    # enabled (old O0 checkpoints load and run untouched).
    night_features = build_night_features(ann)

    # ------------------------------------------------------------------
    # 3. Inference
    # ------------------------------------------------------------------
    # Sliding-window inference: mirror the training-time max_seq_len crop by
    # covering the full night with overlapping windows of that length (window
    # / stride = 768 / 384), averaging the per-window probabilities.  Short
    # nights run whole.  Window length follows the training config so a
    # future change of the crop length propagates automatically.
    window = getattr(train_config, "max_seq_len", None) if train_config is not None else None
    window = int(window) if window else 768
    stride = window // 2

    if model is not None:
        # Single-model path
        return _sliding_window_inference(
            model,
            epoch_features,
            demographics,
            night_features,
            window=window,
            stride=stride,
        )

    # 5-fold ensemble path — the probability is the equal-weight average of
    # the fold probabilities (used by the AUROC-family metrics), while the
    # binary prediction is a majority vote of the per-fold binaries, each
    # produced with its own tuned threshold from its checkpoint config.
    # The folds share the same feature pipeline, so the per-record features
    # above are computed once and reused.
    models: Any = model_dict["models"]
    results = [
        _sliding_window_inference(
            m,
            epoch_features,
            demographics,
            night_features,
            window=window,
            stride=stride,
        )
        for m in models
    ]
    probability_output = float(np.mean([p for _, p in results]))
    binary_output = int(sum(b for b, _ in results) > len(results) // 2)
    return binary_output, probability_output
