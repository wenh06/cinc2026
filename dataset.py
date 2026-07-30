"""Dataset for the CinC 2026 Challenge.

Each sample is one patient's full-night PSG, represented as a sequence of
30-second epochs.  Each epoch is encoded as a fixed-size CAISR feature vector
(channel-agnostic, identical across all recording sites) so the dataset is
robust to the signal heterogeneity described in _CINC2026_INFO.

Two feature sets are supported:

Binary-arousal set (default; used by unofficial submissions 1-4, 21 dims for all models):
  [0:6]   stage one-hot
  [6:11]  stage softmax probs
  [11]    arousal_fraction       mean of binary arousal_caisr over the epoch
  [12:17] resp event fractions
  [17:19] limb event fractions
  [19:21] sin/cos time-position encoding

Arousal-probability-statistics set (experimental; used by unofficial submission 5):
  [0:6]   stage one-hot
  [6:11]  stage softmax probs
  [11:14] arousal prob mean/std/max from caisr_prob_arous
  [14:19] resp event fractions
  [19:21] limb event fractions
  [21:23] optional sin/cos time-position encoding (Transformer only)

Note on caisr_prob_* scaling: the EDF physical-range header for the stage
probability channels was set to [0, 9] instead of [0, 1].  pyedflib faithfully
returns the physical values, so we divide by CAISR_PROB_EDF_SCALE (= 9) and
then re-normalise each row so the five probabilities sum to 1.
"""

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG, DEFAULTS
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_nn import default_collate_fn
from tqdm.auto import tqdm

from cfg import TrainCfg
from const import (  # noqa: F401
    AROUSAL_PROB_STATS_CAISR_EPOCH_DIM,
    AROUSAL_PROB_STATS_CAISR_EPOCH_DIM_NO_TIME,
    AROUSAL_PROB_STATS_FEATURE_SET,
    AROUSAL_SAMPLES_PER_EPOCH,
    BINARY_AROUSAL_CAISR_EPOCH_DIM,
    BINARY_AROUSAL_FEATURE_SET,
    CAISR_EPOCH_DIM,
    CAISR_EPOCH_DIM_NO_TIME,
    CAISR_PROB_EDF_SCALE,
    DEMOGRAPHIC_DIM,
    FIXED_DATA_SPLIT_FILE,
    LABEL_CACHE_DIR,
    LIMB_SAMPLES_PER_EPOCH,
    RESP_SAMPLES_PER_EPOCH,
    STAGE_LABEL_TO_IDX,
    STAGE_ONEHOT_DIM,
    resolve_feature_pipeline,
)
from data_reader import CINC2026

__all__ = [
    "CINC2026Dataset",
    "FastDataReader",
    "collate_fn",
    "build_epoch_features",
    "normalize_epoch_features",
]

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class CINC2026Dataset(Dataset, ReprMixin):
    """Outer dataset for the CinC 2026 Challenge.

    Manages the train/validation split, optional in-memory caching, and
    delegates per-record loading to :class:`FastDataReader`.

    Parameters
    ----------
    config : CFG
        Training configuration.  ``config.db_dir`` must point to the data root.
    training : bool, default True
        Whether this instance represents the training or validation split.
    lazy : bool, default True
        If ``False``, all records are loaded into memory at construction time.
    override_data_split : bool, default False
        If ``False`` (default), the fixed canonical split shipped at
        ``utils/cinc2026-data-split.json`` is used, ensuring fully
        reproducible train/val assignments across runs and machines.
        If ``True``, the historical dynamic flow is used: read
        ``LABEL_CACHE_DIR/cinc2026-data-split.json`` when it exists,
        otherwise generate a fresh stratified split and save it there.
    reader_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`CINC2026`.
    """

    __name__ = "CINC2026Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = True,
        override_data_split: bool = False,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.config["training"] = training  # propagate so FastDataReader.training works correctly
        self.lazy = lazy
        self.override_data_split = override_data_split

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
        else:
            reader_kwargs.pop("db_dir", None)
        assert self.config.db_dir is not None, "db_dir must be specified"
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        self.dtype = np.float32 if self.config.torch_dtype != torch.float64 else np.float64

        self.reader = CINC2026(db_dir=self.config.db_dir, **reader_kwargs)

        # Only use the labelled training partition(s) for the train/val split.
        # Official phase: "training_set_small" / "training_set_large";
        # unofficial phase: "training_set".  All three contain labels.
        # The supplementary_set (I0004, I0007 examples) has no labels and
        # is kept aside for inspection / domain-adaptation experiments.
        _train_parts = {"training_set", "training_set_small", "training_set_large"}
        self._labelled_df = self.reader._df_records[self.reader._df_records["partition"].isin(_train_parts)].copy()

        self.records = self._train_test_split()
        self.fdr = FastDataReader(self.reader, self.records, self.config)

        self.__cache: Optional[Dict] = None
        if not self.lazy:
            self._load_all_data()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.__cache is None:
            return len(self.fdr)
        return len(self.__cache["label"])

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        if self.__cache is None:
            return self.fdr[index]  # type: ignore
        return {k: v[index] for k, v in self.__cache.items()}

    # ------------------------------------------------------------------
    # Train / validation split
    # ------------------------------------------------------------------

    def _train_test_split(self) -> List[str]:
        """Return the record list for this split.

        Two modes controlled by ``self.override_data_split``:

        * ``False`` (default) — load the fixed canonical split from
          ``utils/cinc2026-data-split.json``.  Falls back to the dynamic
          flow only when the file is absent (should not happen in a normal
          installation).
        * ``True`` — dynamic flow: read
          ``LABEL_CACHE_DIR/cinc2026-data-split.json`` when present, else
          generate a fresh stratified split and persist it there.
        """
        part = "train" if self.training else "val"
        available = set(self._labelled_df.index)

        if not self.override_data_split:
            # ----------------------------------------------------------
            # Default path: use the repo-shipped canonical split
            # ----------------------------------------------------------
            fixed_file = Path(FIXED_DATA_SPLIT_FILE)
            if fixed_file.exists():
                with open(fixed_file) as f:
                    split = json.load(f)
                records = [r for r in split.get(part, []) if r in available]
                if records:
                    if self.training:
                        DEFAULTS.RNG.shuffle(records)
                    return records
            # Canonical file missing — warn and fall through to dynamic path
            import warnings

            warnings.warn(
                f"Fixed data-split file not found at {FIXED_DATA_SPLIT_FILE}. " "Falling back to dynamic split generation.",
                RuntimeWarning,
                stacklevel=3,
            )

        # ------------------------------------------------------------------
        # Dynamic path (override_data_split=True, or fallback from above)
        # ------------------------------------------------------------------
        split_file = Path(LABEL_CACHE_DIR) / "cinc2026-data-split.json"

        if split_file.exists():
            with open(split_file) as f:
                split = json.load(f)
            records = [r for r in split.get(part, []) if r in available]
            if records:
                if self.training:
                    DEFAULTS.RNG.shuffle(records)
                return records

        # Generate a fresh stratified split and cache it
        from sklearn.model_selection import StratifiedShuffleSplit

        df = self._labelled_df
        strat_key = df["SiteID"].astype(str) + "_" + df["Cognitive_Impairment"].astype(str)
        train_ratio = self.config.get("train_ratio", 0.8)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
        train_idx, val_idx = next(sss.split(df.index, strat_key))

        train_records = df.index[train_idx].tolist()
        val_records = df.index[val_idx].tolist()

        with open(split_file, "w") as f:
            json.dump({"train": train_records, "val": val_records}, f, indent=2)

        records = train_records if self.training else val_records
        if self.training:
            DEFAULTS.RNG.shuffle(records)
        return records

    # ------------------------------------------------------------------
    # Optional in-memory cache
    # ------------------------------------------------------------------

    def _load_all_data(self, batch_size: int = 16, num_workers: Optional[int] = None) -> None:
        """Load all records into RAM using a DataLoader for parallelism.

        .. warning::
            The cache stores a Python list of per-record tensors because
            variable-length sequences cannot be stacked into a single tensor.
            Use ``lazy=True`` (the default) during normal training.
        """
        if num_workers is None:
            cpu_count = os.cpu_count() or 4
            num_workers = min(max(2, int(cpu_count * 0.5)), 8)

        loader = DataLoader(
            self.fdr,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

        all_batches = []
        start = time.time()
        for batch in tqdm(loader, desc="Caching data", dynamic_ncols=True):
            all_batches.append(batch)

        self.__cache = _merge_batches(all_batches)
        elapsed = time.time() - start
        print(f"Cached {len(self)} records in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_records(self, new_records: List[str], reload: bool = False) -> None:
        """Replace the active record list (e.g. for cross-validation folds)."""
        all_available = set(self._labelled_df.index)
        self.records = [r for r in new_records if r in all_available]
        del self.fdr
        self.fdr = FastDataReader(self.reader, self.records, self.config)
        self.__cache = None
        if reload:
            self._load_all_data()

    @property
    def cache(self) -> Optional[Dict]:
        return self.__cache

    @property
    def data_fields(self) -> Set[str]:
        return {"epoch_features", "demographics", "label", "padding_mask", "n_epochs", "record_id", "site_id"}

    @property
    def labels(self) -> np.ndarray:
        """Binary labels for all records in this split (for class-weight computation)."""
        return self._labelled_df.loc[self.records, "Cognitive_Impairment"].astype(int).values

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]


# ---------------------------------------------------------------------------
# FastDataReader — inner per-record loader
# ---------------------------------------------------------------------------


class FastDataReader(Dataset, ReprMixin):
    """Per-record data loader used internally by :class:`CINC2026Dataset`.

    :meth:`__getitem__` loads CAISR annotations for one record and builds the
    epoch-feature matrix.  Raw physiological signals are NOT loaded here;
    all features are derived exclusively from the CAISR annotations so the
    representation is site-agnostic.

    Returns
    -------
    dict with keys:

    ``epoch_features`` : np.ndarray, shape ``(N_epochs, CAISR_EPOCH_DIM)``
    ``demographics``   : np.ndarray, shape ``(DEMOGRAPHIC_DIM,)``
    ``label``          : np.int64   (0 = no CI, 1 = CI)
    ``n_epochs``       : np.int64   actual sequence length before padding
    """

    __name__ = "FastDataReader"

    def __init__(
        self,
        reader: CINC2026,
        records: Sequence[str],
        config: CFG,
    ) -> None:
        self.reader = reader
        self.records = list(records)
        self.config = config
        self.dtype = np.float32 if config.torch_dtype != torch.float64 else np.float64

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: Union[int, List[int], slice]) -> Dict[str, np.ndarray]:
        if isinstance(index, slice):
            return default_collate_fn([self[i] for i in range(*index.indices(len(self)))])  # type: ignore
        if isinstance(index, list):
            return default_collate_fn([self[i] for i in index])  # type: ignore

        rec = self.records[index]

        # Load CAISR (algorithmic) annotations — available for all splits
        ann = self.reader.load_ann(rec, ann_type="algorithmic")
        feature_set = self.config.get("feature_set", BINARY_AROUSAL_FEATURE_SET)
        include_time = self.config.get("include_time_encoding", None)
        if include_time is None:
            include_time = resolve_feature_pipeline(feature_set, self.config.get("model_name", ""))["include_time_encoding"]

        epoch_features = build_epoch_features(
            ann,
            dtype=self.dtype,
            feature_set=feature_set,
            include_time_encoding=include_time,
        )

        norm_cfg = self.config.get("normalize", None)
        if norm_cfg and getattr(norm_cfg, "method", "") == "per_record_zscore":
            epoch_features = normalize_epoch_features(epoch_features, norm_cfg)

        # Optionally crop to max_seq_len
        max_len = self.config.get("max_seq_len", None)
        n = len(epoch_features)
        if max_len and n > max_len:
            if self.training:
                start = int(DEFAULTS.RNG.integers(0, n - max_len + 1))
            else:
                start = (n - max_len) // 2
            epoch_features = epoch_features[start : start + max_len]

        demographics = self._extract_demographics(rec)

        row = self.reader._df_records.loc[rec]
        label = int(bool(row.get("Cognitive_Impairment", False)))

        return {  # type: ignore
            "epoch_features": epoch_features,  # (N, CAISR_EPOCH_DIM)
            "demographics": demographics,  # (DEMOGRAPHIC_DIM,)
            "label": np.int64(label),
            "n_epochs": np.int64(len(epoch_features)),
            "record_id": rec,
            "site_id": str(row.get("SiteID", "")),
        }

    def _extract_demographics(self, rec: str) -> np.ndarray:
        """Return a normalised [age, sex, bmi] vector."""
        row = self.reader._df_records.loc[rec]

        age_raw = row.get("Age", 60)
        age = float(age_raw) / 100.0 if pd.notna(age_raw) else 0.6  # type: ignore

        sex_raw = str(row.get("Sex", "")).strip().lower()
        sex = 1.0 if sex_raw.startswith("m") else 0.0

        bmi_raw = row.get("BMI", 25.0)
        bmi = float(bmi_raw) / 50.0 if pd.notna(bmi_raw) else 0.5  # type: ignore

        return np.array([age, sex, bmi], dtype=self.dtype)

    @property
    def training(self) -> bool:
        return self.config.get("training", True)

    def extra_repr_keys(self) -> List[str]:
        return ["records"]


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------


def build_epoch_features(
    ann: Dict[str, np.ndarray],
    dtype: type = np.float32,
    feature_set: str = BINARY_AROUSAL_FEATURE_SET,
    include_time_encoding: bool = True,
) -> np.ndarray:
    """Build a per-epoch feature matrix from CAISR annotations.

    Parameters
    ----------
    ann : dict
        Annotation dict from ``CINC2026.load_ann(rec, ann_type='algorithmic')``.
    dtype : numpy dtype
    feature_set : {"binary_arousal", "arousal_prob_stats"}, default "binary_arousal"
        ``"binary_arousal"`` reproduces the 21-dim feature layout used by
        unofficial submissions 1-4.  ``"arousal_prob_stats"`` uses the later
        arousal-probability statistics and model-dependent time encoding from
        submission 5.
    include_time_encoding : bool, default True
        For ``feature_set="binary_arousal"``, must be True because this
        21-dim feature definition always includes sin/cos time-position
        encoding.  For ``feature_set="arousal_prob_stats"``, controls whether
        the trailing sin/cos
        encoding is appended.

    Returns
    -------
    np.ndarray
        Returns empty array with matching second dim when ``stage_caisr`` is absent.
    """
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        if not include_time_encoding:
            raise ValueError("The binary-arousal CAISR feature set always includes time-position encoding.")
        feat_dim = BINARY_AROUSAL_CAISR_EPOCH_DIM
    elif feature_set == AROUSAL_PROB_STATS_FEATURE_SET:
        feat_dim = AROUSAL_PROB_STATS_CAISR_EPOCH_DIM if include_time_encoding else AROUSAL_PROB_STATS_CAISR_EPOCH_DIM_NO_TIME
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    stage = ann.get("stage_caisr", np.array([]))
    n_epochs = len(stage)

    if n_epochs == 0:
        return np.zeros((0, feat_dim), dtype=dtype)

    full_feat_dim = (
        BINARY_AROUSAL_CAISR_EPOCH_DIM if feature_set == BINARY_AROUSAL_FEATURE_SET else AROUSAL_PROB_STATS_CAISR_EPOCH_DIM
    )
    features = np.zeros((n_epochs, full_feat_dim), dtype=dtype)
    stage_int = stage.astype(int)

    # [0:6] Stage one-hot (N3, N2, N1, REM, W, Unknown)
    for ep_i, s in enumerate(stage_int):
        features[ep_i, STAGE_LABEL_TO_IDX.get(s, STAGE_ONEHOT_DIM - 1)] = 1.0

    # [6:11] Stage softmax probs — correct for EDF physical-range scaling bug
    prob_keys = [
        "caisr_prob_n3",
        "caisr_prob_n2",
        "caisr_prob_n1",
        "caisr_prob_r",
        "caisr_prob_w",
    ]
    any_prob_loaded = False
    for j, key in enumerate(prob_keys):
        if key in ann and len(ann[key]) == n_epochs:
            features[:, 6 + j] = (ann[key] / CAISR_PROB_EDF_SCALE).astype(dtype)
            any_prob_loaded = True

    if any_prob_loaded:
        prob_sum = features[:, 6:11].sum(axis=1, keepdims=True)
        valid = prob_sum[:, 0] > 1e-6
        features[valid, 6:11] /= prob_sum[valid]
        # Unavailable epochs (prob_sum≈0): fall back to the one-hot as hard dist
        features[~valid, 6:11] = features[~valid, :5]
    else:
        # No prob channels in this file; use stage one-hot as a hard distribution
        features[:, 6:11] = features[:, :5]

    arousal_caisr = ann.get("arousal_caisr")
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        # [11] Binary-arousal feature: fraction of 0.5 s arousal-positive samples.
        if arousal_caisr is not None and len(arousal_caisr) == n_epochs * AROUSAL_SAMPLES_PER_EPOCH:
            features[:, 11] = arousal_caisr.reshape(n_epochs, AROUSAL_SAMPLES_PER_EPOCH).mean(axis=1)
    else:
        # [11:14] Arousal-probability-statistics from caisr_prob_arous (2 Hz, 60 samples/epoch).
        prob_arous = ann.get("caisr_prob_arous")
        if prob_arous is not None and len(prob_arous) == n_epochs * AROUSAL_SAMPLES_PER_EPOCH:
            ar_mat = prob_arous.reshape(n_epochs, AROUSAL_SAMPLES_PER_EPOCH).astype(np.float64)
        elif arousal_caisr is not None and len(arousal_caisr) == n_epochs * AROUSAL_SAMPLES_PER_EPOCH:
            ar_mat = arousal_caisr.reshape(n_epochs, AROUSAL_SAMPLES_PER_EPOCH).astype(np.float64)
        else:
            ar_mat = None

        if ar_mat is not None:
            features[:, 11] = ar_mat.mean(axis=1)
            features[:, 12] = ar_mat.std(axis=1)
            features[:, 13] = ar_mat.max(axis=1)

    # Respiratory events: [12:17] in binary-arousal, [14:19] in arousal-prob-stats
    resp_offset = 12 if feature_set == BINARY_AROUSAL_FEATURE_SET else 14
    resp = ann.get("resp_caisr")
    if resp is not None and len(resp) == n_epochs * RESP_SAMPLES_PER_EPOCH:
        resp_mat = resp.reshape(n_epochs, RESP_SAMPLES_PER_EPOCH)
        for offset, cls_val in enumerate([1, 2, 3, 4, 5]):
            features[:, resp_offset + offset] = (resp_mat == cls_val).mean(axis=1)

    # Limb events: [17:19] in binary-arousal, [19:21] in arousal-prob-stats
    limb_offset = 17 if feature_set == BINARY_AROUSAL_FEATURE_SET else 19
    limb = ann.get("limb_caisr")
    if limb is not None and len(limb) == n_epochs * LIMB_SAMPLES_PER_EPOCH:
        limb_mat = limb.reshape(n_epochs, LIMB_SAMPLES_PER_EPOCH)
        features[:, limb_offset] = (limb_mat == 1).mean(axis=1)
        features[:, limb_offset + 1] = (limb_mat == 2).mean(axis=1)

    # Time-position encoding: [19:21] in binary-arousal, [21:23] in arousal-prob-stats
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        t = np.linspace(0.0, 1.0, n_epochs, dtype=dtype)
        features[:, 19] = np.sin(2 * np.pi * t)
        features[:, 20] = np.cos(2 * np.pi * t)
        return features.astype(dtype)

    if include_time_encoding:
        t = np.linspace(0.0, 1.0, n_epochs, dtype=dtype)
        features[:, 21] = np.sin(2 * np.pi * t)
        features[:, 22] = np.cos(2 * np.pi * t)
        return features.astype(dtype)

    return features[:, :AROUSAL_PROB_STATS_CAISR_EPOCH_DIM_NO_TIME].astype(dtype)


def normalize_epoch_features(
    features: np.ndarray,
    cfg=None,
) -> np.ndarray:
    """Per-record z-score normalization of a CAISR epoch feature matrix.

    Each column is independently centered and scaled using the mean and std
    computed across **all epochs of that record**.  This removes systematic
    site-level baseline differences without requiring global training-set
    statistics, which makes it safe to apply identically at both training
    and inference time.

    Columns 19-20 (sin/cos time-position encoding) carry absolute positional
    meaning and must **not** be rescaled.  Any other column whose std is
    below ``eps`` (e.g. an all-zero respiratory event column) is left unchanged
    to avoid numerical explosion.

    Parameters
    ----------
    features : np.ndarray, shape (N_epochs, CAISR_EPOCH_DIM)
    cfg : CFG or None
        Optional config object that may supply ``eps`` and ``skip_cols``.
        Defaults: ``eps=1e-8``, ``skip_cols=[19, 20]``.

    Returns
    -------
    np.ndarray, same shape and dtype as *features*
    """
    if features.shape[0] < 2:
        return features

    eps = float(getattr(cfg, "eps", 1e-8)) if cfg is not None else 1e-8
    skip = set(getattr(cfg, "skip_cols", [21, 22]) if cfg is not None else [21, 22])

    out = features.copy()
    for j in range(features.shape[1]):
        if j in skip:
            continue
        col = features[:, j]
        sigma = col.std()
        if sigma > eps:
            out[:, j] = (col - col.mean()) / sigma
    return out


# ---------------------------------------------------------------------------
# Collate function for variable-length sequences
# ---------------------------------------------------------------------------


def collate_fn(
    batch: List[Dict[str, np.ndarray]],
) -> Dict[str, torch.Tensor]:
    """Collate records with variable-length epoch sequences into a padded batch.

    Parameters
    ----------
    batch : list of dicts
        Each dict is the output of :meth:`FastDataReader.__getitem__`.

    Returns
    -------
    dict
    ``epoch_features`` : FloatTensor ``(B, T_max, CAISR_EPOCH_DIM)``  zero-padded
    ``demographics``   : FloatTensor ``(B, DEMOGRAPHIC_DIM)``
    ``label``          : LongTensor  ``(B,)``
    ``padding_mask``   : BoolTensor  ``(B, T_max)``  True = ignored padding position
    ``n_epochs``       : LongTensor  ``(B,)``        actual (unpadded) lengths
    """
    n_epochs_list = [int(item["n_epochs"]) for item in batch]
    t_max = max(n_epochs_list)
    feat_dim = batch[0]["epoch_features"].shape[-1]
    b = len(batch)

    epoch_features = np.zeros((b, t_max, feat_dim), dtype=np.float32)
    padding_mask = np.ones((b, t_max), dtype=bool)  # True = padding

    for i, (item, n) in enumerate(zip(batch, n_epochs_list)):
        epoch_features[i, :n] = item["epoch_features"][:n]
        padding_mask[i, :n] = False  # False = valid position

    demographics = np.stack([item["demographics"] for item in batch]).astype(np.float32)
    labels = np.array([int(item["label"]) for item in batch], dtype=np.int64)

    return {  # type: ignore
        "epoch_features": torch.from_numpy(epoch_features),  # (B, T, D)
        "demographics": torch.from_numpy(demographics),  # (B, D_demo)
        "label": torch.from_numpy(labels),  # (B,)
        "padding_mask": torch.from_numpy(padding_mask),  # (B, T) bool
        "n_epochs": torch.tensor(n_epochs_list, dtype=torch.int64),
        "record_id": [item["record_id"] for item in batch],  # list[str]
        "site_id": [item["site_id"] for item in batch],  # list[str]
    }


# ---------------------------------------------------------------------------
# Internal helper: merge pre-collated batches into a flat cache
# ---------------------------------------------------------------------------


def _merge_batches(
    batches: List[Dict[str, torch.Tensor]],
) -> Dict:
    """Concatenate a list of collated batches.

    ``epoch_features`` and ``padding_mask`` are stored as lists of individual
    (unpadded) per-record tensors because sequence lengths differ.
    """
    result: Dict[str, list] = {k: [] for k in batches[0]}
    for batch in batches:
        n_list = batch["n_epochs"].tolist()
        for k, v in batch.items():
            if k in ("epoch_features", "padding_mask"):
                for i, n in enumerate(n_list):
                    result[k].append(v[i, :n])
            elif k == "n_epochs":
                result[k].extend(n_list)
            elif k in ("record_id", "site_id"):
                result[k].extend(v)  # v is already a list of strings
            else:
                result[k].append(v)

    return {
        "epoch_features": result["epoch_features"],
        "padding_mask": result["padding_mask"],
        "demographics": torch.cat(result["demographics"], dim=0),
        "label": torch.cat(result["label"], dim=0),
        "n_epochs": torch.tensor(result["n_epochs"], dtype=torch.int64),
        "record_id": result["record_id"],
        "site_id": result["site_id"],
    }
