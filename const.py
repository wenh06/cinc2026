"""Constants for the project."""

import os
from pathlib import Path

import pandas as pd

__all__ = [
    "PROJECT_DIR",
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "LABEL_CACHE_DIR",
    "TEST_DATA_CACHE_DIR",
    "FIXED_DATA_SPLIT_FILE",
    "REMOTE_MODELS",
    "CHANNEL_TABLE",
    "SLEEP_STAGE_MAPPING",
    "STAGE_LABEL_TO_IDX",
    "STAGE_ONEHOT_DIM",
    "BINARY_AROUSAL_FEATURE_SET",
    "AROUSAL_PROB_STATS_FEATURE_SET",
    "CAISR_EPOCH_DIM",
    "CAISR_EPOCH_DIM_NO_TIME",
    "LEGACY_CAISR_EPOCH_DIM",
    "ENRICHED_CAISR_EPOCH_DIM",
    "ENRICHED_CAISR_EPOCH_DIM_NO_TIME",
    "CAISR_PROB_EDF_SCALE",
    "DEMOGRAPHIC_DIM",
    "AROUSAL_SAMPLES_PER_EPOCH",
    "RESP_SAMPLES_PER_EPOCH",
    "LIMB_SAMPLES_PER_EPOCH",
    "get_caisr_feature_dim",
    "get_caisr_time_cols",
    "resolve_feature_pipeline",
]


PROJECT_DIR = str(Path(__file__).resolve().parent)


CHANNEL_TABLE = pd.read_csv(Path(PROJECT_DIR) / "channel_table.csv")


# Standardized channel names and their modality groupings
# This serves as the 'vocabulary' for ChannelTransformer (IDs)
# and the branch assignment for MultiBranchNet
STANDARD_CHANNELS = [
    "EEG F3-M2",
    "EEG F4-M1",
    "EEG C3-M2",
    "EEG C4-M1",
    "EEG O1-M2",
    "EEG O2-M1",
    "EOG E1-M2",
    "EOG E2-M1",
    "EMG CHIN",
    "ECG",
    "RESP ABD",
    "RESP CHEST",
    "RESP AIRFLOW",
    "RESP PTAF",
    "RESP SPO2",
]

CHANNEL_ID_MAP = {name: i for i, name in enumerate(STANDARD_CHANNELS)}

MODALITY_MAP = {
    "EEG": ["EEG F3-M2", "EEG F4-M1", "EEG C3-M2", "EEG C4-M1", "EEG O1-M2", "EEG O2-M1"],
    "EOG": ["EOG E1-M2", "EOG E2-M1"],
    "EMG": ["EMG CHIN"],
    "ECG": ["ECG"],
    "RESP": ["RESP ABD", "RESP CHEST", "RESP AIRFLOW", "RESP PTAF", "RESP SPO2"],
}

SLEEP_STAGE_MAPPING = {
    "N3": 1,
    "N2": 2,
    "N1": 3,
    "REM": 4,
    "W": 5,
    "Unavailable": 9,
}

# CAISR annotation sampling resolutions relative to a 30s epoch
AROUSAL_SAMPLES_PER_EPOCH = 60  # arousal_caisr at 2Hz (0.5s resolution)
RESP_SAMPLES_PER_EPOCH = 30  # resp_caisr at 1Hz
LIMB_SAMPLES_PER_EPOCH = 30  # limb_caisr at 1Hz

# stage label → one-hot index (0-based)
# Indices: N3=0, N2=1, N1=2, REM=3, W=4, Unknown=5
STAGE_LABEL_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 9: 5}
STAGE_ONEHOT_DIM = 6  # N3, N2, N1, REM, W, Unknown

# Supported CAISR epoch-feature pipelines.
BINARY_AROUSAL_FEATURE_SET = "binary_arousal"
AROUSAL_PROB_STATS_FEATURE_SET = "arousal_prob_stats"

# Per-epoch CAISR feature layout — binary-arousal pipeline used by unofficial submissions 1-4:
#
# Legacy layout (21 dims, used by all models in submissions 1-4):
#   [0:6]   stage one-hot          STAGE_ONEHOT_DIM = 6
#   [6:11]  stage softmax probs    5  (n3, n2, n1, r, w; normalized to [0,1])
#   [11]    arousal_fraction       1  mean of binary arousal_caisr over 60 sub-epoch samples
#   [12:17] resp event fractions   5  (OA, CA, MA, HY, RERA)
#   [17:19] limb event fractions   2  (isolated, periodic)
#   [19]    sin(2π*t/T)            1  time-position encoding
#   [20]    cos(2π*t/T)            1  time-position encoding
LEGACY_CAISR_EPOCH_DIM = 21

# Enriched layout (23 dims for Transformer, 21 dims for CRNN):
#   [0:6]   stage one-hot          STAGE_ONEHOT_DIM = 6
#   [6:11]  stage softmax probs    5  (n3, n2, n1, r, w; normalized to [0,1])
#   [11]    arousal_prob_mean      1  mean of caisr_prob_arous over 60 sub-epoch samples
#   [12]    arousal_prob_std       1  std  of caisr_prob_arous
#   [13]    arousal_prob_max       1  max  of caisr_prob_arous
#   [14:19] resp event fractions   5  (OA, CA, MA, HY, RERA)
#   [19:21] limb event fractions   2  (isolated, periodic)
#   [21]    sin(2π*t/T)            1  time-position encoding
#   [22]    cos(2π*t/T)            1  time-position encoding
ENRICHED_CAISR_EPOCH_DIM = 23
ENRICHED_CAISR_EPOCH_DIM_NO_TIME = 21

# Backwards-compatible aliases used throughout the project.
CAISR_EPOCH_DIM = LEGACY_CAISR_EPOCH_DIM
CAISR_EPOCH_DIM_NO_TIME = ENRICHED_CAISR_EPOCH_DIM_NO_TIME

# EDF physical-range scaling bug: the prob channels are stored with
# physical_max=9 instead of 1, causing pyedflib to scale up by 9×.
CAISR_PROB_EDF_SCALE = 9.0

# Demographic feature dimension (Age, Sex, BMI)
DEMOGRAPHIC_DIM = 3


MODEL_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_model_dir_cinc2026
        # /challenge/cache/revenger_model_dir
        os.environ.get("MODEL_CACHE_DIR", "~/.cache/cinc2026/revenger_model_dir")
    )
    .expanduser()
    .resolve()
)
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)


DATA_CACHE_DIR = str(
    Path(
        # ~/.cache/revenger_data_dir_cinc2026
        # /challenge/cache/revenger_data_dir
        os.environ.get("DATA_CACHE_DIR", "~/.cache/cinc2026/revenger_data_dir")
    )
    .expanduser()
    .resolve()
)
Path(DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)

LABEL_CACHE_DIR = str(Path(PROJECT_DIR) / "cache")
Path(LABEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Canonical (fixed) train/val split shipped with the repo.  Used by default
# so that training runs are reproducible without regenerating the split.
FIXED_DATA_SPLIT_FILE = str(Path(PROJECT_DIR) / "utils" / "cinc2026-data-split.json")

TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR).parent / "revenger_action_test_data_dir")
Path(TEST_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


REMOTE_MODELS = {}


def get_caisr_feature_dim(feature_set: str, include_time_encoding: bool = True) -> int:
    """Return the epoch-feature dimension for a feature-set / time-encoding choice."""
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        return LEGACY_CAISR_EPOCH_DIM
    if feature_set == AROUSAL_PROB_STATS_FEATURE_SET:
        return ENRICHED_CAISR_EPOCH_DIM if include_time_encoding else ENRICHED_CAISR_EPOCH_DIM_NO_TIME
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def get_caisr_time_cols(feature_set: str, include_time_encoding: bool = True) -> list[int]:
    """Return columns that encode absolute time position and should not be z-scored."""
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        return [19, 20]
    if feature_set == AROUSAL_PROB_STATS_FEATURE_SET:
        return [21, 22] if include_time_encoding else []
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def resolve_feature_pipeline(feature_set: str, model_name: str) -> dict:
    """Resolve model-dependent CAISR feature settings from a high-level feature set."""
    is_crnn = "crnn" in model_name.lower()
    if feature_set == BINARY_AROUSAL_FEATURE_SET:
        include_time_encoding = True
    elif feature_set == AROUSAL_PROB_STATS_FEATURE_SET:
        include_time_encoding = not is_crnn
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    return {
        "feature_set": feature_set,
        "include_time_encoding": include_time_encoding,
        "feature_dim": get_caisr_feature_dim(feature_set, include_time_encoding),
        "time_cols": get_caisr_time_cols(feature_set, include_time_encoding),
    }
