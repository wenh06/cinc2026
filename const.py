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
    "REMOTE_MODELS",
    "CHANNEL_TABLE",
    "SLEEP_STAGE_MAPPING",
    "STAGE_LABEL_TO_IDX",
    "STAGE_ONEHOT_DIM",
    "CAISR_EPOCH_DIM",
    "CAISR_PROB_EDF_SCALE",
    "DEMOGRAPHIC_DIM",
    "AROUSAL_SAMPLES_PER_EPOCH",
    "RESP_SAMPLES_PER_EPOCH",
    "LIMB_SAMPLES_PER_EPOCH",
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

# Per-epoch CAISR feature layout (total = 21 dims):
#   [0:6]   stage one-hot          STAGE_ONEHOT_DIM = 6
#   [6:11]  stage softmax probs    5  (n3, n2, n1, r, w; normalized to [0,1])
#   [11]    arousal fraction       1
#   [12:17] resp event fractions   5  (OA, CA, MA, HY, RERA)
#   [17:19] limb event fractions   2  (isolated, periodic)
#   [19]    sin(2π*t/T)            1
#   [20]    cos(2π*t/T)            1
CAISR_EPOCH_DIM = 21

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

TEST_DATA_CACHE_DIR = str(Path(DATA_CACHE_DIR).parent / "revenger_action_test_data_dir")
Path(TEST_DATA_CACHE_DIR).mkdir(parents=True, exist_ok=True)


REMOTE_MODELS = {}
