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
]


PROJECT_DIR = str(Path(__file__).resolve().parent)


CHANNEL_TABLE = pd.read_csv(Path(PROJECT_DIR) / "channel_table.csv")


SLEEP_STAGE_MAPPING = {
    "N3": 1,
    "N2": 2,
    "N1": 3,
    "REM": 4,
    "W": 5,
    "Unavailable": 9,
}


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
