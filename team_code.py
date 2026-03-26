#!/usr/bin/env python

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from cfg import ModelCfg, TrainCfg
from const import CHANNEL_ID_MAP, STANDARD_CHANNELS
from data_reader import CINC2026
from helper_code import DEMOGRAPHICS_FILE
from models import ChannelTransformer
from outputs import CINC2026Outputs
from trainer import CINC2026Trainer

# NOTE: constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = np.float32

# Path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_MODEL_NAME = "final_model.pth.tar"


def train_model(data_folder: str, model_folder: str, verbose: bool):
    """
    Entry point for training the model. Called by train_model.py.
    """
    if verbose:
        print(f"Initializing CinC 2026 Training on {DEVICE}...")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # 1. Setup configurations
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = Path(data_folder).resolve()
    train_config.model_dir = Path(model_folder).resolve()

    # 2. Initialize Model (Default to ChannelTransformer)
    model_config = deepcopy(ModelCfg.transformer)
    model = ChannelTransformer(config=model_config)
    model.to(DEVICE)

    # 3. Initialize Trainer
    trainer = CINC2026Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )

    # 4. Start Training
    trainer.train()

    # 5. Save final model using torch_ecg save method
    save_path = os.path.join(model_folder, FINAL_MODEL_NAME)
    model.save(save_path, train_config=train_config)

    if verbose:
        print(f"Training complete. Model saved to {save_path}")


def load_model(model_folder: str, verbose: bool) -> Dict[str, Any]:
    """
    Entry point for loading the model. Called by run_model.py.
    """
    if verbose:
        print("Loading CinC 2026 Model...")

    model_path = os.path.join(model_folder, FINAL_MODEL_NAME)

    # Check if the model file exists, fallback to default if not (for dry runs)
    if not os.path.exists(model_path):
        if verbose:
            print(f"Warning: {model_path} not found. Using randomly initialized model.")
        model = ChannelTransformer(config=ModelCfg.transformer)
        train_config = TrainCfg
    else:
        # Load from checkpoint using ChannelTransformer's inherited CkptMixin
        model, train_config = ChannelTransformer.from_checkpoint(model_path, weights_only=False)

    model.to(DEVICE)
    model.eval()

    return {"model": model, "train_config": train_config}


@torch.no_grad()
def run_model(model_dict: Dict[str, Any], record: str, data_folder: str, verbose: bool) -> Tuple[int, float]:
    """
    Entry point for inference. Called by run_model.py.
    """
    model = model_dict["model"]
    train_config = model_dict["train_config"]

    # 1. Use DataReader to load the specific record
    reader = CINC2026(db_dir=data_folder)
    data = reader.load_data(record)

    if not data:
        if verbose:
            print(f"Warning: Could not load data for record {record}")
        return 0, 0.5

    raw_signals = data["signals"]
    raw_fs = data["fs"]

    # 2. Preprocess & Align Signals (Match logic in Dataset.__getitem__)
    processed_signals = []
    present_channel_ids = []
    sig_len = train_config.get("sig_len", 3000)
    target_fs = train_config.get("fs", 100)

    for std_name in STANDARD_CHANNELS:
        # Find matched key in loaded signals
        matched_key = next((k for k in raw_signals.keys() if k.lower() == std_name.lower()), None)

        if matched_key:
            sig = raw_signals[matched_key]
            fs = raw_fs[matched_key]

            # Resample
            if fs != target_fs:
                from scipy.signal import resample

                sig = resample(sig, int(len(sig) * target_fs / fs))

            # Padding/Cropping to fixed length
            if len(sig) > sig_len:
                sig = sig[:sig_len]
            else:
                sig = np.pad(sig, (0, max(0, sig_len - len(sig))), "constant")

            processed_signals.append(sig)
            present_channel_ids.append(CHANNEL_ID_MAP[std_name])

    if not processed_signals:
        return 0, 0.5

    # 3. Extract Demographics
    demographics = get_demographic_features(record, data_folder)

    # 4. Format for model inference
    sig_t = torch.from_numpy(np.array(processed_signals, dtype=np.float32)).unsqueeze(0)  # (1, N, L)
    ids_t = torch.tensor(present_channel_ids, dtype=torch.long).unsqueeze(0)  # (1, N)
    dem_t = torch.from_numpy(demographics.astype(np.float32)).unsqueeze(0)  # (1, D)

    # 5. Call Inference
    outputs: CINC2026Outputs = model.inference(sig_t, ids_t, dem_t)

    # Extraction from CINC2026Outputs (handling batch index 0)
    binary_output = int(outputs.cognitive_impairment[0])
    probability_output = float(outputs.ci_prob[0][1])

    return binary_output, probability_output


def get_demographic_features(record: str, data_folder: str) -> np.ndarray:
    """Extract and normalize demographic features from CSV."""
    # Find demographics file in data_folder or its subfolders
    demo_path = None
    for root, dirs, files in os.walk(data_folder):
        if DEMOGRAPHICS_FILE in files:
            demo_path = os.path.join(root, DEMOGRAPHICS_FILE)
            break

    if demo_path and os.path.exists(demo_path):
        try:
            df = pd.read_csv(demo_path)
            # Match record (BidsFolder)
            row = df[df["BidsFolder"] == record].iloc[0]
            age = float(row.get("Age", 60)) / 100.0
            sex = 1.0 if str(row.get("Sex")).lower() in ["m", "male", "1"] else 0.0
            bmi = float(row.get("BMI", 25)) / 50.0
            return np.array([age, sex, bmi])
        except Exception:
            pass

    return np.array([0.6, 0.5, 0.5])  # Default normalization placeholders
