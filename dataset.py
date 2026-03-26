from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin

from cfg import TrainCfg
from const import CHANNEL_ID_MAP, MODALITY_MAP, STANDARD_CHANNELS
from data_reader import CINC2026


class CINC2026Dataset(Dataset, ReprMixin):
    """
    Dataset for the CinC2026 Challenge.
    Organizes data for ChannelTransformer or MultiBranchNet.
    """

    __name__ = "CINC2026Dataset"

    def __init__(
        self,
        config: CFG,
        training: bool = True,
        lazy: bool = True,
        **reader_kwargs,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.lazy = lazy

        # Initialize reader
        db_dir = self.config.get("db_dir", reader_kwargs.get("db_dir"))
        self.reader = CINC2026(db_dir=db_dir, **reader_kwargs)

        # Stratified split or simple split
        self.records = self._train_test_split()

        self.sig_len = self.config.get("sig_len", 3000)  # e.g., 30s at 100Hz
        self.fs = self.config.get("fs", 100)
        self.dtype = np.float32

    def __len__(self) -> int:
        return len(self.records)

    def _extract_demographics(self, rec: str) -> np.ndarray:
        """Extract and normalize demographic features."""
        row = self.reader._df_records.loc[rec]
        # Example features: Age, Sex, BMI
        age = float(row.get("Age", 60)) / 100.0
        sex = 1.0 if str(row.get("Sex")).lower() in ["m", "male", "1"] else 0.0
        bmi = float(row.get("BMI", 25)) / 50.0
        # Return as vector
        return np.array([age, sex, bmi], dtype=self.dtype)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self.records[index]
        data = self.reader.load_data(rec)
        if not data:
            # Fallback for failed loads
            return self.__getitem__((index + 1) % len(self.records))

        raw_signals = data["signals"]
        raw_fs = data["fs"]

        # 1. Resample and Align Signals
        processed_signals = []
        present_channel_ids = []

        # Grouped by modality for MultiBranch
        modality_data = {mod: [] for mod in MODALITY_MAP.keys()}

        for std_name in STANDARD_CHANNELS:
            # Match standardized names from reader
            # We need to find if any key in raw_signals matches std_name (case-insensitive)
            matched_key = next((k for k in raw_signals.keys() if k.lower() == std_name.lower()), None)

            if matched_key:
                sig = raw_signals[matched_key]
                fs = raw_fs[matched_key]

                # Resample to target fs
                if fs != self.fs:
                    from scipy.signal import resample

                    num_samples = int(len(sig) * self.fs / fs)
                    sig = resample(sig, num_samples)

                # Pad/Crop to fixed length
                if len(sig) > self.sig_len:
                    sig = sig[: self.sig_len]
                else:
                    sig = np.pad(sig, (0, self.sig_len - len(sig)), "constant")

                processed_signals.append(sig)
                present_channel_ids.append(CHANNEL_ID_MAP[std_name])

                # Assign to modality branch
                for mod, mod_list in MODALITY_MAP.items():
                    if std_name in mod_list:
                        modality_data[mod].append(sig)

        # 2. Extract Demographics and Labels
        demographics = self._extract_demographics(rec)
        label = int(self.reader._df_records.loc[rec].get("Cognitive_Impairment", 0))

        # 3. Format output for the specific model
        output = {"demographics": torch.from_numpy(demographics), "labels": torch.tensor(label, dtype=torch.long)}

        if self.config.model_name == "transformer":
            output["signals"] = torch.from_numpy(np.array(processed_signals, dtype=self.dtype))  # (N, L)
            output["channel_ids"] = torch.tensor(present_channel_ids, dtype=torch.long)  # (N)
        else:
            # MultiBranch format
            for mod, sigs in modality_data.items():
                if sigs:
                    output[mod.lower()] = torch.from_numpy(np.array(sigs, dtype=self.dtype))  # (N_mod, L)
                else:
                    # Empty branch handled by model as zero-padding
                    output[mod.lower()] = torch.zeros((0, self.sig_len), dtype=torch.float32)

        return output

    def _train_test_split(self) -> List[str]:
        # Simple split for now; can be improved with StratifiedKFold
        all_recs = self.reader._all_records
        split_idx = int(len(all_recs) * 0.8)
        if self.training:
            return all_recs[:split_idx]
        else:
            return all_recs[split_idx:]
