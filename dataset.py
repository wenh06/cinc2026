""" """

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataset import Dataset
from torch_ecg._preprocessors import PreprocManager
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin
from torch_ecg.utils.utils_data import one_hot_encode  # noqa: F401
from torch_ecg.utils.utils_nn import default_collate_fn  # noqa: F401

from cfg import TrainCfg
from data_reader import CINC2026

__all__ = [
    "CINC2026Dataset",
]

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class CINC2026Dataset(Dataset, ReprMixin):
    """Dataset for the CinC2026 Challenge.

    Parameters
    ----------
    config : CFG
        configuration for the dataset
    training : bool, default True
        whether the dataset is for training or validation
    lazy : bool, default True
        whether to load all data into memory at initialization
    reader_kwargs : dict, optional
        keyword arguments for the data reader class.

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

        if self.config.get("db_dir", None) is None:
            self.config.db_dir = reader_kwargs.pop("db_dir", None)
            assert self.config.db_dir is not None, "db_dir must be specified"
        else:
            reader_kwargs.pop("db_dir", None)
        if self.config.get("use_dbs", None) is not None:
            reader_kwargs["use_dbs"] = self.config.use_dbs
        self.config.db_dir = Path(self.config.db_dir).expanduser().resolve()

        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        self.__cache = None
        self.reader = CINC2026(db_dir=self.config.db_dir, **reader_kwargs)
        self.records = self._train_test_split()

        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: Union[int, slice]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _load_all_data(self, batch_size: int = 256, num_workers: Optional[int] = None) -> None:
        """Load all data into memory using DataLoader for multi-process acceleration.

        Parameters
        ----------
        batch_size : int, default 256
            Number of samples to load in each batch.
        num_workers : int, optional
            Number of worker processes for data loading.
            Set to 0 to disable multiprocessing (useful for debugging).

        .. warning::

            Caching all data into memory is not recommended, which would certainly cause OOM error.
            The RAM of the Challenge is only 64GB.

        """
        raise NotImplementedError

    def _train_test_split(self, train_ratio: float = 0.8, part: Optional[Literal["train", "val", "test"]] = None) -> List[str]:
        """Split the dataset into training and validation sets
        in a stratified manner.

        Parameters
        ----------
        train_ratio : float, default 0.8
            The ratio of the training set.
        part : {"train", "val", "test"}, optional
            The part of the dataset to return.
            If None, it will be determined based on the training flag.

        Returns
        -------
        list
            List of record names.

        """
        _train_ratio = int(train_ratio * 100)
        _test_ratio = 100 - _train_ratio
        assert _train_ratio * _test_ratio > 0, "train_ratio and test_ratio must be positive"

        raise NotImplementedError

    @property
    def cache(self) -> Union[None, Dict[str, torch.Tensor]]:
        return self.__cache

    def extra_repr_keys(self) -> List[str]:
        return ["reader", "training"]


class FastDataReader(ReprMixin, Dataset):
    def __init__(
        self,
        reader: CINC2026,
        records: Sequence[str],
        config: CFG,
        ppm: Optional[PreprocManager] = None,
    ) -> None:
        self.reader = reader
        self.records = records
        self.config = config
        self.ppm = ppm
        if self.config.torch_dtype == torch.float64:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: Union[int, list, slice]) -> Dict[str, np.ndarray]:
        raise NotImplementedError
