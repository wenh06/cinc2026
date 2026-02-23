""" """

import os
from copy import deepcopy
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch_ecg.utils.misc import str2bool

from cfg import _BASE_DIR, ModelCfg, TrainCfg
from evaluate_model import run as model_evaluator_func  # noqa: F401
from run_model import run as model_runner_func  # noqa: F401
from team_code import train_model  # noqa: F401
from utils.misc import func_indicator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype == torch.float64:
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


tmp_data_dir = Path(os.environ.get("mount_data_dir", _BASE_DIR / "tmp" / "CINC2026")).resolve()
print(f"tmp_data_dir: {str(tmp_data_dir)}")
tmp_data_dir.mkdir(parents=True, exist_ok=True)
print("data directory signal files count:", len(list(tmp_data_dir.glob("*.hea"))))

# downloading is done outside the docker container
# and the data folder is mounted to the docker container as read-only
# dr = CINC2026(tmp_data_dir)
# dr.download()
# dr._ls_rec()


tmp_model_dir = Path(os.environ.get("revenger_model_dir", TrainCfg.model_dir)).resolve()
tmp_model_dir.mkdir(parents=True, exist_ok=True)
tmp_output_dir = Path(os.environ.get("revenger_output_dir", _BASE_DIR / "tmp" / "output")).resolve()
tmp_output_dir.mkdir(parents=True, exist_ok=True)


def echo_write_permission(folder: Union[str, Path]) -> None:
    is_writeable = "is writable" if os.access(str(folder), os.W_OK) else "is not writable"
    print(f"{str(folder)} {is_writeable}")


echo_write_permission(tmp_data_dir)
echo_write_permission(tmp_model_dir)
echo_write_permission(tmp_output_dir)


@func_indicator("testing dataset")
def test_dataset() -> None:
    """Test the dataset."""
    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(ds_config.working_dir)

    raise NotImplementedError


@func_indicator("testing models")
def test_models() -> None:
    """Test the models."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    ds_config = deepcopy(TrainCfg)
    ds_config.db_dir = tmp_data_dir
    ds_config.working_dir = tmp_model_dir / "working_dir"
    ds_config.working_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError


@func_indicator("testing challenge metrics")
def test_challenge_metrics() -> None:
    """Test the challenge metrics."""

    raise NotImplementedError


@func_indicator("testing trainer")
def test_trainer() -> None:
    """Test the trainer."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)
    echo_write_permission(tmp_output_dir)

    train_config = deepcopy(TrainCfg)
    train_config.db_dir = tmp_data_dir
    # train_config.model_dir = model_folder
    # train_config.final_model_filename = "final_model.pth.tar"
    train_config.debug = True
    train_config.working_dir = tmp_model_dir / "working_dir"
    train_config.working_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError


@func_indicator("testing challenge entry")
def test_entry() -> None:
    """Test Challenge entry."""
    echo_write_permission(tmp_data_dir)
    echo_write_permission(tmp_model_dir)

    # run the model training function (script)
    print("   Run model training function   ".center(100, "#"))
    data_folder = tmp_data_dir
    model_folder = tmp_model_dir

    raise NotImplementedError


if __name__ == "__main__":
    TEST_FLAG = os.environ.get("CINC2026_REVENGER_TEST", False)
    TEST_FLAG = str2bool(TEST_FLAG)
    if not TEST_FLAG:
        # raise RuntimeError(
        #     "please set CINC2026_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test"
        # )
        print("Test is skipped.")
        print("Please set CINC2026_REVENGER_TEST to true (1, y, yes, true, etc.) to run the test:")
        print("CINC2026_REVENGER_TEST=1 python test_docker.py")
        print("Other environment variables:")
        print("mount_data_dir: the data directory, usage:")
        print("CINC2026_REVENGER_TEST=1 mount_data_dir=/path/to/data python test_docker.py")
        # TODO: add more environment variables here
        exit(0)

    print("#" * 100)
    print("testing team code")
    print("#" * 100)
    print(f"tmp_data_dir: {str(tmp_data_dir)}")
    print(f"tmp_model_dir: {str(tmp_model_dir)}")
    print(f"tmp_output_dir: {str(tmp_output_dir)}")
    print("#" * 100)

    # test_dataset()
    # test_models()
    # test_challenge_metrics()
    # test_trainer()
    # test_entry()
