"""post_docker_build.py — runs inside the Docker image at build time.

CinC2026 does **not** require any pretrained models or pre-downloaded data
to be baked into the image (all data, including CAISR annotations, is
provided by the challenge organisers at evaluation time).  This script
therefore only performs a lightweight environment sanity check.
"""

import sys

import numpy  # noqa: F401 — imported to verify availability at build time
import pandas  # noqa: F401 — imported to verify availability at build time
import pyedflib  # noqa: F401 — imported to verify availability at build time
import torch  # noqa: F401 — imported to verify availability at build time
import torch_ecg  # noqa: F401 — imported to verify availability at build time


def check_env() -> None:
    """Verify that the core dependencies are importable."""
    print("Checking environment …")
    print(f"  Python   : {sys.version}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  torch_ecg: {torch_ecg.__version__}")
    print(f"  numpy    : {numpy.__version__}")
    print(f"  pandas   : {pandas.__version__}")
    print("Environment check passed ✓")


if __name__ == "__main__":
    check_env()
