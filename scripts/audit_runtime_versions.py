"""Record runtime versions for host/container reproducibility audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from urllib.parse import unquote, urlparse

PACKAGES = (
    "torch",
    "xgboost",
    "lightgbm",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "torch-ecg",
    "bib-lookup",
)


def _vcs_commit(package: str) -> str | None:
    """Resolve the installed source commit, including editable installs."""
    try:
        dist = distribution(package)
        direct_url_text = dist.read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        return None
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None

    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
        return commit

    source_url = direct_url.get("url")
    editable = bool((direct_url.get("dir_info") or {}).get("editable"))
    if not editable or not isinstance(source_url, str) or not source_url.startswith("file://"):
        return None
    source_path = Path(unquote(urlparse(source_url).path))
    try:
        result = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    resolved = result.stdout.strip()
    return resolved if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", resolved) else None


def collect_versions() -> dict:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CINC2026_XGB_N_JOBS")
        },
        "build_sha": os.environ.get("CINC2026_BUILD_SHA"),
        "packages": packages,
        "vcs_commits": {
            package: commit for package in ("torch-ecg", "bib-lookup") if (commit := _vcs_commit(package)) is not None
        },
    }
    try:
        import torch

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        payload["torch_runtime"] = {
            "num_threads": torch.get_num_threads(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": cuda_visible_devices,
            "cuda_suppressed": cuda_visible_devices in {"", "-1"},
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": (
                [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            ),
        }
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        payload["torch_runtime"] = {"error": repr(exc)}

    files = {}
    for name in ("requirements-docker.txt", "Dockerfile"):
        path = Path(__file__).resolve().parents[1] / name
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["file_sha256"] = files
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    payload = collect_versions()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
