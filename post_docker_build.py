"""post_docker_build.py — runs inside the Docker image at build time.

Two responsibilities:

1. Environment sanity check (core dependencies importable).
2. Download the Philosopher's Stone checkpoint (BDSP brain-health model,
   third_party/philosophers-stone) into ``$MODEL_CACHE_DIR``.  The official
   runner has no guaranteed network at *runtime*, so the checkpoint must be
   baked into the image here at build time.  The download is pinned by size and
   SHA-256 and fails the build on mismatch (fail-fast instead of failing the
   first inference call on the scoring cluster).  Sources are tried in order:
   huggingface.co, hf-mirror.com, then ``PHI_MEGA_URL`` (optional MEGA link).

Set ``PHI_MODEL_DOWNLOAD=0`` to skip the download (e.g. fast local/CI builds).
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy  # noqa: F401 — imported to verify availability at build time
import pandas  # noqa: F401 — imported to verify availability at build time
import pyedflib  # noqa: F401 — imported to verify availability at build time
import torch  # noqa: F401 — imported to verify availability at build time
import torch_ecg  # noqa: F401 — imported to verify availability at build time

PHI_REPO_ID = "wolfgang-ganglberger/philosophers-stone"
PHI_REVISION = "main"
PHI_CKPT_NAME = "SleepPhilosophersStone.ckpt"
PHI_URLS = (
    f"https://huggingface.co/{PHI_REPO_ID}/resolve/{PHI_REVISION}/{PHI_CKPT_NAME}",
    # China-friendly mirror, same layout as the 2024 entry (hf-mirror.com)
    f"https://hf-mirror.com/{PHI_REPO_ID}/resolve/{PHI_REVISION}/{PHI_CKPT_NAME}",
)
PHI_SIZE = 2393981880
PHI_SHA256 = "b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87"
PHI_RETRIES = 3

FEATURE_CACHE_DIR = Path(__file__).resolve().parent / "data" / "spectral_features"
FEATURE_CACHE_FILES = ("features.csv", "record_meta.csv", "manifest.json")


def check_env() -> None:
    """Verify that the core dependencies are importable."""
    print("Checking environment …")
    print(f"  Python   : {sys.version}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  torch_ecg: {torch_ecg.__version__}")
    print(f"  numpy    : {numpy.__version__}")
    print(f"  pandas   : {pandas.__version__}")
    print("Environment check passed ✓")


def check_phi_source() -> None:
    """Sanity-check that the vendored philosophers-stone source is present."""
    submodule_root = Path(__file__).resolve().parent / "third_party" / "philosophers-stone"
    if not (submodule_root / "src" / "philosophers_stone" / "__init__.py").exists():
        raise RuntimeError(
            "third_party/philosophers-stone is empty — "
            "the source is vendored in the repository and must not depend on git submodules."
        )
    print("Philosopher's Stone source present ✓")


def verify_feature_cache() -> None:
    """Verify the tabular spectral-feature cache downloaded by the Dockerfile."""
    missing = [name for name in FEATURE_CACHE_FILES if not (FEATURE_CACHE_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"tabular feature cache absent at {FEATURE_CACHE_DIR} (missing: {missing})")
    manifest = json.loads((FEATURE_CACHE_DIR / "manifest.json").read_text())
    for entry in manifest.get("files", []):
        name = entry.get("name")
        if name not in FEATURE_CACHE_FILES or name == "manifest.json":
            continue
        digest = _sha256(FEATURE_CACHE_DIR / name)
        if digest != entry.get("sha256"):
            raise RuntimeError(f"feature cache {name}: SHA-256 mismatch ({digest} != {entry.get('sha256')})")
    print(
        f"tabular feature cache verified at {FEATURE_CACHE_DIR} ✓ "
        f"({manifest.get('n_records')} records, {manifest.get('n_features')} features)"
    )


def download_phi_checkpoint() -> None:
    """Download and verify the Philosopher's Stone checkpoint."""
    cache_root = Path(os.environ.get("MODEL_CACHE_DIR", "/challenge/cache/revenger_model_dir"))
    dest = cache_root / "philosophers-stone" / "model_files" / PHI_CKPT_NAME
    if dest.exists() and dest.stat().st_size == PHI_SIZE:
        if _sha256(dest) == PHI_SHA256:
            print(f"Phi checkpoint already cached at {dest} ✓")
            return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".download")
    sources = [(url, "http") for url in PHI_URLS]
    mega_url = os.environ.get("PHI_MEGA_URL", "").strip()
    if mega_url:
        sources.append((mega_url, "mega"))
    last_error: Exception | None = None
    for attempt in range(1, PHI_RETRIES + 1):
        for index, (url, kind) in enumerate(sources, start=1):
            try:
                print(
                    f"Downloading Philosopher's Stone checkpoint "
                    f"({PHI_SIZE / 1e9:.2f} GB, attempt {attempt}/{PHI_RETRIES}, "
                    f"source {index}/{len(sources)} [{kind}]) …"
                )
                if kind == "mega":
                    _download_mega(url, tmp)
                else:
                    _download(url, tmp)
                size = tmp.stat().st_size
                if size != PHI_SIZE:
                    raise RuntimeError(f"size mismatch: expected {PHI_SIZE}, got {size}")
                digest = _sha256(tmp)
                if digest != PHI_SHA256:
                    raise RuntimeError(f"SHA-256 mismatch: {digest}")
                tmp.replace(dest)
                print(f"Phi checkpoint verified and stored at {dest} ✓")
                return
            except Exception as exc:  # noqa: BLE001 — retry any transient failure
                last_error = exc
                print(f"  attempt {attempt} source {index} failed: {exc}", file=sys.stderr)
                if tmp.exists():
                    tmp.unlink()
    raise RuntimeError(f"Failed to download Philosopher's Stone checkpoint after {PHI_RETRIES} attempts: {last_error}")


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "cinc2026-docker-build"})
    with urllib.request.urlopen(request, timeout=120) as response, open(dest, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def _download_mega(url: str, dest: Path) -> None:
    """Download via megadl (MEGA links are not plain HTTP downloads)."""
    import subprocess

    if dest.exists():
        dest.unlink()
    result = subprocess.run(
        ["megadl", "--path", str(dest), url],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-500:]
        raise RuntimeError(f"megadl failed: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    check_env()
    check_phi_source()
    if os.environ.get("FEATURE_CACHE_DOWNLOAD", "1") != "0":
        verify_feature_cache()
    else:
        print("FEATURE_CACHE_DOWNLOAD=0 — skipping feature cache verification.")
    if os.environ.get("PHI_MODEL_DOWNLOAD", "0") != "0":
        download_phi_checkpoint()
    else:
        print("PHI_MODEL_DOWNLOAD=0 — skipping Philosopher's Stone checkpoint download.")


if __name__ == "__main__":
    main()
