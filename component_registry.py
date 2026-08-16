#!/usr/bin/env python
"""Model-component registry: manifest plus per-type artifact names and helpers.

CinC 2026 submissions may bundle several independently trained models with a
per-record fallback chain (e.g. a Phi-latent ranker whose misses fall back to
the sub5 tabular XGB).  ``TrainCfg.components`` is an ordered list of component
specs; :func:`enabled_components` returns the enabled ones sorted by priority.
Every component trains into ``model_folder/components/<name>/`` and the
resulting :data:`MANIFEST_NAME` at the model-folder root records what was
trained, so ``load_model`` routes from the on-disk manifest instead of the
*current* ``TrainCfg`` (which may have drifted between training and loading).

The official runtime is read-only except ``model_folder`` and has no network
access.  This module only ever writes under ``model_folder`` and performs no
network I/O; :func:`resolve_feature_cache` only points at a cache that is baked
into the image at build time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "model_manifest.json"
MANIFEST_FORMAT_VERSION = 1

COMPONENT_TYPES = ("tabular", "crnn", "phi")

# Per-component artifact names.  ``FINAL_MODEL_NAME`` in team_code.py is kept
# as a backward-compatible alias of ``ARTIFACT_NAMES["crnn"]``.
ARTIFACT_NAMES: Dict[str, Any] = {
    "crnn": "final_model.pth.tar",
    "tabular": {
        "config": "tabular_config.json",
        "xgboost": "tabular_model.json",
        "lightgbm": "tabular_model.txt",
    },
}


def enabled_components(train_config: Any) -> List[Any]:
    """Return the enabled component specs ordered by (priority, name).

    ``TrainCfg.tabular.enable`` remains the legacy master switch for the
    tabular path: when it is explicitly off, tabular-type components are
    skipped so old workflows (and ``test_docker.test_entry``'s CRNN branch)
    keep their semantics.
    """
    comps = train_config.get("components", None)
    if not comps:
        return []
    tab = train_config.get("tabular", None)
    tabular_on = tab is None or bool(tab.get("enable", False))
    enabled = []
    for c in comps:
        if not bool(c.get("enable", True)):
            continue
        if str(c.get("type")) == "tabular" and not tabular_on:
            continue
        enabled.append(c)
    enabled.sort(key=lambda c: (int(c.get("priority", 0)), str(c.get("name", ""))))
    return enabled


def component_dir(model_folder: Path, name: str) -> Path:
    """Artifact directory for one component."""
    return model_folder / "components" / name


def write_manifest(model_folder: Path, components: List[Any]) -> Path:
    """Write the component manifest under ``model_folder`` (the only writable dir)."""
    payload = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "components": [
            {
                "name": str(c.get("name")),
                "type": str(c.get("type")),
                "priority": int(c.get("priority", 0)),
                "enable": bool(c.get("enable", True)),
            }
            for c in components
        ],
    }
    path = model_folder / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_manifest(model_folder: Path) -> Optional[Dict[str, Any]]:
    """Read the manifest if present; ``None`` means a legacy single-model layout."""
    path = model_folder / MANIFEST_NAME
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise RuntimeError(f"unsupported manifest format: {manifest.get('format_version')}")
    return manifest


def resolve_feature_cache(train_config: Any) -> None:
    """Fill ``tabular.feature_cache`` from the baked repo cache when unset.

    The Dockerfile bakes the D1 spectral feature bank into
    ``<repo>/data/spectral_features``.  Resolving it here (rather than inside
    ``tabular_pipeline``) keeps ``train_tabular`` / ``load_tabular_model``
    parameter-exact for tests, and only ever reads files bundled into the
    image — no network, no writes outside ``model_folder``.
    """
    tab = train_config.get("tabular", None)
    if tab is None:
        return
    if str(tab.get("feature_cache", "") or "").strip():
        return  # an explicit path wins
    candidate = Path(__file__).resolve().parent / "data" / "spectral_features" / "features.csv"
    if candidate.is_file():
        tab.feature_cache = str(candidate)
