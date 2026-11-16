"""Dry-run the configured submission component chain before training.

The official tabular/Phi paths read ``demographics.csv`` directly and do not
instantiate ``CINC2026Dataset``.  This script therefore validates the actual
submission contract: component names, per-component overrides, feature-cache
coverage, canonical training-record hash, class balance, the 1,103-row /
390-feature cache contract and the expected number of boosters.  It imports the
same config builder and data-root resolver as ``team_code.train_model`` so the
gate and real entry cannot choose different paths.  The exact row/feature
contract belongs to the Entry-1 release components; the global training config
remains permissive for exploratory runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_runtime_versions import collect_versions  # noqa: E402

from component_registry import (  # noqa: E402
    component_config,
    enabled_components,
)
from tabular_pipeline import preflight_tabular  # noqa: E402
from team_code import prepare_train_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/challenge/training_data"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--expect-components",
        nargs="+",
        default=["spec_xgb_10seed", "sub5_tabular_xgb"],
    )
    args = parser.parse_args()

    config = prepare_train_config(str(args.data_dir))
    components = enabled_components(config)
    names = [str(component.get("name")) for component in components]
    if names != list(args.expect_components):
        raise SystemExit(f"component mismatch: expected {args.expect_components}, got {names}")

    component_results = {}
    for component in components:
        ctype = str(component.get("type"))
        name = str(component.get("name"))
        if ctype != "tabular":
            component_results[name] = {"type": ctype, "status": "not dry-run by this script"}
            continue
        effective = component_config(config, component, ctype)
        summary = preflight_tabular(effective, verbose=True)
        if name == "spec_xgb_10seed" and summary["expected_boosters"] != 10:
            raise SystemExit("spec_xgb_10seed must produce exactly 10 boosters")
        if name == "sub5_tabular_xgb" and summary["expected_boosters"] != 1:
            raise SystemExit("sub5_tabular_xgb must remain a single-booster fallback")
        component_results[name] = summary

    payload = {
        "data_dir": str(config.db_dir),
        "components": component_results,
        "runtime": collect_versions(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
