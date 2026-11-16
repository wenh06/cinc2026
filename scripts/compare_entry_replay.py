"""Compare host and official-container predictions record by record.

The official ``run_model.py`` output is not a labelled evaluation file:
``helper_code.update_demographics_table`` overwrites the original
``Cognitive_Impairment`` column with the model's binary prediction.  AUROC must
therefore be computed from an explicit ground-truth demographics file, never
from either prediction output.

The numeric probability difference is a diagnostic, not a version detector: on
a small LOSO holdout, legitimate row-order sensitivity can be larger than
library-version drift.  Runtime manifest comparison is the primary evidence for
R2-B; probability and AUROC differences explain its expected impact.

Two replay scopes are explicit:

``submission``
    The submission-style path (full training → full prediction) with at least
    1,000 records and paired runtime-version manifests.  A first run may use
    ``--calibration-run`` to measure the residual without a release cap; it is
    never release-eligible.  A release run requires an explicit
    ``--calibration-json`` and a ``--max-probability-delta`` at least the
    measured residual times ``--calibration-margin`` (absolute cap <= 0.5).
    The calibration artifact is bound to the prediction, version-manifest,
    label and record-key hashes, and its residual is recomputed at release.

``loso``
    A small held-out-site replay.  Metrics are report-only unless an explicit
    ``--max-probability-delta`` is supplied, in which case that user-selected
    probability cap is an opt-in hard integrity check.  No pass/fail is derived
    automatically from sample-size-sensitive drift statistics.

For hidden validation/test data without labels, pass ``--unlabeled``.  Labelled
runs require a separate original demographics file.  All summary artifacts are
written before a gate exits, including failed gates.  Declare expected manifest
differences with ``--known-version-diff``; unknown differences set
``release_eligible=false`` and require review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def age_conditioned_auroc_with_pairs(
    probs: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    tolerance: float = 2.0,
) -> tuple[float, int]:
    """Return the official-style age-conditioned AUROC and matched-pair count.

    A degenerate cohort has no defined AUROC.  The official evaluator fails
    closed on a zero denominator; this helper does the same rather than returning
    a neutral 0.5 that could silently pass a replay gate.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    ages = np.asarray(ages, dtype=float)
    pairs = age_matched_pair_count(labels, ages, tolerance)
    if pairs == 0:
        raise ValueError(f"age-conditioned AUROC has zero matched pairs at tolerance +/-{tolerance:g} years")

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    agree = 0.0
    for i in pos:
        valid = neg[np.abs(ages[neg] - ages[i]) <= tolerance]
        agree += float((probs[i] > probs[valid]).sum()) + 0.5 * float((probs[i] == probs[valid]).sum())
    return float(agree / pairs), int(pairs)


def age_matched_pair_count(labels: np.ndarray, ages: np.ndarray, tolerance: float = 2.0) -> int:
    labels = np.asarray(labels)
    ages = np.asarray(ages, dtype=float)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    return int(sum(np.count_nonzero(np.abs(ages[neg] - ages[i]) <= tolerance) for i in pos))


def age_conditioned_auroc(probs: np.ndarray, labels: np.ndarray, ages: np.ndarray, tolerance: float = 2.0) -> float:
    """Backward-compatible score-only wrapper; degenerate cohorts fail closed."""
    score, _ = age_conditioned_auroc_with_pairs(probs, labels, ages, tolerance)
    return score


def _record_keys(frame: pd.DataFrame) -> pd.Series:
    required = {"BidsFolder", "SessionID"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing record-key columns: {sorted(missing)}")
    return frame["BidsFolder"].astype(str) + "__" + frame["SessionID"].astype(str)


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"BidsFolder", "SessionID", "Cognitive_Impairment_Probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame["record_key"] = _record_keys(frame)
    if frame["record_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate record keys")
    probs = frame["Cognitive_Impairment_Probability"].astype(float).to_numpy()
    if not np.isfinite(probs).all():
        raise ValueError(f"{path} contains non-finite probabilities")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"{path} contains probabilities outside [0, 1]")
    return frame.set_index("record_key", drop=False)


def load_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"BidsFolder", "SessionID", "Cognitive_Impairment", "Age"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing ground-truth columns: {sorted(missing)}")
    frame["record_key"] = _record_keys(frame)
    if frame["record_key"].duplicated().any():
        raise ValueError(f"{path} contains duplicate record keys")
    labels = frame["Cognitive_Impairment"].astype(float)
    if not np.isfinite(labels).all() or not labels.isin([0.0, 1.0]).all():
        raise ValueError(f"{path} contains non-binary or missing ground-truth labels")
    ages = frame["Age"].astype(float)
    if not np.isfinite(ages).all():
        raise ValueError(f"{path} contains non-finite ground-truth ages")
    return frame.set_index("record_key", drop=False)


def _flatten_version(payload: dict, prefix: str = "") -> dict:
    flattened = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_version(value, name))
        else:
            flattened[name] = value
    return flattened


def compare_runtime_versions(
    host_payload: dict,
    container_payload: dict,
    known_difference_keys: Optional[list[str]] = None,
) -> dict:
    host = _flatten_version(host_payload)
    container = _flatten_version(container_payload)
    differences = []
    for key in sorted(set(host).union(container)):
        if host.get(key) != container.get(key):
            differences.append({"key": key, "host": host.get(key), "container": container.get(key)})

    mandatory_unknown_keys = set()
    host_commit = (host_payload.get("vcs_commits") or {}).get("torch-ecg")
    container_commit = (container_payload.get("vcs_commits") or {}).get("torch-ecg")
    torch_ecg_key = "vcs_commits.torch-ecg"
    if not host_commit or not container_commit or host_commit != container_commit:
        mandatory_unknown_keys.add(torch_ecg_key)
        if not any(difference["key"] == torch_ecg_key for difference in differences):
            differences.append({"key": torch_ecg_key, "host": host_commit, "container": container_commit})

    known_prefixes = [key.strip().rstrip(".") for key in (known_difference_keys or []) if key.strip()]

    def is_known(key: str) -> bool:
        return any(key == prefix or key.startswith(prefix + ".") for prefix in known_prefixes)

    known_differences = [
        difference
        for difference in differences
        if is_known(difference["key"]) and difference["key"] not in mandatory_unknown_keys
    ]
    unknown_differences = [
        difference
        for difference in differences
        if difference["key"] in mandatory_unknown_keys or not is_known(difference["key"])
    ]
    return {
        "match": not differences,
        "difference_count": len(differences),
        "differences": differences,
        "known_differences": known_differences,
        "unknown_differences": unknown_differences,
        "manifest_review_required": bool(unknown_differences),
        "known_difference_prefixes": known_prefixes,
    }


def _emit(summary: dict, json_out: Optional[Path]) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(rendered + "\n")


def _fail(summary: dict, reason: str, json_out: Optional[Path]) -> None:
    summary["status"] = "fail"
    summary["release_eligible"] = False
    summary["failure_reason"] = reason
    _emit(summary, json_out)
    raise SystemExit(reason)


def _load_version_summary(args: argparse.Namespace) -> dict:
    if args.host_versions is None:
        return {"status": "not_supplied"}
    host_versions = json.loads(args.host_versions.read_text())
    container_versions = json.loads(args.container_versions.read_text())
    return compare_runtime_versions(host_versions, container_versions, args.known_version_diff)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_evidence(args: argparse.Namespace, record_keys: list[str]) -> dict:
    return {
        "host_predictions_sha256": _sha256_file(args.host),
        "container_predictions_sha256": _sha256_file(args.container),
        "host_versions_sha256": _sha256_file(args.host_versions) if args.host_versions is not None else None,
        "container_versions_sha256": _sha256_file(args.container_versions) if args.container_versions is not None else None,
        "labels_sha256": _sha256_file(args.label_source) if args.label_source is not None else None,
        "record_keys_sha256": hashlib.sha256("\n".join(record_keys).encode()).hexdigest(),
    }


def _load_calibration_evidence(
    path: Path,
    records: int,
    selected_cap: float,
    margin: float,
    input_evidence: dict,
    recomputed_residual: float,
) -> dict:
    payload = json.loads(path.read_text())
    diagnostics = payload.get("numeric_diagnostics", {})
    residual = diagnostics.get("max_abs_probability_delta")
    calibration_status = payload.get("status")
    calibration_scope = payload.get("scope")
    calibration_records = diagnostics.get("records")
    if calibration_status != "calibration_only" or calibration_scope != "submission":
        raise ValueError("calibration JSON must be a submission-scope calibration_only artifact")
    if calibration_records != records:
        raise ValueError(f"calibration artifact has {calibration_records} records, current replay has {records}")
    if residual is None:
        raise ValueError("calibration artifact has no max_abs_probability_delta")
    residual = float(residual)
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("calibration artifact has an invalid max_abs_probability_delta")
    calibration_inputs = payload.get("input_evidence")
    if not isinstance(calibration_inputs, dict):
        raise ValueError("calibration artifact has no bound input evidence")
    mismatched_inputs = sorted(key for key, value in input_evidence.items() if calibration_inputs.get(key) != value)
    if mismatched_inputs:
        raise ValueError(f"calibration inputs do not match the release replay: {mismatched_inputs}")
    if not np.isclose(residual, recomputed_residual, rtol=0.0, atol=1e-12):
        raise ValueError(f"calibration residual {residual:.17g} does not match recomputed residual {recomputed_residual:.17g}")
    if selected_cap < residual * margin:
        raise ValueError(
            f"selected cap {selected_cap:.6f} is below calibration residual " f"{residual:.6f} x margin {margin:.2f}"
        )
    return {
        "path": str(path.expanduser().resolve()),
        "records": int(calibration_records),
        "calibration_residual_max_abs_probability_delta": residual,
        "selected_cap": float(selected_cap),
        "margin": float(margin),
        "cap_to_residual_ratio": float(selected_cap / residual) if residual else None,
        "recomputed_residual_max_abs_probability_delta": float(recomputed_residual),
        "input_evidence_sha256": hashlib.sha256(
            json.dumps(input_evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--container", type=Path, required=True)
    parser.add_argument(
        "--scope",
        choices=("submission", "loso"),
        required=True,
        help=(
            "submission: n>=1000 and paired manifests; " "loso: diagnostics only unless an explicit probability cap is supplied"
        ),
    )
    parser.add_argument(
        "--label-source",
        type=Path,
        default=None,
        help="original labelled demographics.csv; never use a run_model output file",
    )
    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="explicitly run without ground-truth labels on hidden official data",
    )
    parser.add_argument("--p99-diagnostic-threshold", type=float, default=0.02)
    parser.add_argument(
        "--max-probability-delta",
        type=float,
        default=None,
        help="explicit release cap in [0, 0.5]; choose it from a calibration run",
    )
    parser.add_argument(
        "--calibration-run",
        action="store_true",
        help="report numeric drift without a release cap; never release-eligible",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="calibration_only JSON required for a formal submission release run",
    )
    parser.add_argument("--calibration-margin", type=float, default=1.2)
    parser.add_argument(
        "--known-version-diff",
        action="append",
        default=[],
        help="manifest key expected to differ, e.g. packages.xgboost; repeatable",
    )
    parser.add_argument("--max-auroc-delta", type=float, default=0.01)
    parser.add_argument("--host-versions", type=Path, default=None)
    parser.add_argument("--container-versions", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.label_source is None and not args.unlabeled:
        raise SystemExit("ground-truth labels were not supplied; pass --label-source or --unlabeled")
    if args.label_source is not None and args.unlabeled:
        raise SystemExit("--unlabeled cannot be combined with --label-source")
    if (args.host_versions is None) != (args.container_versions is None):
        raise SystemExit("--host-versions and --container-versions must be supplied together")
    if args.calibration_run and (args.max_probability_delta is not None or args.calibration_json is not None):
        raise SystemExit("--calibration-run cannot be combined with a release cap or calibration JSON")
    if args.calibration_margin < 1.0:
        raise SystemExit("--calibration-margin must be at least 1.0")
    if args.label_source is not None:
        label_path = args.label_source.expanduser().resolve()
        if label_path in {args.host.expanduser().resolve(), args.container.expanduser().resolve()}:
            raise SystemExit("--label-source must be the original demographics file, not a prediction output")

    summary = {
        "scope": args.scope,
        "metrics_mode": "labelled" if args.label_source is not None else "unlabeled",
        "runtime_version_comparison": {"status": "not_loaded"},
        "numeric_diagnostics": {},
    }
    try:
        version_summary = _load_version_summary(args)
        summary["runtime_version_comparison"] = version_summary
        host = load_predictions(args.host)
        container = load_predictions(args.container)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        _fail(summary, f"failed to read replay input: {exc}", args.json_out)
    if args.scope == "submission" and args.host_versions is None:
        _fail(summary, "submission replay requires paired runtime-version manifests", args.json_out)
    if args.scope == "submission" and not args.calibration_run and args.max_probability_delta is None:
        _fail(
            summary,
            "submission replay requires an explicit --max-probability-delta release cap; "
            "run --calibration-run first and choose the cap from the measured residual",
            args.json_out,
        )
    if args.scope == "submission" and not args.calibration_run and args.calibration_json is None:
        _fail(
            summary,
            "submission replay requires --calibration-json from a prior calibration run",
            args.json_out,
        )
    if args.max_probability_delta is not None and not (0.0 <= args.max_probability_delta <= 0.5):
        _fail(
            summary,
            "release probability cap must be within [0, 0.5]; larger caps are not a release gate",
            args.json_out,
        )
    if set(host.index) != set(container.index):
        summary["records_host"] = int(len(host))
        summary["records_container"] = int(len(container))
        _fail(summary, "prediction key mismatch", args.json_out)

    keys = sorted(host.index)
    host = host.loc[keys]
    container = container.loc[keys]
    hp = host["Cognitive_Impairment_Probability"].astype(float).to_numpy()
    cp = container["Cognitive_Impairment_Probability"].astype(float).to_numpy()
    if len(keys) == 0:
        _fail(summary, "prediction files contain no records", args.json_out)
    try:
        input_evidence = _input_evidence(args, keys)
    except OSError as exc:
        _fail(summary, f"failed to hash replay inputs: {exc}", args.json_out)
    summary["input_evidence"] = input_evidence
    delta = cp - hp
    abs_delta = np.abs(delta)
    diagnostics = {
        "records": len(keys),
        "p99_abs_probability_delta": float(np.quantile(abs_delta, 0.99)),
        "max_abs_probability_delta": float(np.max(abs_delta)),
        "mean_abs_probability_delta": float(np.mean(abs_delta)),
        "rmse_probability_delta": float(np.sqrt(np.mean(delta**2))),
        "changed_records": int(np.count_nonzero(delta)),
        "p99_diagnostic_threshold": float(args.p99_diagnostic_threshold),
        "p99_exceeds_diagnostic_threshold": bool(np.quantile(abs_delta, 0.99) > args.p99_diagnostic_threshold),
    }
    summary["numeric_diagnostics"] = diagnostics

    if args.max_probability_delta is not None and diagnostics["max_abs_probability_delta"] > args.max_probability_delta:
        _fail(
            summary,
            f"max absolute probability delta {diagnostics['max_abs_probability_delta']:.6f} "
            f"exceeds explicit integrity cap {args.max_probability_delta:.6f}",
            args.json_out,
        )

    if args.scope == "submission" and diagnostics["records"] < 1000:
        _fail(summary, "submission replay requires at least 1,000 records", args.json_out)

    calibration_evidence = None
    if args.scope == "submission" and not args.calibration_run:
        try:
            calibration_evidence = _load_calibration_evidence(
                args.calibration_json,
                diagnostics["records"],
                args.max_probability_delta,
                args.calibration_margin,
                input_evidence,
                diagnostics["max_abs_probability_delta"],
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            _fail(summary, str(exc), args.json_out)

    auroc_summary = {"status": "not_available", "reason": "unlabeled replay"}
    if args.label_source is not None:
        try:
            labels = load_labels(args.label_source)
        except (OSError, ValueError, KeyError) as exc:
            _fail(summary, str(exc), args.json_out)
        missing_labels = sorted(set(keys).difference(labels.index))
        if missing_labels:
            _fail(
                summary,
                f"label source is missing {len(missing_labels)} prediction keys: {missing_labels[:5]}",
                args.json_out,
            )
        labels = labels.loc[keys]
        y = labels["Cognitive_Impairment"].astype(int).to_numpy()
        ages = labels["Age"].astype(float).to_numpy()
        diagnostics.update(
            {
                "positive_records": int(np.sum(y)),
                "negative_records": int(len(y) - np.sum(y)),
            }
        )
        host_pairs = age_matched_pair_count(y, ages)
        container_pairs = age_matched_pair_count(y, ages)
        auroc_summary = {
            "status": "report_only" if args.scope == "loso" else "diagnostic",
            "host_age_matched_pairs": host_pairs,
            "container_age_matched_pairs": container_pairs,
        }
        summary["auroc_diagnostics"] = auroc_summary
        try:
            host_auroc, host_pairs = age_conditioned_auroc_with_pairs(hp, y, ages)
            container_auroc, container_pairs = age_conditioned_auroc_with_pairs(cp, y, ages)
        except ValueError as exc:
            auroc_summary["status"] = "fail"
            _fail(summary, str(exc), args.json_out)
        auroc_delta = container_auroc - host_auroc
        auroc_summary.update(
            {
                "host_age_conditioned_auroc": host_auroc,
                "container_age_conditioned_auroc": container_auroc,
                "auroc_delta": auroc_delta,
                "diagnostic_tolerance": float(args.max_auroc_delta),
                "auroc_exceeds_diagnostic_tolerance": bool(abs(auroc_delta) > args.max_auroc_delta),
            }
        )
        if args.scope == "submission" and abs(auroc_delta) > args.max_auroc_delta:
            auroc_summary["status"] = "diagnostic_exceeds_configured_tolerance"

    summary["auroc_diagnostics"] = auroc_summary
    summary["release_evidence"] = {
        "primary": "runtime_version_comparison",
        "version_manifests_required": args.scope == "submission",
        "manifest_review_required": bool(version_summary.get("manifest_review_required", False)),
        "known_version_differences": args.known_version_diff,
        "calibration": calibration_evidence,
        "numeric_drift_role": "diagnostic",
        "scope_note": (
            "submission: n>=1000; version manifests are authoritative"
            if args.scope == "submission"
            else (
                "loso: metrics are report-only; explicit probability cap is an opt-in hard integrity check"
                if args.max_probability_delta is not None
                else "loso: metrics are report-only and not a pass/fail gate"
            )
        ),
    }
    diagnostic_flags = [
        diagnostics["p99_exceeds_diagnostic_threshold"],
        bool(auroc_summary.get("auroc_exceeds_diagnostic_tolerance", False)),
    ]
    if args.scope == "loso":
        summary["status"] = "report_only"
        summary["release_eligible"] = False
    elif args.calibration_run:
        summary["status"] = "calibration_only"
        summary["release_eligible"] = False
    elif summary["release_evidence"]["manifest_review_required"]:
        summary["status"] = "manifest_review_required"
        summary["release_eligible"] = False
    else:
        summary["status"] = "pass_with_numeric_diagnostics" if any(diagnostic_flags) else "pass"
        summary["release_eligible"] = True
    _emit(summary, args.json_out)


if __name__ == "__main__":
    main()
