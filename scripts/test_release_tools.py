"""Regression tests for runtime provenance and replay evidence binding."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_entry_replay import _load_calibration_evidence, compare_runtime_versions  # noqa: E402


def _runtime_payload(torch_ecg_commit: str | None) -> dict:
    return {
        "packages": {"torch-ecg": "0.0.33"},
        "vcs_commits": {"torch-ecg": torch_ecg_commit},
    }


class RuntimeVersionTests(unittest.TestCase):
    def test_unresolved_torch_ecg_commit_fails_closed(self) -> None:
        summary = compare_runtime_versions(_runtime_payload(None), _runtime_payload(None), ["vcs_commits"])
        self.assertTrue(summary["manifest_review_required"])
        self.assertEqual([item["key"] for item in summary["unknown_differences"]], ["vcs_commits.torch-ecg"])

    def test_different_torch_ecg_commits_fail_closed(self) -> None:
        summary = compare_runtime_versions(_runtime_payload("a" * 40), _runtime_payload("b" * 40), ["vcs_commits"])
        self.assertTrue(summary["manifest_review_required"])

    def test_matching_torch_ecg_commit_is_accepted(self) -> None:
        summary = compare_runtime_versions(_runtime_payload("a" * 40), _runtime_payload("a" * 40), ["vcs_commits"])
        self.assertFalse(summary["manifest_review_required"])


class CalibrationBindingTests(unittest.TestCase):
    input_evidence = {
        "host_predictions_sha256": "a" * 64,
        "container_predictions_sha256": "b" * 64,
        "host_versions_sha256": "c" * 64,
        "container_versions_sha256": "d" * 64,
        "labels_sha256": "e" * 64,
        "record_keys_sha256": "f" * 64,
    }
    residual = 0.00351831316947937

    def _write_artifact(self, directory: Path, residual: float, input_evidence: dict | None = None) -> Path:
        path = directory / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "status": "calibration_only",
                    "scope": "submission",
                    "numeric_diagnostics": {
                        "records": 1103,
                        "max_abs_probability_delta": residual,
                    },
                    "input_evidence": input_evidence or self.input_evidence,
                }
            )
        )
        return path

    def test_forged_residual_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write_artifact(Path(directory), residual=0.0)
            with self.assertRaisesRegex(ValueError, "does not match recomputed residual"):
                _load_calibration_evidence(artifact, 1103, 0.005, 1.2, self.input_evidence, self.residual)

    def test_input_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stale_inputs = dict(self.input_evidence, container_versions_sha256="0" * 64)
            artifact = self._write_artifact(Path(directory), self.residual, stale_inputs)
            with self.assertRaisesRegex(ValueError, "calibration inputs do not match"):
                _load_calibration_evidence(artifact, 1103, 0.005, 1.2, self.input_evidence, self.residual)

    def test_bound_reproducible_residual_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write_artifact(Path(directory), self.residual)
            evidence = _load_calibration_evidence(artifact, 1103, 0.005, 1.2, self.input_evidence, self.residual)
        self.assertEqual(evidence["recomputed_residual_max_abs_probability_delta"], self.residual)


if __name__ == "__main__":
    unittest.main()
