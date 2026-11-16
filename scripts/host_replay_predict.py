"""Produce host-side per-record predictions for the replay gate.

The official ``run_model.py`` cannot finish its output-table serialization on
hosts with pandas 3.x because the organiser's unmodifiable helper writes integer
predictions into the original boolean label column.  The submission image pins
``pandas<3`` and remains the authoritative end-to-end environment.

This script exercises the same model entry path without calling the organiser's
output-table helper: ``team_code.train_model`` → ``load_model`` → ``run_model``.
The CSV has the keys and probability column expected by
``scripts/compare_entry_replay.py``; ``Cognitive_Impairment`` is the binary
prediction, while ``true_label`` preserves the original label for diagnostics.

Two-stage usage: first train with ``--data-dir <training partition>``, then
produce predictions from the unseen partition with ``--no-train --data-dir
<holdout partition> --model-out <same model directory>``.  The single-dir
default is therefore an in-sample smoke test, not a cache-miss replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--no-train", action="store_true", help="reuse an existing model_out")
    args = parser.parse_args()

    import team_code

    data_dir = team_code._resolve_db_dir(str(args.data_dir.expanduser().resolve()))
    model_dir = args.model_out.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_train:
        print(f"[host] training via team_code.train_model ({data_dir})")
        team_code.train_model(str(data_dir), str(model_dir), verbose=True)

    model_dict = team_code.load_model(str(model_dir), verbose=True)
    demo = pd.read_csv(data_dir / "demographics.csv")
    rows = []
    for row in demo.itertuples():
        record = {
            "BidsFolder": row.BidsFolder,
            "SiteID": row.SiteID,
            "SessionID": row.SessionID,
        }
        binary, probability = team_code.run_model(model_dict, record, str(data_dir), verbose=False)
        rows.append(
            {
                "BidsFolder": row.BidsFolder,
                "SessionID": row.SessionID,
                "SiteID": row.SiteID,
                "Cognitive_Impairment": int(binary),
                "Cognitive_Impairment_Probability": float(probability),
                "true_label": (
                    int(row.Cognitive_Impairment)
                    if "Cognitive_Impairment" in demo.columns and pd.notna(row.Cognitive_Impairment)
                    else float("nan")
                ),
                "Age": float(row.Age) if "Age" in demo.columns and pd.notna(row.Age) else float("nan"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("host replay produced no records")
    probs = frame["Cognitive_Impairment_Probability"].astype(float).to_numpy()
    if not np.isfinite(probs).all() or np.any((probs < 0.0) | (probs > 1.0)):
        raise RuntimeError("host replay produced non-finite or out-of-range probabilities")
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.csv_out, index=False)
    summary = {
        "records": int(len(frame)),
        "prob_min": float(frame["Cognitive_Impairment_Probability"].min()),
        "prob_max": float(frame["Cognitive_Impairment_Probability"].max()),
        "binary_positive_rate": float(frame["Cognitive_Impairment"].mean()),
        "model_dir": str(model_dir),
    }
    print("[host] " + json.dumps(summary, sort_keys=True))
    print(f"[host] wrote {args.csv_out}")


if __name__ == "__main__":
    main()
