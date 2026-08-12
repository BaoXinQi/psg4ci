#!/usr/bin/env python3
"""Run two real records through the official loaded-model inference path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import team_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(team_code._demographics_path(args.data))
    bmi = pd.to_numeric(frame.get("BMI"), errors="coerce")
    selected = []
    for label, keep in (("complete_bmi", bmi.notna()), ("missing_bmi", bmi.isna())):
        candidates = frame.loc[keep]
        if candidates.empty:
            continue
        selected.append((label, candidates.iloc[0]))
    if len(selected) < 2:
        raise RuntimeError("Expected both complete-BMI and missing-BMI records")

    model = team_code.load_model(args.model, verbose=False)
    output = []
    for label, row in selected:
        binary, probability = team_code.run_model(
            model, row, args.data, verbose=True
        )
        metadata_adjustment = team_code._demographics_adjustment(
            model["demographics_rule"], row
        )
        output.append(
            {
                "case": label,
                "record_id": str(row.get("BidsFolder")),
                "site": str(row.get("SiteID")),
                "session": None if pd.isna(row.get("SessionID")) else int(row.get("SessionID")),
                "bmi_available": bool(pd.notna(row.get("BMI"))),
                "metadata_adjustment": float(metadata_adjustment),
                "binary": bool(binary),
                "probability": float(probability),
            }
        )
    print(json.dumps({"status": "ok", "records": output}, indent=2))


if __name__ == "__main__":
    main()
