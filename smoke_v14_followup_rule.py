#!/usr/bin/env python3
"""Verify that the packaged V14 follow-up rule reproduces from Large training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import team_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.manifest)
    fitted = team_code._fit_followup_rule(frame)
    packaged = json.loads(
        (team_code.PRETRAINED_DIR / team_code.FOLLOWUP_RULE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    if fitted["training_site_cutoffs"] != packaged["training_site_cutoffs"]:
        raise AssertionError("Training-site cutoffs do not match the packaged rule")
    if abs(fitted["risk_coefficient"] - packaged["risk_coefficient"]) > 1e-9:
        raise AssertionError("Follow-up coefficient does not match the packaged rule")
    if team_code._followup_adjustment(fitted, {"CreationTime": None}) != 0.0:
        raise AssertionError("Missing CreationTime did not produce zero adjustment")
    print(
        json.dumps(
            {
                "status": "ok",
                "risk_coefficient": fitted["risk_coefficient"],
                "training_site_cutoffs": fitted["training_site_cutoffs"],
                "missing_creation_time_adjustment": 0.0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
