#!/usr/bin/env python3
"""Check metadata priority, EDF fallback, and double-missing behavior."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from pyedflib import highlevel

import team_code
from raw_sequence_runtime import load_runtime


def _check_synthetic_short_record() -> float:
    with tempfile.TemporaryDirectory() as temporary:
        data_root = Path(temporary)
        psg_dir = data_root / "physiological_data" / "I0004"
        psg_dir.mkdir(parents=True)
        psg_path = psg_dir / "sub-ci-smoke_ses-1.edf"
        signal_headers = highlevel.make_signal_headers(["EEG"], sample_frequency=1)
        header = highlevel.make_header(startdate=datetime(2020, 1, 2, 3, 4, 5))
        highlevel.write_edf(
            str(psg_path), [np.zeros(1, dtype=float)], signal_headers, header
        )

        runtime = load_runtime(team_code.PRETRAINED_DIR, threads=1)
        runtime["date_rule"] = json.loads(
            (team_code.PRETRAINED_DIR / team_code.DATE_RULE_FILENAME).read_text()
        )
        runtime["caisr_rule"] = json.loads(
            (team_code.PRETRAINED_DIR / team_code.CAISR_RULE_FILENAME).read_text()
        )
        runtime["followup_rule"] = json.loads(
            (team_code.PRETRAINED_DIR / team_code.FOLLOWUP_RULE_FILENAME).read_text()
        )
        binary, probability = team_code.run_model(
            runtime,
            {
                "BidsFolder": "sub-ci-smoke",
                "SiteID": "I0004",
                "SessionID": 1,
                "CreationTime": None,
            },
            data_root,
            False,
        )
        if not isinstance(binary, bool) or not math.isfinite(probability):
            raise AssertionError("Short-record fallback did not return a finite prediction")
        return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--psg", type=Path, required=True)
    parser.add_argument("--synthetic-short", action="store_true")
    args = parser.parse_args()

    date_rule = json.loads(
        (team_code.PRETRAINED_DIR / team_code.DATE_RULE_FILENAME).read_text()
    )
    followup_rule = json.loads(
        (team_code.PRETRAINED_DIR / team_code.FOLLOWUP_RULE_FILENAME).read_text()
    )
    header_time = team_code._edf_creation_time(args.psg)
    if header_time is None:
        raise AssertionError("The EDF header did not provide a valid start time")

    value, source = team_code._runtime_creation_time(
        {"CreationTime": None}, args.psg
    )
    if source != "edf-header":
        raise AssertionError(f"Unexpected fallback source: {source}")
    fallback_date = team_code._date_adjustment(
        date_rule, {"CreationTime": value}
    )
    fallback_followup = team_code._followup_adjustment(
        followup_rule, {"CreationTime": value}
    )
    expected_date = team_code._date_adjustment(
        date_rule, {"CreationTime": header_time}
    )
    expected_followup = team_code._followup_adjustment(
        followup_rule, {"CreationTime": header_time}
    )
    if not np.isclose(fallback_date, expected_date, atol=1e-12):
        raise AssertionError("Date fallback does not equal the EDF header")
    if not np.isclose(fallback_followup, expected_followup, atol=1e-12):
        raise AssertionError("Follow-up fallback does not equal the EDF header")

    explicit = "2001-01-02 03:04:05"
    value, source = team_code._runtime_creation_time(
        {"CreationTime": explicit}, args.psg
    )
    if source != "metadata" or str(value) != explicit:
        raise AssertionError("Valid current-record metadata did not take priority")

    value, source = team_code._runtime_creation_time(
        {"CreationTime": "not-a-date"}, args.psg
    )
    if source != "edf-header" or value is None:
        raise AssertionError("Malformed metadata did not fall back to the EDF header")

    absent = args.psg.with_name("does-not-exist.edf")
    value, source = team_code._runtime_creation_time(
        {"CreationTime": None}, absent
    )
    if source != "missing" or value is not None:
        raise AssertionError("Double-missing input did not fail closed")

    synthetic_probability = (
        _check_synthetic_short_record() if args.synthetic_short else None
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "edf_start_datetime": str(header_time),
                "fallback_date_adjustment": fallback_date,
                "fallback_followup_adjustment": fallback_followup,
                "valid_metadata_has_priority": True,
                "malformed_metadata_uses_edf": True,
                "double_missing_returns_zero": True,
                "full_demographics_scan_used": False,
                "synthetic_short_probability": synthetic_probability,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
