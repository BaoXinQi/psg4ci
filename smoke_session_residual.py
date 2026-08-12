#!/usr/bin/env python3
"""Verify the joint record-wise demographics residual and field fallbacks."""

from __future__ import annotations

import json

import numpy as np

import team_code


def main() -> None:
    rule = json.loads(
        (team_code.PRETRAINED_DIR / team_code.DEMOGRAPHICS_RULE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    missing_values = [None, "", "unavailable", np.nan]
    for value in missing_values:
        adjustment = team_code._demographics_adjustment(
            rule, {"SessionID": value, "Age": value, "BMI": value, "Sex": value}
        )
        if adjustment != 0.0:
            raise AssertionError(f"Missing fields changed V14: {value!r}")
    values = {
        session: team_code._demographics_adjustment(rule, {"SessionID": session})
        for session in (1, 2, 3, 6)
    }
    if not values[1] < values[2] < values[3] < values[6]:
        raise AssertionError(f"Session residual is not monotone: {values}")

    missing_metadata = [
        {},
        {"Age": None, "BMI": "", "Sex": np.nan},
        {"Age": "unavailable", "BMI": np.nan, "Sex": "unknown"},
    ]
    for row in missing_metadata:
        adjustment = team_code._demographics_adjustment(rule, row)
        if adjustment != 0.0:
            raise AssertionError(f"Missing metadata changed V14: {row!r}")
    female_mean = team_code._demographics_adjustment(
        rule,
        {
            "Age": rule["age"]["mean"],
            "BMI": rule["bmi"]["mean"],
            "Sex": "Female",
        },
    )
    male_mean = team_code._demographics_adjustment(
        rule,
        {
            "Age": rule["age"]["mean"],
            "BMI": rule["bmi"]["mean"],
            "Sex": " male ",
        },
    )
    if female_mean != 0.0:
        raise AssertionError(f"Centered female reference must be zero: {female_mean}")
    if not np.isclose(male_mean, rule["sex"]["male_coefficient"]):
        raise AssertionError(f"Male coefficient mismatch: {male_mean}")
    if not np.isfinite(
        team_code._demographics_adjustment(
            rule, {"Age": 70, "BMI": 28.0, "Sex": "Male"}
        )
    ):
        raise AssertionError("Complete metadata produced a non-finite adjustment")
    print(
        json.dumps(
            {
                "status": "ok",
                "missing_fallback_exact": True,
                "session_adjustments": values,
                "joint_demographics_missing_fallback_exact": True,
                "male_mean_adjustment": male_mean,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
