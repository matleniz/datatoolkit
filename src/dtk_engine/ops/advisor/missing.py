"""Imputation recommendation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dtk_engine.ops.advisor.common import ColumnInfo, Rec
from dtk_engine.ops.missing import REVIEW_PCT
from dtk_engine.ops.transforms.impute import INDICATOR_SUFFIX


def missing_rec(
    col: str,
    semantic: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    info: ColumnInfo,
) -> Rec | None:
    test_missing = test is not None and col in test.columns and test[col].isna().any()
    if semantic == "datetime" or not (train[col].isna().any() or test_missing):
        return None  # datetime: datetime_parts keeps NaT as missing parts
    pct = info.pct_missing
    numeric = semantic == "numeric"
    if numeric:
        strategy, why = "median", "median (robust to skew and outliers)"
    elif pct >= REVIEW_PCT:
        strategy, why = "constant", 'a "MISSING" category (missingness kept as a value)'
    else:
        strategy, why = "most_frequent", "the most frequent value"
    indicator = pct >= REVIEW_PCT and numeric
    advice = f"{pct}% missing in train: impute with {why}, fitted on train"
    if indicator:
        advice += (
            f", plus a {col}{INDICATOR_SUFFIX} flag (missingness may be predictive)"
        )
    if not pct:
        advice = f"missing only in test: impute with {why} learned on train"
    params: dict[str, Any] = {"columns": [col], "strategy": strategy}
    if indicator:
        params["add_indicator"] = True
    return Rec(col, "missing", "info", advice, "impute", "both", params)
