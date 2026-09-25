"""Numeric recommendations: skew (log1p) and scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtk_engine.ops.advisor.common import ColumnInfo, Rec
from dtk_engine.ops.outliers import univariate_outliers

# Course "Scaling and Normalization", who needs it: trees split on order (no);
# regularised linear models (penalty on raw coefficients), distance-based
# models and neural networks (yes).
SCALING_NEEDED = {"tree": False, "linear": True, "distance": True, "neural": True}
# Right skew above this (and no negative value) -> log1p.
SKEW_THRESHOLD = 1.0
# More IQR outliers than this (% of non-null values, after log1p) -> robust scaler.
ROBUST_OUTLIER_PCT = 1.0


def numeric_recs(
    col: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    family: str | None,
    info: ColumnInfo,
) -> list[Rec]:
    recs = []
    values = train[col].dropna().astype(float)
    if len(values) < 3:
        return recs
    skew = float(values.skew())
    info.skew = round(skew, 3) if not np.isnan(skew) else None
    mins = [values.min()]
    if test is not None and col in test.columns and test[col].notna().any():
        mins.append(float(test[col].min()))
    screened = train[[col]].astype(float)
    if family != "tree" and skew > SKEW_THRESHOLD and min(mins) >= 0:
        recs.append(
            Rec(
                col,
                "skew",
                "info",
                f"right-skewed (skew {skew:.2f}, no negative value): log1p compresses "
                "the tail; scaling alone would not reshape it",
                "log1p",
                "both",
                {"columns": [col]},
            )
        )
        screened = np.log1p(screened)
    outliers = univariate_outliers(screened, [col])
    pct_out = float(outliers["pct_iqr"].iloc[0])
    info.pct_outliers_iqr = pct_out
    needed = SCALING_NEEDED.get(family) if family else None
    if needed is False:
        return recs
    method = "robust" if pct_out > ROBUST_OUTLIER_PCT else "standard"
    why = (
        f"robust scaler (median / IQR): {pct_out}% IQR outliers, kept on purpose, "
        "would move a mean and a std"
        if method == "robust"
        else "standard scaler (mean 0, std 1)"
    )
    if needed:
        advice = f"{family} models need scaled inputs: {why}, fitted on train"
    else:
        advice = (
            f"scale for linear (regularised), distance-based or neural models, "
            f"not for trees: {why}"
        )
    recs.append(
        Rec(
            col,
            "scaling",
            "info",
            advice,
            "scale",
            "both",
            {"columns": [col], "method": method},
        )
    )
    return recs
