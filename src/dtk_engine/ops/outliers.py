"""Outlier detection: IQR fences, z-scores, multivariate IsolationForest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from dtk_engine.ops.profile import columns_of_type

IQR_K = 1.5
Z_THRESHOLD = 3.0
# Rows sampled in the flagged-rows table.
FLAGGED_SAMPLE = 20

UNIVARIATE_FIELDS = [
    "column",
    "count",
    "lower_fence",
    "upper_fence",
    "n_iqr",
    "pct_iqr",
    "n_z",
    "pct_z",
]

# The course's decision table: what to do with a flagged value, by its cause.
ACTION_TABLE = """\
Outliers: decide by cause, not by the detector.
- Error (typo, sensor fault, sentinel such as -999): remove the row, or set the
  value to missing and impute.
- Legitimate but extreme, and the model is sensitive (linear, distance-based):
  clip (winsorize) at the fences, or transform (log / sqrt / rank).
- Legitimate and informative (fraud, rare events, the very thing to predict): keep.
- Heavy right skew everywhere: transform the column instead of touching rows.
Fit any fence / clip bound on train only and reuse it on test."""


def _pct(count: float, total: float) -> float:
    return round(100 * count / total, 2) if total else 0.0


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns worth screening: numeric semantic type (ids and constants excluded)."""
    return columns_of_type(df, "numeric")


def univariate_outliers(
    df: pd.DataFrame,
    columns: list[str],
    iqr_k: float = IQR_K,
    z_threshold: float = Z_THRESHOLD,
) -> pd.DataFrame:
    """Per column: IQR fences, count / % outside them, count / % with |z| > threshold
    (percentages over non-null values)."""
    rows = []
    for col in columns:
        values = df[col].dropna().astype(float)
        n = len(values)
        lo = hi = float("nan")
        n_iqr = n_z = 0
        if n:
            q1, q3 = values.quantile([0.25, 0.75])
            lo, hi = q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)
            n_iqr = int(((values < lo) | (values > hi)).sum())
            std = values.std()
            if n > 1 and std > 0:
                n_z = int((((values - values.mean()) / std).abs() > z_threshold).sum())
        rows.append((col, n, lo, hi, n_iqr, _pct(n_iqr, n), n_z, _pct(n_z, n)))
    return pd.DataFrame(rows, columns=UNIVARIATE_FIELDS)


def isolation_forest(
    df: pd.DataFrame,
    columns: list[str],
    contamination: float,
    random_state: int,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Multivariate anomaly scores: a frame indexed like ``df`` with ``score``
    (lower = more anomalous) and ``flagged`` (the ``contamination`` share of
    rows). Missing values are median-filled; returns an empty frame if there is
    nothing to fit."""
    cols = [c for c in columns if df[c].notna().any()]
    if not cols or len(df) < 2:
        return pd.DataFrame({"score": [], "flagged": []}, dtype=float)
    X = df[cols].astype(float)
    X = X.fillna(X.median())
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    ).fit(X)
    return pd.DataFrame(
        {
            "score": model.score_samples(X).round(4),
            "flagged": model.predict(X) == -1,
        },
        index=df.index,
    )


def flagged_rows(
    df: pd.DataFrame, scores: pd.DataFrame, columns: list[str], n: int = FLAGGED_SAMPLE
) -> pd.DataFrame:
    """The ``n`` most anomalous flagged rows (row index, score, then the columns)."""
    if scores.empty:
        return pd.DataFrame(columns=["row", "score", *columns])
    worst = scores[scores["flagged"]].sort_values("score").head(n)
    out = df.loc[worst.index, columns].copy()
    out.insert(0, "score", worst["score"])
    out.insert(0, "row", np.asarray(worst.index))
    return out.reset_index(drop=True)
