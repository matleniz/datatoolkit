"""Imputation benchmark: hide known values, fill them back, measure the error.

For each column, a seeded fraction of its *known* values is masked; each
strategy then fills the masked frame through the ``impute`` transform itself
(``ops/transforms/impute.py``: one implementation of every strategy, fitted on
the masked frame so a held-out value never leaks into a learned statistic).
Every strategy of a column is scored on the same masked rows. Coverage (share
of masked cells the strategy could fill) is reported apart from the error,
which is computed on the filled cells only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtk_engine.ops._util import require_numeric
from dtk_engine.transform_registry import get_transform

COLUMNS = [
    "column",
    "strategy",
    "n_masked",
    "n_filled",
    "coverage",
    "rmse",
    "mae",
    "median_abs_error",
]


def mask_rows(df: pd.DataFrame, column: str, fraction: float, seed: int) -> np.ndarray:
    """Sorted positions of the known values of ``column`` to hide (seeded).

    At least one, never all of them (a strategy needs something to learn from).
    """
    known = np.flatnonzero(df[column].notna().to_numpy())
    if len(known) < 2:
        raise ValueError(
            f"impute_benchmark: column {column!r} has fewer than 2 known values"
        )
    n = min(max(1, round(fraction * len(known))), len(known) - 1)
    return np.sort(np.random.default_rng(seed).choice(known, size=n, replace=False))


def _score(column: str, strategy: str, truth: np.ndarray, filled: np.ndarray) -> dict:
    ok = ~np.isnan(filled)
    err = filled[ok] - truth[ok]
    row = {
        "column": column,
        "strategy": strategy,
        "n_masked": len(truth),
        "n_filled": int(ok.sum()),
        "coverage": round(float(ok.mean()), 4),
        "rmse": None,
        "mae": None,
        "median_abs_error": None,
    }
    if len(err):
        row["rmse"] = float(np.sqrt(np.mean(err**2)))
        row["mae"] = float(np.mean(np.abs(err)))
        row["median_abs_error"] = float(np.median(np.abs(err)))
    return row


def benchmark(
    df: pd.DataFrame,
    columns: list[str],
    strategies: list[str],
    by: str | None,
    order: str | None,
    mask_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """One row per column x strategy (see module doc); no fallback fill."""
    require_numeric(df, columns, "impute_benchmark")
    impute = get_transform("impute")
    rows = []
    for col in columns:
        pos = mask_rows(df, col, mask_fraction, seed)
        truth = df[col].to_numpy(dtype=float)[pos]
        masked = df.copy()
        masked.iloc[pos, masked.columns.get_loc(col)] = np.nan
        for strategy in strategies:
            params = impute.parse(
                {"columns": [col], "strategy": strategy, "by": by, "order": order}
            )
            out = impute.fit_apply(masked, params)
            filled = out[col].to_numpy(dtype=float)[pos]
            rows.append(_score(col, strategy, truth, filled))
    return pd.DataFrame(rows, columns=COLUMNS)


def best_per_column(table: pd.DataFrame) -> pd.DataFrame:
    """The lowest-RMSE strategy of each column (ties: first listed); columns
    no strategy could fill are absent."""
    scored = table.dropna(subset=["rmse"])
    return scored.loc[scored.groupby("column", sort=False)["rmse"].idxmin()]
