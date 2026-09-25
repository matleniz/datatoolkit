"""Correlation matrix of numeric columns and the highly correlated pairs.

Pure pandas, no ``Result``. Pairs use the same rule as the ``drop_correlated``
op (|corr| >= threshold; with a target, the column more correlated with it is
the one kept), so the analysis says what that op would act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtk_engine.ops.selection import COLLINEAR_CORR

METHODS = ("pearson", "spearman")
DEFAULT_THRESHOLD = COLLINEAR_CORR
PAIR_FIELDS = ["a", "b", "corr", "abs_corr", "n_rows", "keep"]


def correlation_matrix(
    df: pd.DataFrame, columns: list[str], method: str
) -> pd.DataFrame:
    """Pairwise-complete correlation matrix (a constant column gives NaN)."""
    return df[columns].astype(float).corr(method=method)


def off_diagonal(corr: pd.DataFrame) -> pd.DataFrame:
    """The matrix with its diagonal (self-correlations) set to NaN."""
    return corr.where(~np.eye(len(corr), dtype=bool))


def undefined_columns(corr: pd.DataFrame) -> list[str]:
    """Columns correlated with nothing (constant, or no overlapping values)."""
    if len(corr) < 2:
        return []
    off = off_diagonal(corr)
    return [str(c) for c in corr.columns if off[c].isna().all()]


def target_correlation(
    df: pd.DataFrame, columns: list[str], target: str, method: str
) -> pd.Series:
    """|corr| of each column with the target (text target: sorted class codes,
    as ``drop_correlated`` does)."""
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        codes = pd.Series(pd.factorize(y, sort=True)[0], index=df.index, dtype=float)
        y = codes.where(df[target].notna())
    with np.errstate(divide="ignore", invalid="ignore"):
        out = df[columns].astype(float).corrwith(y.astype(float), method=method)
    return out.abs().fillna(0.0)


def correlated_pairs(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    to_target: pd.Series | None = None,
) -> pd.DataFrame:
    """Pairs with |corr| >= threshold, strongest first; ``n_rows`` = rows where
    both are present; ``keep`` = the one more correlated with the target (ties
    and no target: the first in column order)."""
    columns = list(corr.columns)
    present = df[columns].notna()
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = corr.loc[a, b]
            if pd.isna(value) or abs(value) < threshold:
                continue
            keep = a
            if to_target is not None and to_target[b] > to_target[a]:
                keep = b
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "corr": float(value),
                    "abs_corr": abs(float(value)),
                    "n_rows": int((present[a] & present[b]).sum()),
                    "keep": keep,
                }
            )
    out = pd.DataFrame(rows, columns=PAIR_FIELDS)
    return out.sort_values(
        ["abs_corr", "a", "b"], ascending=[False, True, True]
    ).reset_index(drop=True)
