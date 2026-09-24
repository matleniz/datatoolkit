"""Column profiling: semantic types and a per-column summary table."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

SEMANTIC_TYPES = (
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "id_like",
    "constant",
)

# Distinct / non-null ratios: above ID_UNIQUE_RATIO an integer or spaceless string
# column is an identifier (tolerates a few duplicated rows); above TEXT_UNIQUE_RATIO
# a string column is free text rather than categories.
ID_UNIQUE_RATIO = 0.95
TEXT_UNIQUE_RATIO = 0.5
N_SAMPLES = 3


def _parses_as_datetime(values: pd.Series) -> bool:
    parsed = pd.to_datetime(values, errors="coerce", format="ISO8601")
    return bool(parsed.notna().all())


def semantic_type(series: pd.Series) -> str:
    """Classify a column into one of SEMANTIC_TYPES from its dtype and values."""
    values = series.dropna()
    n_unique = values.nunique()
    if n_unique <= 1:
        return "constant"
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    ratio = n_unique / len(values)
    if pdt.is_numeric_dtype(series):
        if n_unique == 2 and set(values.unique()) <= {0, 1}:
            return "boolean"
        if pdt.is_integer_dtype(series) and ratio >= ID_UNIQUE_RATIO:
            return "id_like"
        return "numeric"
    strings = values.astype(str)
    if _parses_as_datetime(strings):
        return "datetime"
    if ratio >= ID_UNIQUE_RATIO and not strings.str.contains(r"\s").any():
        return "id_like"
    if ratio > TEXT_UNIQUE_RATIO:
        return "text"
    return "categorical"


def _samples(series: pd.Series) -> str:
    uniques = series.dropna().unique()[:N_SAMPLES]
    return ", ".join(str(v) for v in uniques)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, semantic type, missing, uniques, sample values."""
    n_rows = len(df)
    rows = []
    for name in df.columns:
        col = df[name]
        n_missing = int(col.isna().sum())
        rows.append(
            {
                "column": str(name),
                "dtype": str(col.dtype),
                "semantic_type": semantic_type(col),
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n_rows, 2) if n_rows else 0.0,
                "n_unique": int(col.nunique()),
                "sample_values": _samples(col),
            }
        )
    columns = [
        "column",
        "dtype",
        "semantic_type",
        "n_missing",
        "pct_missing",
        "n_unique",
        "sample_values",
    ]
    return pd.DataFrame(rows, columns=columns)
