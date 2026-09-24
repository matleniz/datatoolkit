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
    "group_id",
    "constant",
)

# Distinct / non-null ratios: above ID_UNIQUE_RATIO an integer or spaceless string
# column is an identifier (tolerates a few duplicated rows); above TEXT_UNIQUE_RATIO
# a string column is free text rather than categories.
ID_UNIQUE_RATIO = 0.95
TEXT_UNIQUE_RATIO = 0.5
# Fewer non-null values than this: too little evidence to call a column id_like
# (a sparse column with a handful of distinct values is all-distinct by chance).
ID_MIN_NON_NULL = 20
# An integer column is identifier-shaped when its distinct values fill their
# [min, max] range densely (e.g. an `Index` 0..n-1); a sparse spread of distinct
# integers is a numeric feature.
ID_RANGE_DENSITY = 0.95
# group_id = entity key repeated over rows (e.g. `patient_id`): at least
# GROUP_MIN_UNIQUE distinct values with a distinct/non-null ratio in
# [GROUP_MIN_RATIO, ID_UNIQUE_RATIO). Strings must be whitespace-free; integers must
# fill at least GROUP_RANGE_DENSITY of their range (else they are a numeric feature).
GROUP_MIN_UNIQUE = 100
GROUP_MIN_RATIO = 0.01
GROUP_RANGE_DENSITY = 0.5
N_SAMPLES = 3


def _parses_as_datetime(strings: pd.Series) -> bool:
    # Pure digit strings ("2024", "7") are years / codes, not dates.
    if strings.str.fullmatch(r"\d+").all():
        return False
    parsed = pd.to_datetime(strings, errors="coerce", format="ISO8601")
    return bool(parsed.notna().all())


def _range_density(values: pd.Series, n_unique: int) -> float:
    span = int(values.max()) - int(values.min()) + 1
    return n_unique / span


def _group_band(n_unique: int, ratio: float) -> bool:
    return n_unique >= GROUP_MIN_UNIQUE and GROUP_MIN_RATIO <= ratio < ID_UNIQUE_RATIO


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
    enough = len(values) >= ID_MIN_NON_NULL
    if pdt.is_numeric_dtype(series):
        if n_unique == 2 and set(values.unique()) <= {0, 1}:
            return "boolean"
        if pdt.is_integer_dtype(series):
            density = _range_density(values, n_unique)
            if enough and ratio >= ID_UNIQUE_RATIO and density >= ID_RANGE_DENSITY:
                return "id_like"
            if _group_band(n_unique, ratio) and density >= GROUP_RANGE_DENSITY:
                return "group_id"
        return "numeric"
    strings = values.astype(str)
    if _parses_as_datetime(strings):
        return "datetime"
    spaceless = not strings.str.contains(r"\s").any()
    if enough and ratio >= ID_UNIQUE_RATIO and spaceless:
        return "id_like"
    if spaceless and _group_band(n_unique, ratio):
        return "group_id"
    if ratio > TEXT_UNIQUE_RATIO:
        return "text"
    return "categorical"


def pct_numeric_parsable(series: pd.Series) -> float:
    """% of non-null values that read as numbers (100 for a numeric dtype).

    A text column scoring high here is numbers polluted by a few tokens
    (e.g. "unknown" in an age column).
    """
    values = series.dropna()
    if values.empty or pdt.is_bool_dtype(series):
        return 0.0
    if pdt.is_numeric_dtype(series):
        return 100.0
    if pdt.is_datetime64_any_dtype(series):
        return 0.0
    parsed = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return round(100 * float(parsed.notna().mean()), 2)


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
                "pct_numeric_parsable": pct_numeric_parsable(col),
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
        "pct_numeric_parsable",
        "sample_values",
    ]
    return pd.DataFrame(rows, columns=columns)
