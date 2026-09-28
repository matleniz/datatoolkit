"""Cleaning recommendations: sentinels, spelling variants, numbers stored as text."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops._util import py as _py
from dtk_engine.ops.advisor.common import Rec
from dtk_engine.ops.consistency import variants
from dtk_engine.ops.missing import (
    DATE_SENTINELS,
    NUMERIC_SENTINELS,
    STRING_SENTINELS,
    sentinel_counts,
)
from dtk_engine.ops.profile import numeric_text_format, pct_numeric_parsable

# Hits of the "implausible 0" sentinel heuristic are only reported, not replaced.
UNSAFE_SENTINELS = frozenset({"0"})


def _sentinel_values(series: pd.Series) -> list:
    """Raw cell values ``sentinel_counts`` flags in ``series`` (same rules, per value).

    A numeric sentinel (-1, 999, ...) counts only outside the range of the other
    values: -1 in a column of temperatures is data, -999 in a column of ages is not.
    """
    if pdt.is_datetime64_any_dtype(series) or pdt.is_bool_dtype(series):
        return []  # replace_sentinels takes scalars, not timestamps
    table = sentinel_counts(series.to_frame("v"))
    if not (set(table["sentinel"]) - UNSAFE_SENTINELS):
        return []
    values = series.dropna()
    if pdt.is_numeric_dtype(series):
        codes = [float(h) for h in table["sentinel"] if h not in UNSAFE_SENTINELS]
        others = values[~values.isin(codes)]
        return sorted(
            _py(v)
            for v in values[values.isin(codes)].unique()
            if others.empty or v < others.min() or v > others.max()
        )
    # Text: the rules of ops.missing, vectorised over the distinct values.
    uniques = pd.Series(values.unique())
    text = uniques.astype(str).str.strip()
    hit = (
        text.str.lower().isin(STRING_SENTINELS)
        | text.str.startswith(DATE_SENTINELS)
        | pd.to_numeric(text, errors="coerce").isin(NUMERIC_SENTINELS)
    )
    return sorted((_py(v) for v in uniques[hit]), key=str)


def sentinel_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    values: list = []
    for df in frames:
        if col not in df.columns:
            continue
        for v in _sentinel_values(df[col]):
            if v not in values:
                values.append(v)
    if not values:
        return None
    return Rec(
        col,
        "sentinels",
        "warning",
        f"disguised missing values {values}: turn them into NaN before imputing "
        "(else they skew the statistics and the model learns the code)",
        "replace_sentinels",
        "both",
        {"sentinels": {col: values}},
    )


def variant_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    series = pd.concat([df[col] for df in frames if col in df.columns])
    _, mapping = variants(series.to_frame(col), [col])
    pairs = {
        str(r.variant).strip(): str(r.canonical).strip()
        for r in mapping.itertuples()
        if str(r.variant).strip() != str(r.canonical).strip()
    }
    if mapping.empty:
        return None
    return Rec(
        col,
        "consistency",
        "warning",
        f"spelling variants of the same value ({len(mapping)} forms): merge them "
        "before encoding, else each becomes its own category",
        "standardize_text",
        "both",
        {"columns": [col], "strip": True, "mapping": pairs},
    )


def type_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    present = [df[col] for df in frames if col in df.columns]
    if all(pdt.is_numeric_dtype(s) for s in present):
        return None
    if any(pdt.is_bool_dtype(s) or pdt.is_datetime64_any_dtype(s) for s in present):
        return None
    if not any(s.notna().any() for s in present):
        return None
    for s in present:
        fmt = numeric_text_format(s) if s.notna().any() else None
        if fmt is not None:
            return Rec(
                col,
                "type",
                "warning",
                "numbers stored as text with a currency symbol, percent sign, "
                f"or thousands separator: parse to float ({fmt})",
                "to_numeric",
                "both",
                {"columns": [col], **fmt},
            )
    if any(s.notna().any() and pct_numeric_parsable(s) < 100 for s in present):
        return None
    return Rec(
        col,
        "type",
        "warning",
        "numbers stored as text (on at least one side): cast to float so train "
        "and test share a numeric type",
        "cast",
        "both",
        {"dtypes": {col: "float64"}},
    )
