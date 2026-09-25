"""Inconsistencies in text / categorical columns: variants, mixed types, dates."""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

VARIANT_COLUMNS = ["column", "canonical", "variant", "count"]
SUMMARY_COLUMNS = ["column", "distinct_before", "distinct_after", "n_merged"]
MIXED_COLUMNS = [
    "column",
    "n_numbers",
    "n_strings",
    "example_numbers",
    "example_strings",
]
DATE_COLUMNS = ["column", "n_ambiguous", "n_parsed", "examples"]
EXAMPLES = 3


def normalize(value: str) -> str:
    """The merge key: strip, collapse inner whitespace, lower-case."""
    return " ".join(value.split()).lower()


def text_columns(df: pd.DataFrame) -> list[str]:
    """Object / string columns worth checking for variants."""
    return [
        c
        for c in df.columns
        if pdt.is_object_dtype(df[c]) or pdt.is_string_dtype(df[c])
    ]


def _strings(series: pd.Series) -> pd.Series:
    return series[series.map(lambda v: isinstance(v, str))]


def variants(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spelling variants that merge after normalisation.

    Returns (summary: distinct before / after per affected column, mapping:
    variant -> canonical, the most frequent form of each merged set).
    """
    summary, mapping = [], []
    for col in columns:
        counts = _strings(df[col]).value_counts()
        if counts.empty:
            continue
        keys = pd.Series([normalize(v) for v in counts.index], index=counts.index)
        n_after = int(keys.nunique())
        if n_after == len(counts):
            continue
        summary.append(
            {
                "column": col,
                "distinct_before": len(counts),
                "distinct_after": n_after,
                "n_merged": len(counts) - n_after,
            }
        )
        for _, forms in keys.groupby(keys, sort=False):
            if len(forms) < 2:
                continue
            # value_counts is sorted by count desc: the first form is canonical.
            canonical = forms.index[0]
            for form in forms.index:
                mapping.append(
                    {
                        "column": col,
                        "canonical": canonical,
                        "variant": form,
                        "count": int(counts[form]),
                    }
                )
    return (
        pd.DataFrame(summary, columns=SUMMARY_COLUMNS),
        pd.DataFrame(mapping, columns=VARIANT_COLUMNS),
    )


def mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns holding both numbers and strings."""
    rows = []
    for col in df.columns:
        if not pdt.is_object_dtype(df[col]):
            continue
        values = df[col].dropna()
        is_num = values.map(
            lambda v: isinstance(v, int | float) and not isinstance(v, bool)
        )
        is_str = values.map(lambda v: isinstance(v, str))
        if is_num.any() and is_str.any():
            rows.append(
                {
                    "column": col,
                    "n_numbers": int(is_num.sum()),
                    "n_strings": int(is_str.sum()),
                    "example_numbers": ", ".join(
                        map(str, values[is_num].unique()[:EXAMPLES])
                    ),
                    "example_strings": ", ".join(
                        map(str, values[is_str].unique()[:EXAMPLES])
                    ),
                }
            )
    return pd.DataFrame(rows, columns=MIXED_COLUMNS)


_DATE_PATTERN = r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$"


def ambiguous_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """String columns with dd/mm vs mm/dd ambiguous dates (both parts <= 12,
    different), e.g. ``01/02/2023``. Only columns where most values look like
    day-first-or-month-first dates are reported."""
    rows = []
    for col in columns:
        strings = _strings(df[col]).astype(str)
        if strings.empty:
            continue
        parts = strings.str.extract(_DATE_PATTERN).dropna().astype(int)
        if len(parts) < 0.5 * len(strings):
            continue
        ambiguous = (parts[0] <= 12) & (parts[1] <= 12) & (parts[0] != parts[1])
        if ambiguous.any():
            rows.append(
                {
                    "column": col,
                    "n_ambiguous": int(ambiguous.sum()),
                    "n_parsed": len(parts),
                    "examples": ", ".join(
                        strings.loc[parts.index[ambiguous]].unique()[:EXAMPLES]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=DATE_COLUMNS)
