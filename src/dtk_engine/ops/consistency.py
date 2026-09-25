"""Inconsistencies in text / categorical columns: variants, mixed types, dates
(ambiguous day / month order, several date formats in one column)."""

from __future__ import annotations

import re

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


MIXED_DATE_COLUMNS = [
    "column",
    "n_values",
    "n_date_like",
    "n_formats",
    "formats",
    "n_not_date",
    "examples",
]
# A column is a date column when at least this share of its strings look like a
# date in one of the recognised formats.
DATE_LIKE_MIN_SHARE = 0.5
_MONTHS = (
    *("jan", "feb", "mar", "apr", "may", "jun"),
    *("jul", "aug", "sep", "oct", "nov", "dec"),
)
_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[ap]m|Z|[+-]\d{2}:?\d{2})?)?"
_ISO = rf"^(\d{{4}})([-/.])(\d{{1,2}})\2(\d{{1,2}}){_TIME}$"
_NUMERIC = rf"^(\d{{1,2}})([-/.])(\d{{1,2}})\2(\d{{4}}|\d{{2}}){_TIME}$"
_DAY_MON = rf"^\d{{1,2}}[ -]([A-Za-z]{{3,9}})\.?,?[ -]\d{{2,4}}{_TIME}$"
_MON_DAY = rf"^([A-Za-z]{{3,9}})\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}{_TIME}$"


def _is_month(word: str) -> bool:
    return word[:3].lower() in _MONTHS


def date_format(value: str) -> str | None:
    """The written format of a date string (``yyyy-mm-dd``, ``dd/mm/yyyy``,
    ``mm/dd/yyyy``, ``nn/nn/yyyy`` when day and month cannot be told apart,
    ``dd mon yyyy``, ``mon dd yyyy``); None when it does not look like a date.
    A time part, if any, is ignored."""
    text = value.strip()
    if m := re.match(_ISO, text):
        return f"yyyy{m[2]}mm{m[2]}dd"
    if m := re.match(_NUMERIC, text):
        a, sep, b, year = int(m[1]), m[2], int(m[3]), "y" * len(m[4])
        if a > 31 or b > 31 or (a > 12 and b > 12):
            return None
        order = "dd{0}mm" if a > 12 else "mm{0}dd" if b > 12 else "nn{0}nn"
        return f"{order.format(sep)}{sep}{year}"
    if (m := re.match(_DAY_MON, text)) and _is_month(m[1]):
        return "dd mon yyyy"
    if (m := re.match(_MON_DAY, text)) and _is_month(m[1]):
        return "mon dd yyyy"
    return None


def _merge_ambiguous(formats: pd.Series) -> pd.Series:
    """Fold ``nn/nn/yyyy`` into ``dd/mm/yyyy`` or ``mm/dd/yyyy`` when the column
    shows only one of the two (then it is that order, not a second format)."""
    present = set(formats)
    out = formats.copy()
    for fmt in present:
        if not fmt.startswith("nn"):
            continue
        rest = fmt[2:]
        day_first, month_first = f"dd{rest[0]}mm{rest[3:]}", f"mm{rest[0]}dd{rest[3:]}"
        candidates = [f for f in (day_first, month_first) if f in present]
        if len(candidates) == 1:
            out[out == fmt] = candidates[0]
    return out


def mixed_date_formats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Date columns written several ways (``2020-01-15``, ``01/15/2020``,
    ``15-01-2020`` in one column) or holding non-date strings.

    A column counts as a date column when >= DATE_LIKE_MIN_SHARE of its strings
    look like a date; it is reported when it mixes >= 2 formats or has values
    that are not dates. Day-first vs month-first slashes count as two formats
    when both show up (a day > 12 on each side).
    """
    rows = []
    for col in columns:
        strings = _strings(df[col])
        strings = strings[strings.str.strip() != ""]
        if strings.empty:
            continue
        formats = strings.map(date_format)
        dated = formats.dropna()
        if len(dated) < DATE_LIKE_MIN_SHARE * len(strings):
            continue
        dated = _merge_ambiguous(dated)
        counts = dated.value_counts()
        not_date = strings[formats.isna()]
        if len(counts) < 2 and not_date.empty:
            continue
        examples = [strings[dated.index[dated == fmt][0]] for fmt in counts.index]
        examples += list(not_date.unique()[:EXAMPLES])
        rows.append(
            {
                "column": col,
                "n_values": len(strings),
                "n_date_like": len(dated),
                "n_formats": len(counts),
                "formats": ", ".join(f"{f}: {n}" for f, n in counts.items()),
                "n_not_date": len(not_date),
                "examples": ", ".join(examples),
            }
        )
    return pd.DataFrame(rows, columns=MIXED_DATE_COLUMNS)
