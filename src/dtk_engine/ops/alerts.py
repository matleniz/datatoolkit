"""Severity-ranked data-quality alerts for ``dataset_overview``.

Every check reads what the overview already computed (column profile, numeric
stats, duplicate count): no extra pass over the data. Same finding shape and
severity vocabulary as ``ops.compare.issues``.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import pandas as pd

from dtk_engine.ops.compare.issues import ISSUE_FIELDS, SEVERITIES, Issue

# Share of missing cells in a column (%): a warning from MISSING_WARNING_PCT, an info
# from MISSING_INFO_PCT. A fully empty column is an error whatever the thresholds.
MISSING_INFO_PCT = 20.0
MISSING_WARNING_PCT = 50.0
# Distinct values of a categorical column: from this many, one-hot encoding explodes.
HIGH_CARDINALITY_UNIQUE = 50
# Share of duplicated rows (%): any duplicate is an info, a warning from this share.
DUPLICATE_ROWS_WARNING_PCT = 5.0
# |skewness| of a numeric column from which an info is raised (a modelling hint only).
SKEW_INFO = 2.0
# A column cannot be called constant (or all-distinct) from fewer rows than this.
MIN_ROWS_CONSTANT = 2
# Characters whose presence in the only column's name betrays a wrong ``sep``.
DELIMITER_CHARS = (";", "\t", "|")


def _column_issues(profile: pd.DataFrame, n_rows: int) -> Iterator[Issue]:
    """Empty / mostly missing, constant, high-cardinality and id-like columns.

    Thresholds compare the exact share ``n_missing / n_rows``; the rounded
    ``pct_missing`` is only displayed. Below 2 rows nothing is "constant".
    """
    for r in profile.itertuples(index=False):
        pct = 100 * r.n_missing / n_rows if n_rows else 0.0
        if n_rows and r.n_missing >= n_rows:
            yield "error", "empty", r.column, "column is entirely missing"
            continue
        if pct >= MISSING_INFO_PCT:
            sev = "warning" if pct >= MISSING_WARNING_PCT else "info"
            yield sev, "missing", r.column, f"{r.pct_missing}% missing values"
        if r.semantic_type == "constant":
            if n_rows < MIN_ROWS_CONSTANT:
                continue
            if r.n_missing > 0:
                msg = "one value besides missing: a presence flag"
                yield "info", "constant", r.column, msg
            else:
                msg = "single distinct value: carries no information"
                yield "warning", "constant", r.column, msg
        elif r.semantic_type == "id_like":
            msg = "looks like an identifier (almost all values distinct)"
            yield "info", "id_like", r.column, msg
        elif r.semantic_type == "categorical" and r.n_unique >= HIGH_CARDINALITY_UNIQUE:
            msg = f"{r.n_unique} distinct categories: costly to one-hot encode"
            yield "info", "high_cardinality", r.column, msg


def _duplicate_issues(n_rows: int, n_dups: int) -> Iterator[Issue]:
    """Duplicated rows: info, warning from DUPLICATE_ROWS_WARNING_PCT of the rows."""
    if not n_dups:
        return
    pct = 100 * n_dups / n_rows
    sev = "warning" if pct >= DUPLICATE_ROWS_WARNING_PCT else "info"
    yield sev, "duplicates", None, f"{n_dups} duplicate rows ({pct:.1f}%)"


def _skew_issues(numeric: pd.DataFrame) -> Iterator[Issue]:
    """Strongly skewed numeric columns: info from SKEW_INFO."""
    for r in numeric.itertuples(index=False):
        if pd.notna(r.skew) and abs(r.skew) >= SKEW_INFO:
            yield "info", "skew", r.column, f"strongly skewed (skewness {r.skew:+.2f})"


def _delimiter_issues(profile: pd.DataFrame) -> Iterator[Issue]:
    """One column whose name holds a delimiter: the file was read with the wrong ``sep``."""
    if len(profile) != 1:
        return
    name = str(profile["column"].iloc[0])
    for ch in DELIMITER_CHARS:
        if ch in name:
            shown = repr(ch)[1:-1] if ch == "\t" else ch
            label = "tab" if ch == "\t" else f"'{ch}'"
            msg = (
                f"single column whose header contains {label}: the file is probably "
                f"{label}-separated (set the source's sep to '{shown}')"
            )
            yield "error", "delimiter", name, msg
            return


def dataset_alerts(
    profile: pd.DataFrame, numeric: pd.DataFrame, n_rows: int, n_dups: int
) -> pd.DataFrame:
    """Alerts sorted error -> warning -> info (stable within a severity)."""
    issues = [
        dict(zip(ISSUE_FIELDS, issue, strict=True))
        for issue in chain(
            _delimiter_issues(profile),
            _column_issues(profile, n_rows),
            _duplicate_issues(n_rows, n_dups),
            _skew_issues(numeric),
        )
    ]
    df = pd.DataFrame(issues, columns=ISSUE_FIELDS)
    rank = df["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    return df.iloc[rank.argsort(kind="stable")].reset_index(drop=True)
