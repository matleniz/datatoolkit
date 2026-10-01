"""Turn a train / test comparison into severity-ranked findings."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain

import pandas as pd

from dtk_engine.ops.compare.drift import (
    KS_WARNING,
    OUTSIDE_P1_P99_INFO,
    OUTSIDE_P1_P99_WARNING,
    PSI_INFO,
    PSI_WARNING,
    SMD_WARNING,
    TVD_WARNING,
)
from dtk_engine.ops.compare.schema import (
    MISSING_DELTA_INFO,
    MISSING_DELTA_WARNING,
    OUT_OF_RANGE_WARNING,
    is_number,
    schema_diff,
)

SEVERITIES = ("error", "warning", "info")
ISSUE_FIELDS = ["severity", "check", "column", "message"]

Issue = tuple[str, str, str | None, str]  # (severity, check, column, message)


def _schema_issues(
    train: pd.DataFrame, test: pd.DataFrame, missing_id_columns: Sequence[str]
) -> Iterator[Issue]:
    """Only-in-train / only-in-test / reordered columns, absent id columns."""
    schema = schema_diff(train, test)
    for col in schema["only_train"]:
        yield "info", "schema", str(col), "only in train (probable target)"
    for col in schema["only_test"]:
        msg = "only in test: unusable by a model fit on train"
        yield "warning", "schema", str(col), msg
    if schema["order_differs"]:
        yield "info", "schema", None, "common columns are in a different order"
    for col in missing_id_columns:
        yield "error", "overlap", col, "id column not present in both tables"


def _dtype_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """A dtype mismatch is an error, except int vs float (typically a NaN): info."""
    if r.dtype_train == r.dtype_test:
        return
    msg = f"dtype {r.dtype_train} in train vs {r.dtype_test} in test"
    if is_number(train[r.column]) and is_number(test[r.column]):
        yield "info", "schema", r.column, f"{msg} (numeric on both sides)"
    else:
        yield "error", "schema", r.column, msg


def _semantic_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """A semantic-type mismatch is a warning."""
    if r.semantic_train != r.semantic_test:
        msg = (
            f"semantic type {r.semantic_train} in train "
            f"vs {r.semantic_test} in test"
        )
        yield "warning", "schema", r.column, msg


def _missing_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """A missing-rate shift: warning when large, info when small."""
    delta = abs(r.pct_missing_delta)
    if delta >= MISSING_DELTA_INFO:
        sev = "warning" if delta >= MISSING_DELTA_WARNING else "info"
        msg = (
            f"{r.pct_missing_train}% missing in train "
            f"vs {r.pct_missing_test}% in test"
        )
        yield sev, "missing", r.column, msg


def _range_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """Test values outside the train range: warning when large, info when small."""
    out = r.pct_test_out_of_range
    if pd.notna(out) and out > 0:
        sev = "warning" if out > OUT_OF_RANGE_WARNING else "info"
        msg = (
            f"{out}% of test values outside train range "
            f"[{r.min_train:g}, {r.max_train:g}]"
        )
        yield sev, "numeric", r.column, msg


def _unseen_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """Test categories unseen in train are a warning."""
    if pd.notna(r.n_unseen_categories) and r.n_unseen_categories > 0:
        counts = (
            f" [{r.unseen_category_counts}]"
            if pd.notna(r.unseen_category_counts) and r.unseen_category_counts
            else ""
        )
        msg = (
            f"{int(r.n_unseen_categories)} categories unseen in train "
            f"({r.unseen_categories}){counts} on "
            f"{r.pct_test_rows_unseen}% of test rows"
        )
        if pd.notna(r.near_match_hint) and r.near_match_hint:
            msg = f"{msg}; {r.near_match_hint}"
        yield "warning", "categorical", r.column, msg


def _train_only_issues(r, train: pd.DataFrame, test: pd.DataFrame) -> Iterator[Issue]:
    """Train categories absent from test are info."""
    if pd.notna(r.n_train_only_categories) and r.n_train_only_categories > 0:
        msg = (
            f"{int(r.n_train_only_categories)} train categories absent from test "
            f"({r.train_only_categories})"
        )
        yield "info", "categorical", r.column, msg


# Run per shared column, in this order (findings of one column stay together).
_COLUMN_RULES = (
    _dtype_issues,
    _semantic_issues,
    _missing_issues,
    _range_issues,
    _unseen_issues,
    _train_only_issues,
)


def _column_issues(
    train: pd.DataFrame, test: pd.DataFrame, columns: pd.DataFrame
) -> Iterator[Issue]:
    both = columns[columns["in_train"] & columns["in_test"]]
    for r in both.itertuples(index=False):
        for rule in _COLUMN_RULES:
            yield from rule(r, train, test)


def _overlap_issues(overlap_table: pd.DataFrame) -> Iterator[Issue]:
    """Test rows found in train warn, an id leak errors, a row counter is info."""
    for r in overlap_table.itertuples(index=False):
        if r.n_test_in_train == 0:
            continue
        if r.kind == "rows":
            msg = (
                f"{r.n_test_in_train} test rows ({r.pct_test_in_train}%) "
                "also appear in train"
            )
            yield "warning", "overlap", None, msg
        elif r.row_counter:
            msg = "row counter (0..n-1 / 1..n on both sides): id overlap is not a leak"
            yield "info", "overlap", r.column, msg
        else:
            msg = (
                f"{r.n_test_in_train} test ids ({r.pct_test_in_train}%) "
                "also in train (entity leak)"
            )
            yield "error", "overlap", r.column, msg


def _numeric_drift_issues(table: pd.DataFrame) -> Iterator[Issue]:
    for r in table[table["skipped"].isna()].itertuples(index=False):
        outside = r.pct_test_below_train_p1 + r.pct_test_above_train_p99
        # (statistic, warning threshold, info threshold or None, label)
        stats = (
            (r.psi, PSI_WARNING, PSI_INFO, f"PSI {r.psi:.2f}"),
            (abs(r.smd), SMD_WARNING, None, f"SMD {r.smd:+.2f}"),
            (r.ks, KS_WARNING, None, f"KS {r.ks:.2f}"),
            (
                outside,
                OUTSIDE_P1_P99_WARNING,
                OUTSIDE_P1_P99_INFO,
                f"{outside:.1f}% of test outside train p1-p99",
            ),
        )
        warn = [txt for v, w, _, txt in stats if v >= w]
        info = [txt for v, w, i, txt in stats if i is not None and i <= v < w]
        if warn or info:
            msg = "distribution drift: " + ", ".join(warn + info)
            yield "warning" if warn else "info", "drift", r.column, msg


def _categorical_drift_issues(table: pd.DataFrame) -> Iterator[Issue]:
    for col, tvd in table.groupby("column", sort=False)["tvd"].first().items():
        if tvd >= TVD_WARNING:
            msg = f"category shares drift: total variation distance {tvd:.2f}"
            yield "warning", "drift", col, msg


def _drift_issues(
    numeric_drift_table: pd.DataFrame | None,
    categorical_drift_table: pd.DataFrame | None,
) -> Iterator[Issue]:
    """Drift (``check="drift"``): one finding per column, worst severity wins.

    Uses the ``PSI_*``, ``SMD_WARNING``, ``KS_WARNING``, ``OUTSIDE_P1_P99_*``
    and ``TVD_WARNING`` constants.
    """
    if numeric_drift_table is not None:
        yield from _numeric_drift_issues(numeric_drift_table)
    if categorical_drift_table is not None and len(categorical_drift_table):
        yield from _categorical_drift_issues(categorical_drift_table)


def find_issues(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: pd.DataFrame,
    overlap_table: pd.DataFrame,
    missing_id_columns: Sequence[str] = (),
    numeric_drift_table: pd.DataFrame | None = None,
    categorical_drift_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn the comparison into findings, sorted error -> warning -> info.

    Each check is a rule function yielding ``(severity, check, column,
    message)``; the severity each one picks is described on the rule. The sort
    is stable, so findings keep their rule order within a severity.
    """
    issues = [
        dict(zip(ISSUE_FIELDS, issue, strict=True))
        for issue in chain(
            _schema_issues(train, test, missing_id_columns),
            _column_issues(train, test, columns),
            _overlap_issues(overlap_table),
            _drift_issues(numeric_drift_table, categorical_drift_table),
        )
    ]
    df = pd.DataFrame(issues, columns=ISSUE_FIELDS)
    rank = df["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    return df.iloc[rank.argsort(kind="stable")].reset_index(drop=True)
