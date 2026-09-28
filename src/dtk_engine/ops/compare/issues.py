"""Turn a train / test comparison into severity-ranked findings."""

from __future__ import annotations

from collections.abc import Sequence

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

    Severity choice: a dtype mismatch between non-numeric kinds (numeric vs
    string, datetime or bool vs other) or test ids found in train (entity leak)
    is an error, while int vs float (typically a NaN on one side) is info; a column only in test, a semantic-type mismatch, unseen test
    categories, a large missing-rate or range shift, or rows shared by both
    tables is a warning; a column only in train (probable target), a column
    order change, train-only categories, small shifts and row-counter id
    overlap are info. Drift (`check="drift"`, one finding per column, worst
    severity wins) uses the `PSI_*`, `SMD_WARNING`, `KS_WARNING`,
    `OUTSIDE_P1_P99_*` and `TVD_WARNING` constants and the optional drift tables.
    """
    issues: list[dict] = []

    def add(severity: str, check: str, column: str | None, message: str) -> None:
        issues.append(
            {"severity": severity, "check": check, "column": column, "message": message}
        )

    schema = schema_diff(train, test)
    for col in schema["only_train"]:
        add("info", "schema", str(col), "only in train (probable target)")
    for col in schema["only_test"]:
        add(
            "warning",
            "schema",
            str(col),
            "only in test: unusable by a model fit on train",
        )
    if schema["order_differs"]:
        add("info", "schema", None, "common columns are in a different order")
    for col in missing_id_columns:
        add("error", "overlap", col, "id column not present in both tables")

    both = columns[columns["in_train"] & columns["in_test"]]
    for r in both.itertuples(index=False):
        if r.dtype_train != r.dtype_test:
            if is_number(train[r.column]) and is_number(test[r.column]):
                add(
                    "info",
                    "schema",
                    r.column,
                    f"dtype {r.dtype_train} in train vs {r.dtype_test} in test "
                    "(numeric on both sides)",
                )
            else:
                add(
                    "error",
                    "schema",
                    r.column,
                    f"dtype {r.dtype_train} in train vs {r.dtype_test} in test",
                )
        if r.semantic_train != r.semantic_test:
            add(
                "warning",
                "schema",
                r.column,
                f"semantic type {r.semantic_train} in train vs {r.semantic_test} in test",
            )
        delta = abs(r.pct_missing_delta)
        if delta >= MISSING_DELTA_INFO:
            add(
                "warning" if delta >= MISSING_DELTA_WARNING else "info",
                "missing",
                r.column,
                f"{r.pct_missing_train}% missing in train vs {r.pct_missing_test}% in test",
            )
        out = r.pct_test_out_of_range
        if pd.notna(out) and out > 0:
            add(
                "warning" if out > OUT_OF_RANGE_WARNING else "info",
                "numeric",
                r.column,
                f"{out}% of test values outside train range "
                f"[{r.min_train:g}, {r.max_train:g}]",
            )
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
            add("warning", "categorical", r.column, msg)
        if pd.notna(r.n_train_only_categories) and r.n_train_only_categories > 0:
            add(
                "info",
                "categorical",
                r.column,
                f"{int(r.n_train_only_categories)} train categories absent from test "
                f"({r.train_only_categories})",
            )

    for r in overlap_table.itertuples(index=False):
        if r.n_test_in_train == 0:
            continue
        if r.kind == "rows":
            add(
                "warning",
                "overlap",
                None,
                f"{r.n_test_in_train} test rows ({r.pct_test_in_train}%) "
                "also appear in train",
            )
        elif r.row_counter:
            add(
                "info",
                "overlap",
                r.column,
                "row counter (0..n-1 / 1..n on both sides): id overlap is not a leak",
            )
        else:
            add(
                "error",
                "overlap",
                r.column,
                f"{r.n_test_in_train} test ids ({r.pct_test_in_train}%) "
                "also in train (entity leak)",
            )

    if numeric_drift_table is not None:
        for r in numeric_drift_table[numeric_drift_table["skipped"].isna()].itertuples(
            index=False
        ):
            outside = r.pct_test_below_train_p1 + r.pct_test_above_train_p99
            warn, info = [], []
            if r.psi >= PSI_WARNING:
                warn.append(f"PSI {r.psi:.2f}")
            elif r.psi >= PSI_INFO:
                info.append(f"PSI {r.psi:.2f}")
            if abs(r.smd) >= SMD_WARNING:
                warn.append(f"SMD {r.smd:+.2f}")
            if r.ks >= KS_WARNING:
                warn.append(f"KS {r.ks:.2f}")
            if outside >= OUTSIDE_P1_P99_WARNING:
                warn.append(f"{outside:.1f}% of test outside train p1-p99")
            elif outside >= OUTSIDE_P1_P99_INFO:
                info.append(f"{outside:.1f}% of test outside train p1-p99")
            if warn or info:
                add(
                    "warning" if warn else "info",
                    "drift",
                    r.column,
                    "distribution drift: " + ", ".join(warn + info),
                )
    if categorical_drift_table is not None and len(categorical_drift_table):
        tvds = categorical_drift_table.groupby("column", sort=False)["tvd"].first()
        for col, tvd in tvds.items():
            if tvd >= TVD_WARNING:
                add(
                    "warning",
                    "drift",
                    col,
                    f"category shares drift: total variation distance {tvd:.2f}",
                )

    df = pd.DataFrame(issues, columns=ISSUE_FIELDS)
    rank = df["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    return df.iloc[rank.argsort(kind="stable")].reset_index(drop=True)
