"""Train vs test consistency: schema, missing, ranges, categories, overlap, issues."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops.profile import column_profile

# Semantic types (as strings, see ops.profile) treated as entity identifiers /
# as categories. `group_id` may not exist yet in profile: comparing on strings
# keeps this module valid before and after it lands.
ID_TYPES = ("id_like", "group_id")
CATEGORY_TYPES = ("categorical", "boolean")

SEVERITIES = ("error", "warning", "info")
# |pct_missing test - train| (points): >= WARNING -> warning, >= INFO -> info.
MISSING_DELTA_WARNING = 5.0
MISSING_DELTA_INFO = 1.0
# % of non-null test values outside the train [min, max]: > WARNING -> warning.
OUT_OF_RANGE_WARNING = 5.0

COLUMN_FIELDS = [
    "column",
    "in_train",
    "in_test",
    "dtype_train",
    "dtype_test",
    "semantic_train",
    "semantic_test",
    "pct_missing_train",
    "pct_missing_test",
    "pct_missing_delta",
    "min_train",
    "max_train",
    "min_test",
    "max_test",
    "pct_test_out_of_range",
    "mean_train",
    "mean_test",
    "n_unseen_categories",
    "unseen_categories",
    "pct_test_rows_unseen",
    "n_train_only_categories",
    "train_only_categories",
]
OVERLAP_FIELDS = [
    "kind",
    "column",
    "n_test",
    "n_test_in_train",
    "pct_test_in_train",
    "row_counter",
]
ISSUE_FIELDS = ["severity", "check", "column", "message"]


def _pct(part: float, total: float) -> float:
    return round(100 * part / total, 2) if total else 0.0


def _is_number(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)


def schema_diff(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Columns only in train / only in test / common, and whether order differs."""
    test_cols = set(test.columns)
    train_cols = set(train.columns)
    common = [c for c in train.columns if c in test_cols]
    return {
        "only_train": [c for c in train.columns if c not in test_cols],
        "only_test": [c for c in test.columns if c not in train_cols],
        "common": common,
        "order_differs": common != [c for c in test.columns if c in train_cols],
    }


def numeric_shift(train: pd.Series, test: pd.Series) -> dict:
    """Train / test min, max, mean and % of test values outside the train range.

    Empty dict unless both columns are numeric (a dtype mismatch is reported
    by the schema check instead).
    """
    if not (_is_number(train) and _is_number(test)):
        return {}
    tr, te = train.dropna(), test.dropna()
    if tr.empty or te.empty:
        return {}
    lo, hi = tr.min(), tr.max()
    outside = int(((te < lo) | (te > hi)).sum())
    return {
        "min_train": float(lo),
        "max_train": float(hi),
        "min_test": float(te.min()),
        "max_test": float(te.max()),
        "pct_test_out_of_range": _pct(outside, len(te)),
        "mean_train": round(float(tr.mean()), 4),
        "mean_test": round(float(te.mean()), 4),
    }


def category_shift(train: pd.Series, test: pd.Series) -> dict:
    """Categories seen in test but not in train (and the reverse).

    Values are compared as strings so a dtype mismatch (1 vs "1") is not
    counted as a new category.
    """
    tr = train.dropna().astype(str)
    te = test.dropna().astype(str)
    train_cats, test_cats = set(tr), set(te)
    unseen, train_only = test_cats - train_cats, train_cats - test_cats
    return {
        "n_unseen_categories": len(unseen),
        "unseen_categories": ", ".join(sorted(unseen)),
        "pct_test_rows_unseen": _pct(int(te.isin(unseen).sum()), len(test)),
        "n_train_only_categories": len(train_only),
        "train_only_categories": ", ".join(sorted(train_only)),
    }


def is_row_counter(train: pd.Series, test: pd.Series) -> bool:
    """True if both sides are integer counters 0..n-1 or 1..n (distinct values)."""

    def counter(series: pd.Series) -> bool:
        values = series.dropna()
        if values.empty or not pdt.is_integer_dtype(values):
            return False
        distinct = values.nunique()
        lo, hi = int(values.min()), int(values.max())
        return lo in (0, 1) and hi - lo + 1 == distinct

    return counter(train) and counter(test)


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    # Stringify first: rows are compared on their printed values across frames
    # whose dtypes may differ.
    return pd.util.hash_pandas_object(df.astype(str), index=False)


def overlap(
    train: pd.DataFrame, test: pd.DataFrame, id_columns: list[str]
) -> pd.DataFrame:
    """Test rows found verbatim in train (common columns) and test ids found in train.

    One `rows` record, then one `id` record per id column present on both sides.
    """
    common = schema_diff(train, test)["common"]
    records = []
    if common:
        in_train = _row_hashes(test[common]).isin(set(_row_hashes(train[common])))
        n_in = int(in_train.sum())
        records.append(
            {
                "kind": "rows",
                "column": None,
                "n_test": len(test),
                "n_test_in_train": n_in,
                "pct_test_in_train": _pct(n_in, len(test)),
                "row_counter": False,
            }
        )
    for col in id_columns:
        if col not in train.columns or col not in test.columns:
            continue
        test_ids = set(test[col].dropna().astype(str))
        n_in = len(test_ids & set(train[col].dropna().astype(str)))
        records.append(
            {
                "kind": "id",
                "column": str(col),
                "n_test": len(test_ids),
                "n_test_in_train": n_in,
                "pct_test_in_train": _pct(n_in, len(test_ids)),
                "row_counter": is_row_counter(train[col], test[col]),
            }
        )
    return pd.DataFrame(records, columns=OVERLAP_FIELDS)


def auto_id_columns(columns: pd.DataFrame) -> list[str]:
    """Common columns typed as an identifier on either side (from `compare_columns`)."""
    both = columns[columns["in_train"] & columns["in_test"]]
    is_id = both["semantic_train"].isin(ID_TYPES) | both["semantic_test"].isin(ID_TYPES)
    return both.loc[is_id, "column"].tolist()


def compare_columns(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """One row per column of the union (train order, then test-only columns)."""
    prof_train = column_profile(train).set_index("column")
    prof_test = column_profile(test).set_index("column")
    names = [str(c) for c in train.columns] + [
        str(c) for c in test.columns if c not in set(train.columns)
    ]
    rows = []
    for name in names:
        in_train, in_test = name in prof_train.index, name in prof_test.index
        row: dict = {"column": name, "in_train": in_train, "in_test": in_test}
        for side, prof, present in (
            ("train", prof_train, in_train),
            ("test", prof_test, in_test),
        ):
            if present:
                row[f"dtype_{side}"] = prof.at[name, "dtype"]
                row[f"semantic_{side}"] = prof.at[name, "semantic_type"]
                row[f"pct_missing_{side}"] = float(prof.at[name, "pct_missing"])
        if in_train and in_test:
            row["pct_missing_delta"] = round(
                row["pct_missing_test"] - row["pct_missing_train"], 2
            )
            semantics = {row["semantic_train"], row["semantic_test"]}
            # Ids get the overlap check instead: their range always shifts.
            if not semantics & set(ID_TYPES):
                row.update(numeric_shift(train[name], test[name]))
            if semantics & set(CATEGORY_TYPES):
                row.update(category_shift(train[name], test[name]))
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMN_FIELDS)


def find_issues(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: pd.DataFrame,
    overlap_table: pd.DataFrame,
    missing_id_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Turn the comparison into findings, sorted error -> warning -> info.

    Severity choice: a dtype mismatch or test ids found in train (entity leak)
    is an error; a column only in test, a semantic-type mismatch, unseen test
    categories, a large missing-rate or range shift, or rows shared by both
    tables is a warning; a column only in train (probable target), a column
    order change, train-only categories, small shifts and row-counter id
    overlap are info.
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
            add(
                "warning",
                "categorical",
                r.column,
                f"{int(r.n_unseen_categories)} categories unseen in train "
                f"({r.unseen_categories}) on "
                f"{r.pct_test_rows_unseen}% of test rows",
            )
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

    df = pd.DataFrame(issues, columns=ISSUE_FIELDS)
    rank = df["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    return df.iloc[rank.argsort(kind="stable")].reset_index(drop=True)
