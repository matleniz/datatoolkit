"""Train vs test schema, per-column shifts and row / id overlap."""

from __future__ import annotations

import string

import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops._util import pct as _pct
from dtk_engine.ops.profile import as_text, column_profile

# Semantic types (as strings, see ops.profile) treated as entity identifiers /
# as categories.
ID_TYPES = ("id_like", "group_id")
CATEGORY_TYPES = ("categorical", "boolean")

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
    "unseen_category_counts",
    "pct_test_rows_unseen",
    "n_train_only_categories",
    "train_only_categories",
    "near_match_hint",
]
OVERLAP_FIELDS = [
    "kind",
    "column",
    "n_test",
    "n_test_in_train",
    "pct_test_in_train",
    "row_counter",
]


def is_number(series: pd.Series) -> bool:
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
    if not (is_number(train) and is_number(test)):
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


def _value_near_key(value: str) -> str:
    """Normalise a category for near-match: strip, casefold, trailing punctuation."""
    return value.strip().casefold().rstrip(string.punctuation).strip()


def _near_match_pairs(unseen: set[str], train_cats: set[str]) -> list[dict[str, str]]:
    """Unseen test values that collapse onto a train value after ``_value_near_key``."""
    train_by_key: dict[str, str] = {}
    for t in train_cats:
        train_by_key.setdefault(_value_near_key(t), t)
    pairs: list[dict[str, str]] = []
    for u in sorted(unseen):
        key = _value_near_key(u)
        train_hit = train_by_key.get(key)
        if train_hit is not None and train_hit != u:
            pairs.append({"test": u, "train": train_hit})
    return pairs


def _near_match_hint(pairs: list[dict[str, str]]) -> str | None:
    if not pairs:
        return None
    mapped = ", ".join(f"{p['test']!r}→{p['train']!r}" for p in pairs)
    return (
        f"near-match after strip/casefold/trailing punctuation: {mapped}; "
        "try standardize_text or map on test"
    )


def category_shift(train: pd.Series, test: pd.Series) -> dict:
    """Categories seen in test but not in train (and the reverse).

    Values are compared as strings so a dtype mismatch (1 vs "1") is not
    counted as a new category. When unseen test values normalise onto train
    values (strip, casefold, trailing punctuation), ``near_match_hint``
    suggests a ``standardize_text`` / map step on test.
    """
    tr = train.dropna().astype(str)
    te = test.dropna().astype(str)
    train_cats, test_cats = set(tr), set(te)
    unseen, train_only = test_cats - train_cats, train_cats - test_cats
    counts = te[te.isin(unseen)].value_counts()
    only_in_test = [
        {"value": str(v), "count": int(c)} for v, c in counts.items()
    ]
    counts_str = ", ".join(f"{r['value']} ({r['count']})" for r in only_in_test)
    near_matches = _near_match_pairs(unseen, train_cats)
    return {
        "n_unseen_categories": len(unseen),
        "unseen_categories": ", ".join(sorted(unseen)),
        "unseen_category_counts": counts_str,
        "pct_test_rows_unseen": _pct(int(te.isin(unseen).sum()), len(test)),
        "n_train_only_categories": len(train_only),
        "train_only_categories": ", ".join(sorted(train_only)),
        "near_match_hint": _near_match_hint(near_matches),
        # Structured form for align_report (not stored on the columns table).
        "_only_in_test": only_in_test,
        "_near_matches": near_matches,
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
    text = pd.DataFrame({c: as_text(df[c]) for c in df.columns}, index=df.index)
    return pd.util.hash_pandas_object(text, index=False)


def overlap(
    train: pd.DataFrame, test: pd.DataFrame, id_columns: list[str]
) -> pd.DataFrame:
    """Test rows found verbatim in train (common columns) and test ids found in train.

    One `rows` record, then one `id` record per id column present on both sides.
    The row hash leaves out id columns (given, or typed as ids on either side) and
    row counters, which differ between two tables even for an identical row.
    """
    common = schema_diff(train, test)["common"]
    sem_train = column_profile(train).set_index("column")["semantic_type"]
    sem_test = column_profile(test).set_index("column")["semantic_type"]
    skip = {str(c) for c in id_columns}
    keep = [
        c
        for c in common
        if str(c) not in skip
        and sem_train.get(str(c)) not in ID_TYPES
        and sem_test.get(str(c)) not in ID_TYPES
        and not is_row_counter(train[c], test[c])
    ]
    records = []
    if keep:
        in_train = _row_hashes(test[keep]).isin(set(_row_hashes(train[keep])))
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
