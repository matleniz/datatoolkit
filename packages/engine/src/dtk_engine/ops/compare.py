"""Train vs test consistency: schema, missing, ranges, categories, overlap, issues."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
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

# Drift thresholds (see `drift_severity`). Population stability index: >= WARNING
# -> warning, >= INFO -> info (the usual 0.1 / 0.25 rules of thumb).
PSI_WARNING = 0.25
PSI_INFO = 0.1
# |standardized mean difference| (pooled std): >= WARNING -> warning.
SMD_WARNING = 0.5
# Two-sample Kolmogorov-Smirnov statistic: >= WARNING -> warning.
KS_WARNING = 0.2
# % of test values outside the train [p1, p99]: >= WARNING -> warning, >= INFO -> info.
# Without drift ~2 % of values fall outside p1-p99 by construction, so INFO sits above
# that baseline.
OUTSIDE_P1_P99_WARNING = 5.0
OUTSIDE_P1_P99_INFO = 3.0
# Total variation distance between train and test category shares: >= WARNING -> warning.
TVD_WARNING = 0.2
PSI_BINS = 10
PSI_EPSILON = 1e-4
CATEGORICAL_TOP = 20

NUMERIC_DRIFT_FIELDS = [
    "column",
    "mean_train",
    "mean_test",
    "std_train",
    "std_test",
    "p1_train",
    "p1_test",
    "median_train",
    "median_test",
    "p99_train",
    "p99_test",
    "min_train",
    "min_test",
    "max_train",
    "max_test",
    "smd",
    "ks",
    "psi",
    "pct_test_below_train_p1",
    "pct_test_above_train_p99",
    "pct_test_outside_train_range",
    "skipped",
]
CATEGORICAL_DRIFT_FIELDS = [
    "column",
    "tvd",
    "value",
    "pct_train",
    "pct_test",
    "diff",
]

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


def drift_columns(train: pd.DataFrame, test: pd.DataFrame, columns: pd.DataFrame):
    """Common columns to check for drift: `(numeric, categorical)` name lists.

    Ids (either side) and row counters are excluded. Numeric = numeric non-bool on
    both sides; categorical = typed categorical / boolean on either side.
    """
    both = columns[columns["in_train"] & columns["in_test"]]
    numeric, categorical = [], []
    for r in both.itertuples(index=False):
        semantics = {r.semantic_train, r.semantic_test}
        if semantics & set(ID_TYPES) or is_row_counter(train[r.column], test[r.column]):
            continue
        if _is_number(train[r.column]) and _is_number(test[r.column]):
            numeric.append(r.column)
        if semantics & set(CATEGORY_TYPES):
            categorical.append(r.column)
    return numeric, categorical


def _clean(series: pd.Series) -> np.ndarray:
    """Finite-or-not non-null values as a sorted float array (NaN dropped)."""
    values = series.to_numpy(dtype=float, na_value=np.nan)
    return np.sort(values[~np.isnan(values)])


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic between two sorted arrays (max CDF gap)."""
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def psi(train: np.ndarray, test: np.ndarray) -> float:
    """Population stability index, `PSI_BINS` bins cut on train quantiles.

    Tied quantiles collapse into fewer bins; empty bins get `PSI_EPSILON`.
    """
    cuts = np.unique(np.quantile(train, np.linspace(0, 1, PSI_BINS + 1)[1:-1]))
    n_bins = len(cuts) + 1
    share_train = np.bincount(np.searchsorted(cuts, train), minlength=n_bins)
    share_test = np.bincount(np.searchsorted(cuts, test), minlength=n_bins)
    p = np.clip(share_train / len(train), PSI_EPSILON, None)
    q = np.clip(share_test / len(test), PSI_EPSILON, None)
    return float(((q - p) * np.log(q / p)).sum())


def numeric_drift(
    train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Side-by-side distribution stats and drift scores per numeric column.

    NaN ignored. A column with < 2 non-null values on a side keeps its row with
    only `skipped` set (the reason).
    """
    rows = []
    for col in columns:
        tr, te = _clean(train[col]), _clean(test[col])
        row: dict = {"column": str(col), "skipped": None}
        few = [s for s, v in (("train", tr), ("test", te)) if len(v) < 2]
        if few:
            row["skipped"] = "< 2 non-null values in " + " and ".join(few)
            rows.append(row)
            continue
        p1, p99 = np.percentile(tr, [1, 99])
        pooled = np.sqrt((tr.var(ddof=1) + te.var(ddof=1)) / 2)
        for side, v in (("train", tr), ("test", te)):
            q1, med, q99 = np.percentile(v, [1, 50, 99])
            row.update(
                {
                    f"mean_{side}": float(v.mean()),
                    f"std_{side}": float(v.std(ddof=1)),
                    f"p1_{side}": float(q1),
                    f"median_{side}": float(med),
                    f"p99_{side}": float(q99),
                    f"min_{side}": float(v[0]),
                    f"max_{side}": float(v[-1]),
                }
            )
        row["smd"] = float((te.mean() - tr.mean()) / pooled) if pooled else np.nan
        row["ks"] = ks_statistic(tr, te)
        row["psi"] = psi(tr, te)
        row["pct_test_below_train_p1"] = _pct(int((te < p1).sum()), len(te))
        row["pct_test_above_train_p99"] = _pct(int((te > p99).sum()), len(te))
        row["pct_test_outside_train_range"] = _pct(
            int(((te < tr[0]) | (te > tr[-1])).sum()), len(te)
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=NUMERIC_DRIFT_FIELDS)


def categorical_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    top: int = CATEGORICAL_TOP,
) -> pd.DataFrame:
    """Long table `column, tvd, value, pct_train, pct_test, diff` (test - train).

    `tvd` is the total variation distance over all categories (values compared as
    strings, NaN ignored); only the `top` most frequent values (max of the two
    shares) are listed. A column with no non-null value on a side is left out.
    """
    parts = []
    for col in columns:
        tr = train[col].dropna().astype(str).value_counts(normalize=True)
        te = test[col].dropna().astype(str).value_counts(normalize=True)
        if tr.empty or te.empty:
            continue
        shares = pd.concat([tr.rename("train"), te.rename("test")], axis=1).fillna(0.0)
        tvd = float(0.5 * (shares["train"] - shares["test"]).abs().sum())
        shares = shares.loc[shares.max(axis=1).sort_values(ascending=False).index[:top]]
        parts.append(
            pd.DataFrame(
                {
                    "column": str(col),
                    "tvd": round(tvd, 4),
                    "value": shares.index,
                    "pct_train": (100 * shares["train"]).round(2).to_numpy(),
                    "pct_test": (100 * shares["test"]).round(2).to_numpy(),
                    "diff": (100 * (shares["test"] - shares["train"]))
                    .round(2)
                    .to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=CATEGORICAL_DRIFT_FIELDS)
    return pd.concat(parts, ignore_index=True)[CATEGORICAL_DRIFT_FIELDS]


def histogram_pair(
    train: pd.Series, test: pd.Series, bins: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(edges, train_density, test_density)` on shared bins, each summing to 1."""
    tr, te = _clean(train), _clean(test)
    edges = np.histogram_bin_edges(np.concatenate([tr, te]), bins=bins)
    return (
        edges,
        np.histogram(tr, bins=edges)[0] / len(tr),
        np.histogram(te, bins=edges)[0] / len(te),
    )


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
            if _is_number(train[r.column]) and _is_number(test[r.column]):
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
