"""Train vs test distribution drift: KS, PSI, SMD, category shares."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from dtk_engine.ops._util import pct as _pct
from dtk_engine.ops.compare.schema import (
    CATEGORY_TYPES,
    ID_TYPES,
    is_number,
    is_row_counter,
)

# Drift thresholds (used by `find_issues`). Population stability index: >= WARNING
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
        if is_number(train[r.column]) and is_number(test[r.column]):
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
