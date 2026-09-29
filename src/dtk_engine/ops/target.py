"""Each feature vs the label: association strength, per-class stats, target rate
per category, binned mean of a regression target.

Pure pandas / numpy / sklearn, no ``Result``. Numeric features reuse the filter
scores of ``ops.selection`` (mutual information, ANOVA F / f_regression, median
fill for missing values); categorical features get a discrete mutual
information on their (top-k) labels, missing as its own label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from dtk_engine.ops.distribution import TOP_K, natural_key, top_labels, value_label
from dtk_engine.ops.selection import filter_scores, target_vector

# Bins (quantiles of the feature) of the mean-of-target curve (regression).
FEATURE_BINS = 10

ASSOCIATION_FIELDS = [
    "rank",
    "column",
    "kind",
    "mutual_info",
    "association",
    "measure",
    "f_score",
    "f_pvalue",
    "pearson",
    "spearman",
    "pct_missing",
]
CLASS_STATS_FIELDS = [
    "column",
    "class",
    "count",
    "mean",
    "std",
    "min",
    "q1",
    "median",
    "q3",
    "max",
]


def labeled_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows with a target value (the analysis ignores unlabeled rows)."""
    return df[df[target].notna()]


def class_balance(y: pd.Series) -> pd.DataFrame:
    """class, count, pct (over labeled rows), most frequent first."""
    counts = y.map(value_label).value_counts()
    return pd.DataFrame(
        {
            "class": counts.index,
            "count": counts.to_numpy(),
            "pct": (100 * counts / counts.sum()).round(2).to_numpy(),
        }
    )


def correlation_ratio(labels: pd.Series, values: pd.Series) -> float:
    """eta: share of the variance of ``values`` explained by the groups
    ``labels`` (sqrt of between / total sum of squares), in [0, 1]."""
    values = values.astype(float)
    total = ((values - values.mean()) ** 2).sum()
    if total == 0:
        return 0.0
    means = values.groupby(labels).transform("mean")
    between = ((means - values.mean()) ** 2).sum()
    return float(np.sqrt(between / total))


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V of two label series (0 = independent, 1 = one decides the other)."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    r, k = table.shape
    if min(r, k) < 2:
        return 0.0
    n = table.sum()
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def association_table(
    rows: pd.DataFrame,
    target: str,
    task: str,
    kinds: dict[str, str],
    top_k: int = TOP_K,
    random_state: int = 0,
) -> pd.DataFrame:
    """One row per feature, strongest first (rank 1 = highest mutual information).

    ``association`` is on a 0-1 scale, ``measure`` says which: numeric feature ->
    eta by class (classification) or |spearman| (regression); categorical ->
    Cramér's V (classification) or eta of y by category (regression).
    """
    op = "target_analysis"
    y = target_vector(rows, target, task, op)
    y_series = pd.Series(y, index=rows.index)
    classes = rows[target].map(value_label)
    numeric = [c for c, k in kinds.items() if k == "numeric"]
    categorical = [c for c, k in kinds.items() if k == "categorical"]
    out = []
    if numeric:
        X = rows[numeric].astype(float)
        filled = X.fillna(X.median()).fillna(0.0)
        scores = filter_scores(filled.to_numpy(), y, task, random_state)
        for i, col in enumerate(numeric):
            x = filled[col]
            row = {
                "column": col,
                "kind": "numeric",
                "mutual_info": float(scores["mutual_info"][i]),
                "f_score": float(scores["f_score"][i]),
                "f_pvalue": float(scores["f_pvalue"][i]),
            }
            if task == "classification":
                row.update(association=correlation_ratio(classes, x), measure="eta")
            else:
                pearson = _corr(x, y_series, "pearson")
                spearman = _corr(x, y_series, "spearman")
                row.update(
                    association=abs(spearman),
                    measure="|spearman|",
                    pearson=pearson,
                    spearman=spearman,
                )
            out.append(row)
    if categorical:
        labels = {c: top_labels(rows[c], top_k) for c in categorical}
        codes = np.column_stack(
            [pd.factorize(labels[c], sort=True)[0] for c in categorical]
        )
        mi_fn = (
            mutual_info_classif if task == "classification" else mutual_info_regression
        )
        mi = mi_fn(codes, y, discrete_features=True, random_state=random_state)
        for i, col in enumerate(categorical):
            if task == "classification":
                assoc, measure = cramers_v(labels[col], classes), "cramers_v"
            else:
                assoc, measure = correlation_ratio(labels[col], y_series), "eta"
            out.append(
                {
                    "column": col,
                    "kind": "categorical",
                    "mutual_info": float(mi[i]),
                    "association": assoc,
                    "measure": measure,
                }
            )
    table = pd.DataFrame(out).reindex(columns=ASSOCIATION_FIELDS)
    table["pct_missing"] = [
        round(100 * rows[c].isna().mean(), 2) for c in table["column"]
    ]
    table = table.sort_values(
        ["mutual_info", "association", "column"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = x.corr(y, method=method)
    return 0.0 if pd.isna(value) else float(value)


def numeric_by_class(rows: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """Box stats of a numeric feature per class (missing feature values ignored)."""
    out = []
    classes = rows[target].map(value_label)
    for cls in sorted(classes.unique(), key=natural_key):
        v = rows.loc[classes == cls, column].dropna().astype(float)
        n = len(v)
        q = v.quantile([0.25, 0.5, 0.75]).to_numpy() if n else [np.nan] * 3
        out.append(
            {
                "column": column,
                "class": cls,
                "count": n,
                "mean": v.mean() if n else np.nan,
                "std": v.std() if n > 1 else np.nan,
                "min": v.min() if n else np.nan,
                "q1": q[0],
                "median": q[1],
                "q3": q[2],
                "max": v.max() if n else np.nan,
            }
        )
    return pd.DataFrame(out, columns=CLASS_STATS_FIELDS)


def category_class_rates(
    rows: pd.DataFrame, column: str, target: str, top_k: int = TOP_K
) -> pd.DataFrame:
    """Per (top-k) value of a categorical feature: its row count and the share of
    each class among those rows (``rate``, 0-1). Long format."""
    labels = top_labels(rows[column], top_k)
    classes = rows[target].map(value_label)
    counts = pd.crosstab(labels, classes)
    rates = counts.div(counts.sum(axis=1), axis=0)
    order = counts.sum(axis=1).sort_values(ascending=False, kind="stable").index
    out = [
        {
            "column": column,
            "value": value,
            "count": int(counts.loc[value].sum()),
            "class": cls,
            "rate": float(rates.loc[value, cls]),
        }
        for value in order
        for cls in sorted(counts.columns, key=natural_key)
    ]
    return pd.DataFrame(out, columns=["column", "value", "count", "class", "rate"])


def category_target_means(
    rows: pd.DataFrame, column: str, target: str, top_k: int = TOP_K
) -> pd.DataFrame:
    """Per (top-k) value of a categorical feature: row count, mean target and its
    spread (std, quartiles)."""
    labels = top_labels(rows[column], top_k)
    y = rows[target].astype(float)
    by = y.groupby(labels)
    grouped = by.agg(["count", "mean", "std"])
    grouped["q1"] = by.quantile(0.25)
    grouped["q3"] = by.quantile(0.75)
    grouped = grouped.sort_values("count", ascending=False, kind="stable")
    return pd.DataFrame(
        {
            "column": column,
            "value": grouped.index,
            "count": grouped["count"].to_numpy(),
            "mean_target": grouped["mean"].to_numpy(),
            "std_target": grouped["std"].to_numpy(),
            "q1_target": grouped["q1"].to_numpy(),
            "q3_target": grouped["q3"].to_numpy(),
        }
    )


def binned_target_mean(
    rows: pd.DataFrame, column: str, target: str, bins: int = FEATURE_BINS
) -> pd.DataFrame:
    """Mean target per quantile bin of a numeric feature (missing feature: own bin)."""
    x = rows[column].astype(float)
    y = rows[target].astype(float)
    present = x.dropna()
    fields = [
        "column",
        "bin",
        "feature_mean",
        "count",
        "mean_target",
        "std_target",
        "q1_target",
        "q3_target",
    ]
    if present.empty:
        return pd.DataFrame(columns=fields)
    codes = pd.qcut(present, q=min(bins, present.nunique()), duplicates="drop")
    out = []
    for interval, idx in present.groupby(codes, observed=True).groups.items():
        out.append(
            {
                "column": column,
                "bin": str(interval),
                "feature_mean": float(x[idx].mean()),
                "count": len(idx),
                **_spread(y[idx]),
            }
        )
    missing = x.isna()
    if missing.any():
        out.append(
            {
                "column": column,
                "bin": value_label(np.nan),
                "feature_mean": np.nan,
                "count": int(missing.sum()),
                **_spread(y[missing]),
            }
        )
    return pd.DataFrame(out, columns=fields)


def _spread(y: pd.Series) -> dict:
    """Mean, std and quartiles of a target slice (``*_target`` table fields)."""
    return {
        "mean_target": float(y.mean()),
        "std_target": float(y.std()) if len(y) > 1 else np.nan,
        "q1_target": float(y.quantile(0.25)),
        "q3_target": float(y.quantile(0.75)),
    }


def _count_rows(counts: pd.DataFrame, order: list[str], key: str, column: str) -> list:
    """Long rows (one per group x class) from a group x class count crosstab;
    ``pct_of_<key>`` is the class share (0-100) of the group's rows."""
    totals = counts.sum(axis=1)
    return [
        {
            "column": column,
            key: group,
            "class": cls,
            "count": int(counts.loc[group, cls]),
            f"pct_of_{key}": float(100 * counts.loc[group, cls] / totals[group]),
        }
        for group in order
        for cls in sorted(counts.columns, key=natural_key)
    ]


def class_counts_by_category(
    rows: pd.DataFrame, column: str, target: str, top_k: int = TOP_K
) -> pd.DataFrame:
    """Row count per (top-k value of a categorical feature, class), long format,
    values most frequent first; ``pct_of_value`` = class share of the value's rows."""
    labels = top_labels(rows[column], top_k)
    counts = pd.crosstab(labels, rows[target].map(value_label))
    order = counts.sum(axis=1).sort_values(ascending=False, kind="stable").index
    return pd.DataFrame(
        _count_rows(counts, list(order), "value", column),
        columns=["column", "value", "class", "count", "pct_of_value"],
    )


def class_counts_by_bin(
    rows: pd.DataFrame, column: str, target: str, bins: int = FEATURE_BINS
) -> pd.DataFrame:
    """Row count per (quantile bin of a numeric feature, class), long format, bins
    in increasing order then ``(missing)``; ``pct_of_bin`` = class share of the bin."""
    fields = ["column", "bin", "class", "count", "pct_of_bin"]
    x = rows[column].astype(float)
    present = x.dropna()
    if present.empty:
        return pd.DataFrame(columns=fields)
    codes = pd.qcut(present, q=min(bins, present.nunique()), duplicates="drop")
    order = [str(i) for i in codes.cat.categories]
    labels = codes.astype(object).map(str).reindex(x.index)
    labels = labels.where(x.notna(), value_label(np.nan))
    if x.isna().any():
        order.append(value_label(np.nan))
    counts = pd.crosstab(labels, rows[target].map(value_label))
    return pd.DataFrame(_count_rows(counts, order, "bin", column), columns=fields)
