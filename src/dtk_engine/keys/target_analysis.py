"""Target analysis: each picked feature vs the label (course "per-label analysis")."""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.columns import pick_columns, value_kind
from dtk_engine.ops.distribution import TOP_K
from dtk_engine.ops.selection import resolve_task
from dtk_engine.ops.target import (
    FEATURE_BINS,
    association_table,
    binned_target_mean,
    category_class_rates,
    category_target_means,
    class_balance,
    labeled_rows,
    numeric_by_class,
)
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load

# Features analysed when none are picked (the first eligible ones).
MAX_COLUMNS = 30
# Per-feature figures for the best-ranked features.
FIGURE_TOP = 6


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    target: str = column_field("Survived", "Label column of `source`")
    columns: list[str] = columns_field(
        f"Features to analyse (empty = every numeric / categorical / boolean "
        f"column but the target, first {MAX_COLUMNS})"
    )
    task: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="auto: classification for a non-numeric target or a few "
        "integer values, regression otherwise",
    )
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=100,
        description="Categories kept per feature, rest (other)",
    )
    bins: int = Field(
        default=FEATURE_BINS,
        ge=2,
        le=50,
        description="Quantile bins of a numeric feature (regression: mean target per bin)",
    )
    random_state: int = Field(default=0, description="Mutual information seed")


@key(
    id="target_analysis",
    title="Target analysis",
    category="analysis",
    description="Each feature vs the label: ranking by association (mutual "
    "information, eta / Cramér's V / Spearman, ANOVA F), class balance; "
    "classification: per-class stats of numeric features and class rate per "
    "category; regression: mean target per feature bin / category.",
)
def run(params: Params) -> Result:
    return target_result(
        load(params.source),
        params.target,
        params.columns,
        params.task,
        params.top_k,
        params.bins,
        params.random_state,
    )


def target_result(
    df: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    task: str = "auto",
    top_k: int = TOP_K,
    bins: int = FEATURE_BINS,
    random_state: int = 0,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.target_analysis``)."""
    op = "target_analysis"
    if target not in df.columns:
        raise KeyParamsError(f"{op}: target {target!r} not in the frame")
    rows = labeled_rows(df, target)
    if rows.empty:
        raise KeyParamsError(f"{op}: target {target!r} has no value")
    picked, n_capped = pick_columns(
        df, columns or [], op, exclude=[target], cap=MAX_COLUMNS
    )
    task = resolve_task(rows[target], task)
    kinds = {c: value_kind(df[c]) for c in picked}
    ranking = association_table(rows, target, task, kinds, top_k, random_state)

    metrics: dict = {
        "task": task,
        "n_rows": len(rows),
        "n_unlabeled": len(df) - len(rows),
        "n_features": len(picked),
        "n_columns_capped": n_capped,
        "top_feature": str(ranking["column"].iloc[0]),
    }
    result = Result(metrics=metrics)
    result.add_table("ranking", ranking)
    top = ranking["column"].head(FIGURE_TOP).tolist()
    result.add_figure(
        "Mutual information with the target",
        px.bar(ranking, x="column", y="mutual_info", color="kind"),
    )
    if task == "classification":
        balance = class_balance(rows[target])
        result.metrics["n_classes"] = len(balance)
        result.metrics["minority_pct"] = float(balance["pct"].min())
        result.add_table("class_balance", balance)
        _classification_details(result, rows, target, picked, kinds, top, top_k)
    else:
        y = rows[target].astype(float)
        result.metrics["target_mean"] = float(y.mean())
        result.metrics["target_std"] = float(y.std()) if len(y) > 1 else 0.0
        _regression_details(result, rows, target, picked, kinds, top, top_k, bins)
    if n_capped:
        result.text = (
            f"{n_capped} more eligible features not analysed (first "
            f"{MAX_COLUMNS}); pick columns to see them."
        )
    return result


def _classification_details(result, rows, target, picked, kinds, top, top_k):
    numeric = [c for c in picked if kinds[c] == "numeric"]
    categorical = [c for c in picked if kinds[c] == "categorical"]
    stats = {c: numeric_by_class(rows, c, target) for c in numeric}
    rates = {c: category_class_rates(rows, c, target, top_k) for c in categorical}
    if stats:
        result.add_table(
            "numeric_by_class", pd.concat(stats.values(), ignore_index=True), "numeric"
        )
    if rates:
        result.add_table(
            "class_rate_by_category",
            pd.concat(rates.values(), ignore_index=True),
            "categorical",
        )
    for col in top:
        if col in stats:
            result.add_figure(
                f"{col} by class", _box_figure(stats[col], col), "numeric"
            )
        else:
            fig = px.bar(
                rates[col],
                x="value",
                y="rate",
                color="class",
                labels={"value": col, "rate": "share of the value's rows"},
            )
            result.add_figure(f"class rate by {col}", fig, "categorical")


def _regression_details(result, rows, target, picked, kinds, top, top_k, bins):
    binned = {
        c: binned_target_mean(rows, c, target, bins)
        for c in picked
        if kinds[c] == "numeric"
    }
    means = {
        c: category_target_means(rows, c, target, top_k)
        for c in picked
        if kinds[c] == "categorical"
    }
    if binned:
        result.add_table(
            "binned_target_mean",
            pd.concat(binned.values(), ignore_index=True),
            "numeric",
        )
    if means:
        result.add_table(
            "target_mean_by_category",
            pd.concat(means.values(), ignore_index=True),
            "categorical",
        )
    labels = {"mean_target": f"mean {target}"}
    for col in top:
        if col in binned:
            fig = px.line(
                binned[col].dropna(subset=["feature_mean"]),
                x="feature_mean",
                y="mean_target",
                markers=True,
                labels={**labels, "feature_mean": col},
            )
            result.add_figure(f"mean {target} by {col} bin", fig, "numeric")
        else:
            fig = px.bar(
                means[col], x="value", y="mean_target", labels={**labels, "value": col}
            )
            result.add_figure(f"mean {target} by {col}", fig, "categorical")


def _box_figure(stats: pd.DataFrame, col: str) -> go.Figure:
    """Box per class from precomputed quartiles (no raw points in the JSON)."""
    fig = go.Figure(
        go.Box(
            x=stats["class"],
            q1=stats["q1"],
            median=stats["median"],
            q3=stats["q3"],
            lowerfence=stats["min"],
            upperfence=stats["max"],
            mean=stats["mean"],
        )
    )
    fig.update_layout(xaxis_title="class", yaxis_title=col)
    return fig
