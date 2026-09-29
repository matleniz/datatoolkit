"""Target analysis: each picked feature vs the label (course "per-label analysis")."""

import math
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.columns import pick_columns, value_kind
from dtk_engine.ops.distribution import MISSING_LABEL, OTHER_LABEL, TOP_K, natural_key
from dtk_engine.ops.selection import resolve_task
from dtk_engine.ops.target import (
    FEATURE_BINS,
    association_table,
    binned_target_mean,
    category_class_rates,
    category_target_means,
    class_balance,
    class_counts_by_bin,
    class_counts_by_category,
    labeled_rows,
    numeric_by_class,
)
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import CsvSource, SourceSpec, load

# Features analysed when none are picked (the first eligible ones).
MAX_COLUMNS = 30
# A bin / category needs this share of the rows (and 5 rows) to bound a headline.
MIN_GROUP_SHARE = 0.05
# Features drawn in the association ranking figure.
RANKING_SHOWN = 30


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
        description="Quantile bins of a numeric feature (class counts / mean target per bin)",
    )
    target_bins: int = Field(
        default=FEATURE_BINS,
        ge=2,
        le=50,
        description="Histogram bins of a numeric (regression) target",
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
        params.target_bins,
    )


def target_result(
    df: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    task: str = "auto",
    top_k: int = TOP_K,
    bins: int = FEATURE_BINS,
    random_state: int = 0,
    target_bins: int = FEATURE_BINS,
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
        "bins": bins,
        "target_bins": target_bins,
    }
    # Focused feature: the first picked one, else the best-ranked.
    focus = columns[0] if columns and columns[0] in picked else metrics["top_feature"]
    metrics["focus_feature"] = focus
    result = Result(metrics=metrics)
    result.add_table("ranking", ranking)
    if task == "classification":
        balance = class_balance(rows[target])
        result.metrics["n_classes"] = len(balance)
        result.metrics["minority_pct"] = float(balance["pct"].min())
        result.add_table("class_balance", balance)
        result.headline = _classification_details(
            result, rows, target, picked, kinds, focus, top_k, bins
        ) or (
            f"Top feature: {ranking['column'].iloc[0]}; "
            f"{len(balance)}-class classification ({float(balance['pct'].min()):.1f} % minority)"
        )
    else:
        y = rows[target].astype(float)
        result.metrics["target_mean"] = float(y.mean())
        result.metrics["target_std"] = float(y.std()) if len(y) > 1 else 0.0
        result.headline = _regression_details(
            result, rows, target, picked, kinds, focus, top_k, bins
        ) or (
            f"Top feature: {ranking['column'].iloc[0]}; "
            f"regression target across {len(picked)} feature{'s' if len(picked) != 1 else ''}"
        )
    with plotly_lock:
        result.add_figure("Association with the target", _ranking_figure(ranking))
    if task != "classification":
        target_hist = _target_histogram(y, target, target_bins)
        result.add_table("target_histogram", target_hist)
        with plotly_lock:
            result.add_figure(
                f"{target} distribution",
                px.bar(
                    target_hist,
                    x="bin_left",
                    y="count",
                    labels={"bin_left": target, "count": "count"},
                ),
            )
    if not any(f.main for f in result.figures):
        result.figures[0].main = True
    if n_capped:
        result.text = (
            f"{n_capped} more eligible features not analysed (first "
            f"{MAX_COLUMNS}); pick columns to see them."
        )
    return result


@plotly_lock
def _ranking_figure(ranking: pd.DataFrame) -> go.Figure:
    """Horizontal bars of the mutual information, strongest on top."""
    shown = ranking.head(RANKING_SHOWN).iloc[::-1]
    fig = px.bar(
        shown,
        x="mutual_info",
        y="column",
        orientation="h",
        hover_data={"kind": True, "association": ":.3f", "measure": True},
        labels={"mutual_info": "mutual information", "column": ""},
    )
    fig.update_layout(height=max(300, 24 * len(shown) + 120))
    return fig


def _target_histogram(y: pd.Series, target: str, bins: int) -> pd.DataFrame:
    """Simple count histogram of a numeric target (shared edges for the figure)."""
    values = y.dropna().astype(float).to_numpy()
    if len(values) == 0:
        return pd.DataFrame(
            columns=["column", "bin_left", "bin_right", "count", "share"]
        )
    counts, edges = np.histogram(values, bins=bins)
    n = len(values)
    return pd.DataFrame(
        {
            "column": target,
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": counts,
            "share": counts / n if n else 0.0,
        }
    )


def _classification_details(result, rows, target, picked, kinds, col, top_k, bins):
    """Tables for every picked feature, figures for the focused one (``col``); returns
    its headline, None when it cannot be summarised."""
    numeric = [c for c in picked if kinds[c] == "numeric"]
    categorical = [c for c in picked if kinds[c] == "categorical"]
    stats = {c: numeric_by_class(rows, c, target) for c in numeric}
    rates = {c: category_class_rates(rows, c, target, top_k) for c in categorical}
    counts = {c: class_counts_by_category(rows, c, target, top_k) for c in categorical}
    counts_bin = {c: class_counts_by_bin(rows, c, target, bins) for c in numeric}
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
    if counts:
        result.add_table(
            "class_counts_by_category",
            pd.concat(counts.values(), ignore_index=True),
            "categorical",
        )
    if counts_bin:
        result.add_table(
            "class_counts_by_bin",
            pd.concat(counts_bin.values(), ignore_index=True),
            "numeric",
        )
    if col in counts_bin:
        table, key, group, what = counts_bin[col], "bin", "numeric", "bin"
    else:
        table, key, group, what = counts[col], "value", "categorical", "category"
    with plotly_lock:
        result.add_figure(
            f"{target} count by {col} {what}",
            _class_counts_figure(table, key, col, target, percent=False),
            group,
            main=True,
        )
        result.add_figure(
            f"{target} share by {col} {what} (%)",
            _class_counts_figure(table, key, col, target, percent=True),
            group,
        )
        if col in stats:
            result.add_figure(f"{col} by class", _box_figure(stats[col], col), group)
    return _class_headline(table, key, col, target)


@plotly_lock
def _class_counts_figure(
    table: pd.DataFrame, key: str, col: str, target: str, percent: bool
) -> go.Figure:
    """Bars per class along the feature groups: grouped row counts, or stacked
    100 % shares of each group."""
    order = list(dict.fromkeys(table[key]))
    fig = px.bar(
        table,
        x=key,
        y=f"pct_of_{key}" if percent else "count",
        color="class",
        category_orders={key: order},
        barmode="relative" if percent else "group",
        custom_data=["count"],
        labels={
            key: col,
            "class": target,
            "count": "rows",
            f"pct_of_{key}": f"% of the {key}'s rows",
        },
    )
    if percent:
        fig.update_traces(hovertemplate="%{x}<br>%{y:.1f} %  (%{customdata[0]} rows)")
        fig.update_yaxes(range=[0, 100], title=f"% of the {key}'s rows")
    else:
        fig.update_yaxes(title="number of rows")
    fig.update_xaxes(title=col)
    return fig


_GROUPS = {"bin": "bins", "value": "categories"}


def _big_enough(sizes: pd.Series) -> pd.Series:
    """Mask of the groups fit to bound a headline: at least max(5 rows, 5 % of the
    rows), neither ``other`` nor ``(missing)``."""
    floor = max(5, math.ceil(MIN_GROUP_SHARE * sizes.sum()))
    return (sizes >= floor) & ~sizes.index.isin([OTHER_LABEL, MISSING_LABEL])


def _class_headline(table: pd.DataFrame, key: str, col: str, target: str) -> str | None:
    """ "<target> rate ranges from a % (group) to b % (group) across <col>": the
    class whose rate varies the most (the last one on a tie: the positive class),
    over groups with enough rows (see ``_big_enough``); too few of them: a cautious
    "varies across" sentence."""
    rate = f"pct_of_{key}"
    sizes = table.groupby(key, sort=False)["count"].sum()
    if len(sizes) < 2:
        return None
    kept = sizes[_big_enough(sizes)]
    if len(kept) < 2:
        return f"{target} rate varies across {col} {_GROUPS[key]} (small sample)"
    sub = table[table[key].isin(kept.index)]
    spreads = {
        cls: round(g[rate].max() - g[rate].min(), 6) for cls, g in sub.groupby("class")
    }
    # max() keeps the first maximum: walk from the last class for the tie-break.
    best = max(sorted(spreads, key=natural_key, reverse=True), key=spreads.get)
    g = sub[sub["class"] == best].sort_values(rate, kind="stable")
    lo, hi = g.iloc[0], g.iloc[-1]
    name = f"{target} rate" if best in ("1", "True") else f"{target}={best} rate"
    return (
        f"{name} ranges from {lo[rate]:.0f} % ({lo[key]}) to "
        f"{hi[rate]:.0f} % ({hi[key]}) across {col}"
    )


def _regression_details(result, rows, target, picked, kinds, col, top_k, bins):
    """Tables for every picked feature, figure for the focused one (``col``); returns
    its headline, None when it cannot be summarised."""
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
    if col in binned:
        table = binned[col].dropna(subset=["feature_mean"])
        result.add_figure(
            f"mean {target} by {col} bin",
            lambda: _mean_curve(table, "feature_mean", col, target),
            "numeric",
            main=True,
        )
        return _mean_headline(table, "bin", col, target, True)
    result.add_figure(
        f"mean {target} by {col}",
        lambda: _mean_curve(means[col], "value", col, target),
        "categorical",
        main=True,
    )
    return _mean_headline(means[col], "value", col, target, False)


@plotly_lock
def _mean_curve(table: pd.DataFrame, x: str, col: str, target: str) -> go.Figure:
    """Mean target per bin / category with its inter-quartile band; the hover shows
    the rows per point. Categories are sorted by mean (a curve reads left to right)."""
    if x == "value":
        table = table.sort_values("mean_target", kind="stable")
    else:
        table = table.sort_values(x, kind="stable")
    label = table["bin"] if "bin" in table else table["value"]
    custom = np.column_stack(
        [label, table["count"], table["q1_target"], table["q3_target"]]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=table[x],
            y=table["q3_target"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table[x],
            y=table["q1_target"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=table[x],
            y=table["mean_target"],
            mode="lines+markers",
            line={"color": "rgb(99, 110, 250)"},
            customdata=custom,
            hovertemplate=(
                "%{customdata[0]}<br>mean %{y:.4g}"
                "<br>IQR %{customdata[2]:.4g} to %{customdata[3]:.4g}"
                "<br>%{customdata[1]} rows<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(xaxis_title=col, yaxis_title=f"mean {target} (band = IQR)")
    return fig


def _compact(v: float) -> str:
    """120000 -> 120k, 1.5e6 -> 1.5M, else 3 significant digits."""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= size:
            return f"{v / size:.3g}{suffix}"
    return f"{v:.3g}"


def _mean_headline(
    table: pd.DataFrame, key: str, col: str, target: str, ordered: bool
) -> str | None:
    """Numeric feature (ordered bins): "Mean <t> rises / falls from a to b across
    <col> bins"; categorical: "... ranges from a (cat) to b (cat) across <col>"."""
    table = table.dropna(subset=["mean_target"])
    if len(table) < 2:
        return None
    kept = table[_big_enough(table.set_index(key)["count"]).to_numpy()]
    if len(kept) < 2:
        return f"Mean {target} varies across {col} {_GROUPS[key]} (small sample)"
    if ordered:
        first, last = kept["mean_target"].iloc[0], kept["mean_target"].iloc[-1]
        verb = "rises" if last >= first else "falls"
        return (
            f"Mean {target} {verb} from {_compact(first)} to {_compact(last)} "
            f"across {col} bins"
        )
    lo = kept.loc[kept["mean_target"].idxmin()]
    hi = kept.loc[kept["mean_target"].idxmax()]
    return (
        f"Mean {target} ranges from {_compact(lo['mean_target'])} ({lo[key]}) to "
        f"{_compact(hi['mean_target'])} ({hi[key]}) across {col}"
    )


@plotly_lock
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
