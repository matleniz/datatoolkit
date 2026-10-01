"""Correlations: numeric correlation heatmap over picked columns + correlated pairs."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.columns import is_numeric, numeric_feature, pick_columns
from dtk_engine.ops.correlation import (
    DEFAULT_THRESHOLD,
    correlated_pairs,
    correlation_matrix,
    off_diagonal,
    target_correlation,
    undefined_columns,
)
from dtk_engine.params import SourceParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import load

# Columns in the matrix when none are picked (the first numeric ones).
MAX_COLUMNS = 50
# Heatmap cells carry their value up to this many columns.
ANNOTATE_MAX = 12
# Pairs drawn in the "Strongest pairs" bar chart.
TOP_PAIRS = 10
# Longest axis label before truncation.
LABEL_MAX = 18


class Params(SourceParams):
    columns: list[str] = columns_field(
        f"Numeric columns to correlate (empty = every numeric column but ids and the "
        f"target, first {MAX_COLUMNS})",
        dtype="numeric",
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="pearson (linear) | spearman (monotonic, on ranks) | "
        "kendall (ordinal); the drop_correlated op uses pearson",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        gt=0,
        le=1,
        description="Pairs with |corr| >= this are listed (same rule as drop_correlated)",
    )
    top_n: int = Field(
        default=15,
        ge=2,
        le=50,
        description="Columns drawn in the heatmap: past this many, keep those with "
        "the highest |corr| with another column (tables keep every column)",
    )
    target: str | None = column_field(
        None,
        "Optional label column: excluded from the matrix; in each pair the column "
        "more correlated with it is the one to keep",
    )


@key(
    id="correlations",
    title="Correlations",
    category="analysis",
    description="Correlation matrix (Pearson, Spearman or Kendall) of picked "
    "numeric columns as a heatmap, plus the pairs above a threshold with the "
    "column drop_correlated would keep, and the matching suggested step.",
)
def run(params: Params) -> Result:
    return correlations_result(
        load(params.source),
        params.columns,
        params.method,
        params.threshold,
        params.target,
        params.top_n,
    )


def correlations_result(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    threshold: float = DEFAULT_THRESHOLD,
    target: str | None = None,
    top_n: int = 15,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.correlations``)."""
    op = "correlations"
    if not 2 <= top_n <= MAX_COLUMNS:
        raise KeyParamsError(f"{op}: top_n must be between 2 and {MAX_COLUMNS}")
    if target is not None and target not in df.columns:
        raise KeyParamsError(f"{op}: target {target!r} not in the frame")
    picked, n_capped = pick_columns(
        df,
        columns or [],
        op,
        exclude=[target],
        eligible=numeric_feature,
        required=is_numeric,
        requirement="are not numeric (encode first)",
        cap=MAX_COLUMNS,
    )
    corr = correlation_matrix(df, picked, method)
    to_target = (
        target_correlation(df, picked, target, method) if target is not None else None
    )
    pairs = correlated_pairs(df, corr, threshold, to_target)
    undefined = undefined_columns(corr)
    max_abs = off_diagonal(corr).abs().max().max()

    lines = []
    if n_capped:
        lines.append(
            f"{n_capped} more numeric columns not in the matrix (first "
            f"{MAX_COLUMNS}); pick columns to see them."
        )
    if undefined:
        lines.append(f"Correlation undefined (constant or no overlap): {undefined}")
    steps = _suggested_steps(pairs, picked, method, threshold, target)
    if len(steps):
        lines.append(
            "Suggested step (a 'both' step, fitted on train): drop_correlated "
            f"threshold {threshold} drops one column of each pair."
        )
    elif len(pairs):
        lines.append(
            "drop_correlated uses pearson: rerun with method=pearson for the "
            "matching step."
        )

    shown = _top_columns(corr, top_n)
    if len(shown) < len(picked):
        lines.append(
            f"Heatmap shows the {len(shown)} of {len(picked)} columns with the "
            f"highest |corr| with another (top_n={top_n}); tables keep all."
        )
    headline = _headline(corr, pairs, threshold)
    result = Result(
        headline=headline,
        metrics={
            "method": method,
            "n_columns": len(picked),
            "n_columns_capped": n_capped,
            "n_columns_shown": len(shown),
            "n_pairs": len(pairs),
            "threshold": threshold,
            "max_abs_corr": 0.0 if pd.isna(max_abs) else float(max_abs),
            "n_undefined": len(undefined),
        },
        text="\n".join(lines),
    )
    result.add_table(f"correlated_pairs (|corr| >= {threshold})", pairs)
    result.add_table("matrix", corr.rename_axis("column").reset_index())
    if to_target is not None:
        result.add_table(
            "abs_corr_with_target",
            to_target.rename("abs_corr").rename_axis("column").reset_index(),
        )
    result.add_table("suggested_steps", steps, kind="steps")
    with plotly_lock:
        result.add_figure(
            f"{method} correlation", _heatmap(corr.loc[shown, shown]), main=True
        )
        strongest = _strongest_pairs(corr)
        if len(strongest):
            result.add_figure("Strongest pairs", _pairs_bar(strongest))
    return result


def _top_columns(corr: pd.DataFrame, top_n: int) -> list[str]:
    """Up to ``top_n`` columns with the highest max |corr| with another, in order."""
    cols = list(corr.columns)
    if len(cols) <= top_n:
        return cols
    strength = off_diagonal(corr).abs().max().fillna(0.0)
    keep = set(strength.sort_values(ascending=False, kind="stable").index[:top_n])
    return [c for c in cols if c in keep]


def _short(label: str) -> str:
    return label if len(label) <= LABEL_MAX else label[: LABEL_MAX - 1] + "…"


def _heatmap(sub: pd.DataFrame):
    """Lower triangle only (no diagonal), diverging scale centred on 0."""
    n = len(sub)
    if n < 2:
        return px.imshow(sub, zmin=-1, zmax=1, color_continuous_scale="RdBu")
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    tri = sub.where(mask).iloc[1:, :-1]
    labels_x = [_short(c) for c in tri.columns]
    labels_y = [_short(c) for c in tri.index]
    fig = px.imshow(
        tri.to_numpy(),
        x=labels_x,
        y=labels_y,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        text_auto=".2f" if n <= ANNOTATE_MAX else False,
        aspect="auto",
    )
    fig.update_traces(
        customdata=[[[r, c] for c in tri.columns] for r in tri.index],
        hovertemplate="%{customdata[0]} × %{customdata[1]}<br>corr %{z:.2f}"
        "<extra></extra>",
    )
    fig.update_xaxes(tickangle=-45, side="bottom", showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(coloraxis_colorbar={"title": "corr"})
    return fig


def _strongest_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """The ``TOP_PAIRS`` pairs with the highest |corr| (columns a, b, corr)."""
    vals = corr.to_numpy(dtype=float)
    i, j = np.triu_indices(len(corr), k=1)
    df = pd.DataFrame(
        {"a": corr.columns[i], "b": corr.columns[j], "corr": vals[i, j]}
    ).dropna()
    df["abs"] = df["corr"].abs()
    return df.sort_values("abs", ascending=False, kind="stable").head(TOP_PAIRS)


def _pairs_bar(pairs: pd.DataFrame):
    pairs = pairs.iloc[::-1]  # strongest on top of a horizontal bar chart
    label = [f"{_short(a)} × {_short(b)}" for a, b in zip(pairs["a"], pairs["b"])]
    sign = np.where(pairs["corr"] >= 0, "positive", "negative")
    fig = px.bar(
        x=pairs["abs"],
        y=label,
        color=sign,
        color_discrete_map={"positive": "#2166ac", "negative": "#b2182b"},
        orientation="h",
        labels={"x": "|corr|", "y": "", "color": "sign"},
        text=[f"{v:+.2f}" for v in pairs["corr"]],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 1.05])
    fig.update_layout(legend_title_text="sign")
    return fig


def _headline(corr: pd.DataFrame, pairs: pd.DataFrame, threshold: float) -> str:
    if len(corr) < 2:
        return ""
    off = off_diagonal(corr)
    top_pair = None
    max_val = -1.0
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            val = off.loc[a, b]
            if not pd.isna(val) and abs(float(val)) > max_val:
                max_val = abs(float(val))
                top_pair = (a, b, float(val))
    if top_pair is None:
        return ""
    a, b, val = top_pair
    n = len(pairs)
    return (
        f"Strongest pair: {a} / {b} ({val:.2f}); "
        f"{n} pair{'s' if n != 1 else ''} with |corr| >= {threshold}"
    )


def _suggested_steps(
    pairs: pd.DataFrame,
    columns: list[str],
    method: str,
    threshold: float,
    target: str | None,
) -> pd.DataFrame:
    fields = ["order", "op", "target", "params", "why"]
    if pairs.empty or method != "pearson":
        return pd.DataFrame(columns=fields)
    params: dict = {"threshold": threshold, "columns": columns}
    if target is not None:
        params["target"] = target
    return pd.DataFrame(
        [
            {
                "order": 1,
                "op": "drop_correlated",
                "target": "both",
                "params": params,
                "why": f"drop one column of each of the {len(pairs)} pairs "
                f"with |corr| >= {threshold}",
            }
        ],
        columns=fields,
    )
