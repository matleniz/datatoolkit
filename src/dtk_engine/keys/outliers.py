"""Outliers: per-column IQR / z-score, multivariate IsolationForest, action table."""

from typing import Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import Field

from dtk_engine.ops.columns import is_numeric, pick_columns
from dtk_engine.ops.outliers import (
    ACTION_TABLE,
    BOX_FIELDS,
    IQR_K,
    Z_THRESHOLD,
    box_stats,
    flagged_rows,
    isolation_forest,
    numeric_columns,
    outlier_points,
    univariate_outliers,
)
from dtk_engine.params import SourceParams, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import load

OutlierMethod = Literal["all", "iqr", "zscore", "isolation_forest"]


class Params(SourceParams):
    columns: list[str] = columns_field(
        "Numeric columns to screen (empty = every numeric column but ids / constants)",
        dtype="numeric",
    )
    method: OutlierMethod = Field(
        default="all", description="Which detector(s) to run: all | iqr | zscore | isolation_forest"
    )
    iqr_k: float = Field(default=IQR_K, gt=0, description="IQR fence multiplier (1.5 = Tukey)")
    z_threshold: float = Field(
        default=Z_THRESHOLD, gt=0, description="Flag values with |z| above this"
    )
    contamination: float = Field(
        default=0.01,
        gt=0,
        le=0.5,
        description="Expected share of anomalous rows for the IsolationForest "
        "(explicit, no 'auto')",
    )
    random_state: int = Field(default=0, description="IsolationForest seed")


@key(
    id="outliers",
    title="Outliers",
    category="analysis",
    description="Per numeric column IQR fences and |z| counts, a multivariate "
    "IsolationForest flagging anomalous rows, and what to do with them "
    "(remove / clip / keep / transform).",
)
def run(params: Params) -> Result:
    return outliers_result(
        load(params.source),
        params.iqr_k,
        params.z_threshold,
        params.contamination,
        params.random_state,
        params.columns,
        params.method,
    )


def outliers_result(
    df: pd.DataFrame,
    iqr_k: float = IQR_K,
    z_threshold: float = Z_THRESHOLD,
    contamination: float = 0.01,
    random_state: int = 0,
    columns: list[str] | None = None,
    method: OutlierMethod = "all",
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.outliers``)."""
    picked = _pick(df, columns)
    run_iqr = method in ("all", "iqr")
    run_if = method in ("all", "isolation_forest")
    table = _univariate_table(df, picked, iqr_k, z_threshold, run_iqr, method)
    scores = (
        isolation_forest(df, picked, contamination, random_state)
        if run_if
        else pd.DataFrame({"score": [], "flagged": []}, dtype=float)
    )
    flagged = (
        flagged_rows(df, scores, picked)
        if run_if
        else pd.DataFrame(columns=["row", "score", *picked])
    )
    n_flagged = int(scores["flagged"].sum()) if len(scores) else 0
    boxes = box_stats(df, picked, iqr_k) if run_iqr else pd.DataFrame(columns=BOX_FIELDS)
    result = Result(
        headline=_headline(picked, table, boxes, n_flagged, run_iqr),
        metrics={
            "n_rows": len(df),
            "n_numeric_columns": len(picked),
            "n_columns_with_iqr_outliers": int((table["n_iqr"] > 0).sum()),
            "n_columns_with_z_outliers": int((table["n_z"] > 0).sum()),
            "n_rows_flagged": n_flagged,
            "contamination": contamination,
            "method": method,
        },
        text=ACTION_TABLE,
    )
    result.add_table("outliers_per_column", table)
    result.add_table("flagged_rows", flagged)
    if run_iqr:
        result.add_table("box_stats", boxes)
    has_main = _add_iqr_figures(result, df, picked, table, boxes) if run_iqr else False
    if run_if and len(scores):
        with plotly_lock:
            fig = px.histogram(
                scores,
                x="score",
                color="flagged",
                labels={"score": "anomaly score", "flagged": "flagged"},
            )
        result.add_figure("IsolationForest scores", fig, main=not has_main)
    return result


def _pick(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    if not columns:
        return numeric_columns(df)
    picked, _ = pick_columns(
        df, columns, "outliers", required=is_numeric, requirement="must be numeric"
    )
    return picked


def _univariate_table(df, picked, iqr_k, z_threshold, run_iqr, method) -> pd.DataFrame:
    # Always built (empty columns ok); zero out unused detectors.
    table = univariate_outliers(df, picked, iqr_k, z_threshold)
    if not run_iqr:
        table = table.assign(
            n_iqr=0, pct_iqr=0.0, lower_fence=float("nan"), upper_fence=float("nan")
        )
    if method not in ("all", "zscore"):
        table = table.assign(n_z=0, pct_z=0.0)
    return table


def _add_iqr_figures(result, df, picked, table, boxes) -> bool:
    """IQR figures (single box plot or per-column bars); True when a main one is added."""
    hit = table[table["n_iqr"] > 0].sort_values("pct_iqr", ascending=False)
    if len(picked) == 1:
        col = picked[0]
        result.add_figure(
            f"{col}: box plot" + (" with outliers" if len(hit) else ""),
            lambda: _single_box(df, col, boxes.iloc[0]),
            main=True,
        )
        return True
    if not len(hit):
        return False
    result.add_figure("% outliers per column (IQR)", lambda: _pct_bars(hit), main=True)
    top = boxes.set_index("column").loc[hit["column"].head(MULTIPLES)]
    result.add_figure(
        "Box plots of the most affected columns", lambda: _multiples(df, top.reset_index())
    )
    return True


MULTIPLES = 6
_OUTLIER_COLOR = "#d62728"
_BOX_COLOR = "#1f77b4"


def _fmt(x: float) -> str:
    return f"{x:,.4g}"


def _single_box(df: pd.DataFrame, col: str, row: pd.Series) -> go.Figure:
    lo, hi = row["lower_fence"], row["upper_fence"]
    pts = outlier_points(df, col, lo, hi)
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=[col],
            orientation="h",
            q1=[row["q1"]],
            median=[row["median"]],
            q3=[row["q3"]],
            lowerfence=[row["lower_whisker"]],
            upperfence=[row["upper_whisker"]],
            marker_color=_BOX_COLOR,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if len(pts):
        fig.add_trace(
            go.Scatter(
                x=pts.tolist(),
                y=[col] * len(pts),
                mode="markers",
                marker={"color": _OUTLIER_COLOR, "size": 8, "opacity": 0.7},
                name="outliers",
                hovertemplate="%{x}<extra>outlier</extra>",
                showlegend=False,
            )
        )
    values = df[col].dropna().astype(float)
    vmin, vmax = values.min(), values.max()
    for side, fence in (("low", lo), ("high", hi)):
        # A fence past the data range only stretches the axis, unless there is
        # nothing to flag: then both fences show why the column is clean.
        if vmin <= fence <= vmax or not (row["n_below"] or row["n_above"]):
            fig.add_vline(
                x=fence,
                line={"dash": "dot", "color": _OUTLIER_COLOR},
                annotation_text=f"{side} fence {_fmt(fence)}",
                annotation_position="top",
            )
    fig.update_layout(
        xaxis_title=col, yaxis_title=None, yaxis_showticklabels=False, showlegend=False
    )
    return fig


def _pct_bars(hit: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=hit["pct_iqr"].tolist(),
            y=hit["column"].tolist(),
            orientation="h",
            marker_color=_BOX_COLOR,
            text=[f"{p:.1f} %" for p in hit["pct_iqr"]],
            textposition="outside",
            customdata=hit["n_iqr"].tolist(),
            hovertemplate="%{y}: %{customdata} values (%{x:.1f} %)<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="% of values outside the IQR fences",
        yaxis_title=None,
        yaxis={"autorange": "reversed"},
        showlegend=False,
    )
    return fig


def _multiples(df: pd.DataFrame, top: pd.DataFrame) -> go.Figure:
    n = len(top)
    cols = min(3, n)
    rows = -(-n // cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(top["column"]))
    for i, row in enumerate(top.itertuples(index=False)):
        r, c = divmod(i, cols)
        name = row.column
        pts = outlier_points(df, name, row.lower_fence, row.upper_fence)
        fig.add_trace(
            go.Box(
                x=[name],
                q1=[row.q1],
                median=[row.median],
                q3=[row.q3],
                lowerfence=[row.lower_whisker],
                upperfence=[row.upper_whisker],
                marker_color=_BOX_COLOR,
                hoverinfo="skip",
            ),
            row=r + 1,
            col=c + 1,
        )
        if len(pts):
            fig.add_trace(
                go.Scatter(
                    x=[name] * len(pts),
                    y=pts.tolist(),
                    mode="markers",
                    marker={"color": _OUTLIER_COLOR, "size": 6, "opacity": 0.7},
                    hovertemplate="%{y}<extra>outlier</extra>",
                ),
                row=r + 1,
                col=c + 1,
            )
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(showlegend=False, height=300 * rows)
    return fig


def _headline(
    picked: list[str], table: pd.DataFrame, boxes: pd.DataFrame, n_rows_flagged: int, run_iqr: bool
) -> str:
    hit = table[table["n_iqr"] > 0]
    top = hit.sort_values("pct_iqr", ascending=False).iloc[0] if len(hit) else None
    single = len(picked) == 1
    rules = [
        (not picked, lambda: ""),
        (
            run_iqr and hit.empty and single,
            lambda: (
                f"No outliers outside the IQR fences in {picked[0]} "
                f"(fences {_fmt(boxes.iloc[0]['lower_fence'])}"
                f"–{_fmt(boxes.iloc[0]['upper_fence'])})"
            ),
        ),
        (run_iqr and hit.empty, lambda: "No outliers outside the IQR fences"),
        (run_iqr and single, lambda: _single_hit_headline(top, boxes.iloc[0])),
        (
            run_iqr,
            lambda: (
                f"{len(hit)} column{'s' if len(hit) > 1 else ''} with outliers; "
                f"most: {top['column']} ({top['pct_iqr']:.1f} %)"
            ),
        ),
        (n_rows_flagged == 0, lambda: "No anomalous rows flagged"),
        (
            True,
            lambda: f"{n_rows_flagged} anomalous row{'s' if n_rows_flagged > 1 else ''} flagged",
        ),
    ]
    return next(msg() for ok, msg in rules if ok)


def _single_hit_headline(top: pd.Series, b: pd.Series) -> str:
    n = int(top["n_iqr"])
    where = []
    if b["n_above"]:
        where.append(f"above {_fmt(b['upper_fence'])}")
    if b["n_below"]:
        where.append(f"below {_fmt(b['lower_fence'])}")
    return (
        f"{n} outlier{'s' if n > 1 else ''} in {top['column']} "
        f"({top['pct_iqr']:.1f} %), {' and '.join(where)}"
    )
