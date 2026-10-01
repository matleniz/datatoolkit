"""Build a Plotly Express figure from a DataFrame for the ``chart`` key.

Pure pandas / numpy / plotly — no ``Result``. Sampling and OLS trendlines are
done here so fronts only render the figure JSON.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.columns import is_numeric, numeric_feature
from dtk_engine.result import plotly_lock

ChartType = Literal[
    "histogram",
    "box",
    "violin",
    "bar",
    "count",
    "scatter",
    "line",
    "heatmap",
    "density_heatmap",
    "pie",
    "scatter_matrix",
]
Agg = Literal["count", "mean", "sum", "median"]

DEFAULT_SAMPLE_SIZE = 10_000
# Scatter-matrix dimensions when none are picked (first numeric features).
SCATTER_MATRIX_MAX = 6
# Plotly histfunc for bar/line aggregations (median via groupby + px).
_HISTFUNC = {"count": "count", "mean": "avg", "sum": "sum"}
_AGG_FN = {"count": "size", "mean": "mean", "sum": "sum", "median": "median"}

# Numeric color columns with at most this many distinct values are treated as
# categorical (discrete legend / traces instead of a continuous coloraxis).
DISCRETE_COLOR_MAX_UNIQUE = 10

OP = "chart"


def _should_cast_color_discrete(series: pd.Series) -> bool:
    """True when color is bool or low-cardinality numeric (<= 10 distinct values)."""
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= DISCRETE_COLOR_MAX_UNIQUE
    return False


def _format_discrete_color(series: pd.Series) -> pd.Series:
    """Format low-cardinality numeric or boolean series to clean string labels."""
    if pd.api.types.is_float_dtype(series):
        valid = series.dropna()
        if len(valid) > 0 and (valid % 1 == 0).all():
            return series.map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return series.astype(str)


def sample_frame(
    df: pd.DataFrame,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    random_state: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Return ``(frame, n_sampled_out)``. ``sample_size=None`` keeps every row."""
    if sample_size is None or len(df) <= sample_size:
        return df, 0
    return df.sample(sample_size, random_state=random_state), len(df) - sample_size


@plotly_lock
def build_figure(
    df: pd.DataFrame,
    chart: ChartType,
    *,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    facet_row: str | None = None,
    facet_col: str | None = None,
    size: str | None = None,
    columns: list[str] | None = None,
    agg: Agg | None = None,
    trendline: bool = False,
    log_x: bool = False,
    log_y: bool = False,
    bins: int = 30,
) -> go.Figure:
    """Build a Plotly figure for ``chart`` on ``df`` (already sampled if needed)."""
    cols = _used_columns(chart, x, y, color, facet_row, facet_col, size, columns)
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise KeyParamsError(f"{OP}: columns not in the frame {absent}")

    if color is not None:
        series = df[color]
        if _should_cast_color_discrete(series) and not (
            chart == "scatter_matrix" and columns and color in columns
        ):
            df = df.copy()
            df[color] = _format_discrete_color(series)

    spec = _BUILDERS.get(chart)
    if spec is None:  # pragma: no cover — Literal exhaustiveness
        raise KeyParamsError(f"{OP}: unknown chart type {chart!r}")
    builder, names = spec
    args = {
        "chart": chart, "x": x, "y": y, "color": color, "facet_row": facet_row,
        "facet_col": facet_col, "size": size, "columns": columns or [],
        "agg": agg, "trendline": trendline, "bins": bins,
    }
    fig = builder(df, **{name: args[name] for name in names.split()})

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def _used_columns(
    chart: ChartType,
    x: str | None,
    y: str | None,
    color: str | None,
    facet_row: str | None,
    facet_col: str | None,
    size: str | None,
    columns: list[str] | None,
) -> list[str]:
    if chart == "scatter_matrix":
        cols = list(columns or [])
        if color is not None:
            cols.append(color)
        return list(dict.fromkeys(cols))
    return [c for c in (x, y, color, facet_row, facet_col, size) if c is not None]


def _require_x(chart: ChartType, x: str | None) -> None:
    if x is None:
        raise KeyParamsError(f"{OP}: chart {chart!r} requires x")


def _require_xy(chart: ChartType, x: str | None, y: str | None) -> None:
    if x is None or y is None:
        raise KeyParamsError(f"{OP}: chart {chart!r} requires x and y")


def _box_violin(df, *, chart, x, y, color, facet_row, facet_col):
    _require_x(chart, x)
    plot = getattr(px, chart)
    return plot(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col)


def _histogram(df, *, x, y, color, facet_row, facet_col, bins, agg):
    _require_x("histogram", x)
    kwargs: dict = {
        "x": x,
        "color": color,
        "facet_row": facet_row,
        "facet_col": facet_col,
        "nbins": bins,
    }
    if y is not None:
        histfunc = _HISTFUNC.get(agg or "sum", "sum")
        kwargs["y"] = y
        kwargs["histfunc"] = histfunc
    return px.histogram(df, **kwargs)


def _bar_or_count(df, *, chart, x, y, color, facet_row, facet_col, agg):
    _require_x(chart, x)
    effective = "count" if chart == "count" else (agg or ("count" if y is None else "mean"))
    if effective == "count" or y is None:
        return px.histogram(
            df,
            x=x,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            histfunc="count",
        )
    if effective in _HISTFUNC:
        return px.histogram(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            histfunc=_HISTFUNC[effective],
        )
    # median (and any future agg) via groupby + bar
    group_cols = [c for c in (x, color, facet_row, facet_col) if c is not None]
    grouped = (
        df.groupby(group_cols, dropna=False)[y]
        .agg(_AGG_FN[effective])
        .reset_index(name=y)
    )
    return px.bar(
        grouped,
        x=x,
        y=y,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
    )


def _scatter(df, *, x, y, color, facet_row, facet_col, size, trendline):
    _require_xy("scatter", x, y)
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        size=size,
    )
    if trendline:
        _add_ols_trendlines(fig, df, x, y, color=color)
    return fig


def _line(df, *, x, y, color, facet_row, facet_col, agg):
    _require_x("line", x)
    effective = agg or ("count" if y is None else "mean")
    group_cols = [c for c in (x, color, facet_row, facet_col) if c is not None]
    if effective == "count" or y is None:
        grouped = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
        y_col = "count"
    else:
        grouped = (
            df.groupby(group_cols, dropna=False)[y]
            .agg(_AGG_FN[effective])
            .reset_index(name=y)
        )
        y_col = y
    grouped = grouped.sort_values(x)
    return px.line(
        grouped,
        x=x,
        y=y_col,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        markers=True,
    )


def _density_heatmap(df, *, x, y, facet_row, facet_col, bins):
    _require_xy("density_heatmap", x, y)
    # Count of (x, y) pairs in each bin; both axes are coordinates.
    return px.density_heatmap(
        df,
        x=x,
        y=y,
        facet_row=facet_row,
        facet_col=facet_col,
        nbinsx=bins,
        nbinsy=bins,
        histfunc="count",
    )


def _pie(df, *, x, y, color, agg):
    # names = x (or color), values = y aggregated or counts of names
    names = x or color
    if names is None:
        raise KeyParamsError(f"{OP}: chart 'pie' requires x (slice names)")
    if y is None or (agg or "count") == "count":
        counts = df[names].value_counts(dropna=False).reset_index()
        counts.columns = [names, "count"]
        return px.pie(counts, names=names, values="count")
    effective = agg or "sum"
    grouped = df.groupby(names, dropna=False)[y].agg(_AGG_FN[effective]).reset_index()
    return px.pie(grouped, names=names, values=y)


def _scatter_matrix(df, *, columns: list[str], color: str | None):
    if columns:
        dims = columns
        bad = [c for c in dims if not is_numeric(df[c])]
        if bad:
            raise KeyParamsError(
                f"{OP}: scatter_matrix columns must be numeric, got {bad}"
            )
    else:
        dims = [
            str(c)
            for c in df.columns
            if c != color and numeric_feature(df[c])
        ][:SCATTER_MATRIX_MAX]
        if len(dims) < 2:
            raise KeyParamsError(
                f"{OP}: scatter_matrix needs at least 2 numeric columns"
            )
    return px.scatter_matrix(df, dimensions=dims, color=color)


def _add_ols_trendlines(
    fig: go.Figure,
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
) -> None:
    """Overlay ordinary-least-squares lines (numpy polyfit; no statsmodels)."""
    if color is None:
        _add_one_ols(fig, df, x, y, name="OLS")
        return
    for value, part in df.groupby(color, dropna=False):
        _add_one_ols(fig, part, x, y, name=f"OLS ({value})")


def _add_one_ols(
    fig: go.Figure, df: pd.DataFrame, x: str, y: str, *, name: str
) -> None:
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 2:
        return
    xv = pair[x].to_numpy(dtype=float)
    yv = pair[y].to_numpy(dtype=float)
    if np.allclose(xv, xv[0]):
        return
    coef = np.polyfit(xv, yv, 1)
    xs = np.linspace(float(xv.min()), float(xv.max()), 50)
    ys = np.polyval(coef, xs)
    fig.add_trace(
        go.Scatter(x=xs, y=ys, mode="lines", name=name, showlegend=True)
    )


# chart type -> (builder, the arguments it takes from build_figure's)
_XYCF = "x y color facet_row facet_col"
_BUILDERS = {
    "histogram": (_histogram, f"{_XYCF} bins agg"),
    "box": (_box_violin, f"chart {_XYCF}"),
    "violin": (_box_violin, f"chart {_XYCF}"),
    "bar": (_bar_or_count, f"chart {_XYCF} agg"),
    "count": (_bar_or_count, f"chart {_XYCF} agg"),
    "scatter": (_scatter, f"{_XYCF} size trendline"),
    "line": (_line, f"{_XYCF} agg"),
    "heatmap": (_density_heatmap, "x y facet_row facet_col bins"),
    "density_heatmap": (_density_heatmap, "x y facet_row facet_col bins"),
    "pie": (_pie, "x y color agg"),
    "scatter_matrix": (_scatter_matrix, "columns color"),
}
