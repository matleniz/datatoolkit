"""Custom chart: build a Plotly Express figure from typed params on a source frame."""

from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.chart import (
    DEFAULT_SAMPLE_SIZE,
    Agg,
    ChartType,
    build_figure,
    sample_frame,
)
from dtk_engine.ops.profile import HIST_BINS
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    chart: ChartType = Field(
        default="histogram",
        description="Chart type: histogram, box, violin, bar, count, scatter, "
        "line, heatmap, density_heatmap, pie, scatter_matrix",
    )
    x: str | None = column_field(
        "Age",
        "X axis / category / slice names (histogram, box, bar, pie, …)",
    )
    y: str | None = column_field(
        None,
        "Y axis / value column (scatter, box, bar with agg, line, density)",
    )
    color: str | None = column_field(None, "Color / group column")
    facet_row: str | None = column_field(None, "Facet rows by this column")
    facet_col: str | None = column_field(None, "Facet columns by this column")
    size: str | None = column_field(
        None, "Marker size column (scatter only)", dtype="numeric"
    )
    columns: list[str] = columns_field(
        "Dimensions for scatter_matrix (empty = first numeric features)",
        dtype="numeric",
    )
    agg: Agg | None = Field(
        default=None,
        description="Aggregation for bar / count / line (count, mean, sum, median); "
        "null picks count when y is absent, mean otherwise",
    )
    trendline: bool = Field(
        default=False,
        description="Overlay an OLS trendline on scatter (numpy polyfit)",
    )
    log_x: bool = Field(default=False, description="Log scale on the x axis")
    log_y: bool = Field(default=False, description="Log scale on the y axis")
    bins: int = Field(
        default=HIST_BINS, ge=2, le=200, description="Bins for histogram / density"
    )
    sample_size: int | None = Field(
        default=DEFAULT_SAMPLE_SIZE,
        ge=1,
        description="Cap rows before plotting (null = no sampling); sampling is "
        "engine-side so the figure JSON stays small",
    )


@key(
    id="chart",
    title="Chart",
    category="analysis",
    description="Build a custom Plotly Express chart (histogram, box, violin, bar, "
    "scatter, line, density heatmap, pie, scatter matrix) on a source frame, "
    "with optional sampling and OLS trendline.",
)
def run(params: Params) -> Result:
    return chart_result(
        load(params.source),
        params.chart,
        x=params.x,
        y=params.y,
        color=params.color,
        facet_row=params.facet_row,
        facet_col=params.facet_col,
        size=params.size,
        columns=params.columns,
        agg=params.agg,
        trendline=params.trendline,
        log_x=params.log_x,
        log_y=params.log_y,
        bins=params.bins,
        sample_size=params.sample_size,
    )


def chart_result(
    df,
    chart: ChartType = "histogram",
    *,
    x: str | None = "Age",
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
    bins: int = HIST_BINS,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.chart``)."""
    frame, n_dropped = sample_frame(df, sample_size)
    fig = build_figure(
        frame,
        chart,
        x=x,
        y=y,
        color=color,
        facet_row=facet_row,
        facet_col=facet_col,
        size=size,
        columns=columns,
        agg=agg,
        trendline=trendline,
        log_x=log_x,
        log_y=log_y,
        bins=bins,
    )
    title = _title(chart, x, y)
    result = Result(
        metrics={
            "chart": chart,
            "n_rows": len(frame),
            "n_rows_source": len(df),
            "n_sampled_out": n_dropped,
            "trendline": int(trendline),
        },
        text=_text(n_dropped, sample_size),
    )
    result.add_figure(title, fig)
    return result


def _title(chart: str, x: str | None, y: str | None) -> str:
    if chart == "scatter_matrix":
        return "scatter matrix"
    if x and y:
        return f"{chart}: {y} vs {x}"
    if x:
        return f"{chart}: {x}"
    return chart


def _text(n_dropped: int, sample_size: int | None) -> str:
    if not n_dropped:
        return ""
    return (
        f"Sampled {sample_size} rows for the figure "
        f"({n_dropped} rows left out); raise sample_size or set it to null "
        "to plot more."
    )
