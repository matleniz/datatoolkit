"""Column distribution: histograms / value counts of picked columns, optionally
train vs test and split by label or by any column."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field, model_validator

from dtk_engine.demo_data import TEST_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.columns import is_numeric, pick_columns, value_kind
from dtk_engine.ops.distribution import (
    GROUP,
    TARGET_BINS,
    TOP_K,
    by_binner,
    categorical_distribution,
    group_order,
    grouped_frame,
    label_binner,
    numeric_distribution,
    sample_scatter,
    vs_by_correlations,
)
from dtk_engine.params import SourceParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import CsvSource, SourceSpec, load

# Columns analysed when none are picked (the first eligible ones).
MAX_COLUMNS = 20

BinsSpec = int | Literal["auto"]
NormSpec = Literal["count", "density", "share"]


class Params(SourceParams):
    columns: list[str] = columns_field(
        f"Columns to describe (empty = every numeric / categorical / boolean "
        f"column, first {MAX_COLUMNS})"
    )
    compare: Literal["none", "train_vs_test"] = Field(
        default="none", description="train_vs_test: overlay `test` on `source` (same bins)"
    )
    test: SourceSpec = Field(
        default=CsvSource(path=TEST_CSV),
        description="Test source (read only when compare = train_vs_test)",
    )
    by: str | None = column_field(
        None,
        "Split each column by the classes of this column (top-k + (other) for "
        "categorical / low-cardinality; quantile bins for continuous numeric); "
        "independent from `target` / `by_label`",
    )
    by_label: bool = Field(
        default=False,
        description="Split each column by the classes of `target` (quantile bins "
        "for a numeric target); a frame without the target stays one group",
    )
    target: str | None = column_field(
        None, "Label column (read only by by_label, then not a described column)"
    )
    bins: BinsSpec = Field(
        default="auto",
        description="Histogram bins: 'auto' (Freedman–Diaconis / Sturges / "
        "integer-aligned) or an explicit count 2–200; ignored when bin_edges is set",
    )
    bin_edges: list[float] | None = Field(
        default=None,
        description="Explicit histogram edges (strictly increasing, length >= 2); "
        "overrides bins when set",
    )
    range_min_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Lower percentile for the histogram range (clip before binning)",
    )
    range_max_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Upper percentile for the histogram range (clip before binning)",
    )
    log_x: bool = Field(
        default=False, description="Bin log1p(x) on the x-axis (drops negatives from the histogram)"
    )
    log_y: bool = Field(
        default=False, description="Log scale on the y-axis of the histogram figure"
    )
    norm: NormSpec = Field(
        default="share", description="Histogram height: share of rows | raw count | density"
    )
    cumulative: bool = Field(
        default=False, description="Plot / report cumulative counts (or share) instead of per-bin"
    )
    top_k: int = Field(
        default=TOP_K, ge=1, le=100, description="Categories kept per column, rest (other)"
    )
    target_bins: int = Field(
        default=TARGET_BINS,
        ge=2,
        le=20,
        description="Quantile bins of a numeric target / by column when continuous",
    )

    @model_validator(mode="after")
    def _check_bins(self):
        if isinstance(self.bins, int) and not (2 <= self.bins <= 200):
            raise ValueError("bins must be 'auto' or an integer in [2, 200]")
        if self.range_min_pct > self.range_max_pct:
            raise ValueError("range_min_pct must be <= range_max_pct")
        if self.bin_edges is not None and len(self.bin_edges) < 2:
            raise ValueError("bin_edges needs at least two edges")
        return self


@key(
    id="column_distribution",
    title="Column distribution",
    category="analysis",
    description="Per picked column: histogram on shared bins + summary (numeric) or "
    "top-k value counts (categorical), side by side per group: train vs test "
    "and / or per label class / any `by` column.",
)
def run(params: Params) -> Result:
    test = load(params.test) if params.compare == "train_vs_test" else None
    return distribution_result(
        load(params.source),
        test,
        params.columns,
        params.target,
        params.by_label,
        params.bins,
        params.top_k,
        params.target_bins,
        params.by,
        params.bin_edges,
        params.range_min_pct,
        params.range_max_pct,
        params.log_x,
        params.log_y,
        params.norm,
        params.cumulative,
    )


def distribution_result(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    target: str | None = None,
    by_label: bool = False,
    bins: int | str = "auto",
    top_k: int = TOP_K,
    target_bins: int = TARGET_BINS,
    by: str | None = None,
    bin_edges: list[float] | None = None,
    range_min_pct: float = 0.0,
    range_max_pct: float = 100.0,
    log_x: bool = False,
    log_y: bool = False,
    norm: NormSpec = "share",
    cumulative: bool = False,
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.distribution``).

    ``target`` is only read with ``by_label`` (then it is not a described column).
    ``by`` splits by any column (top-k + other or quantile bins), independent of
    ``target``; mutually exclusive with ``by_label``.
    """
    _check_split(df, by, by_label, target)
    exclude = [c for c in (by, target if by_label else None) if c is not None]
    picked, n_capped = pick_columns(df, columns or [], _OP, exclude=exclude, cap=MAX_COLUMNS)
    frames = {"train": df, "test": test} if test is not None else {"all": df}
    group_col, binner = _group_spec(df, by, by_label, target, top_k, target_bins)
    data = grouped_frame(frames, picked, group_col, binner)
    kinds = {c: value_kind(df[c]) for c in picked}
    hist_opts = (bins, bin_edges, range_min_pct, range_max_pct, log_x)

    figures, hists, summaries, counts = _describe_all(data, picked, kinds, top_k, hist_opts)

    groups = group_order(data[GROUP])
    absent = [c for c in picked if c not in test.columns] if test is not None else []
    compare = "train_vs_test" if test is not None else "none"
    result = Result(
        headline=_headline(picked, kinds, compare, by, by_label, target, summaries, counts),
        metrics={
            "n_columns": len(picked),
            "n_numeric": sum(k == "numeric" for k in kinds.values()),
            "n_categorical": sum(k == "categorical" for k in kinds.values()),
            "n_columns_capped": n_capped,
            "n_groups": len(groups),
            "compare": compare,
            "by_label": target if by_label else "none",
            "by": by if by is not None else "none",
            "n_missing_in_test": len(absent),
            "bins": bins,
            "norm": norm,
            "cumulative": int(cumulative),
            "log_x": int(log_x),
            "log_y": int(log_y),
        },
        text=_text(n_capped, absent),
    )
    _add_tables(result, data, groups, picked, kinds, summaries, hists, counts)
    with plotly_lock:
        _add_figures(result, figures, norm, cumulative, log_y)
        if by is not None and is_numeric(df[by]):
            numeric_cols = [c for c in picked if kinds[c] == "numeric"]
            if numeric_cols:
                _add_vs_by(result, df, numeric_cols, by)
    return result


_OP = "column_distribution"


def _check_split(df: pd.DataFrame, by: str | None, by_label: bool, target: str | None) -> None:
    if by is not None and by_label:
        raise KeyParamsError(f"{_OP}: pass by or by_label, not both")
    if by is not None and by not in df.columns:
        raise KeyParamsError(f"{_OP}: by {by!r} not in the frame")
    if by_label:
        if target is None:
            raise KeyParamsError(f"{_OP}: by_label needs a target column")
        if target not in df.columns:
            raise KeyParamsError(f"{_OP}: target {target!r} not in the frame")


def _group_spec(df, by, by_label, target, top_k, target_bins):
    if by is not None:
        return by, by_binner(df[by], top_k, target_bins)
    if by_label:
        return target, label_binner(df[target], target_bins)
    return None, None


def _describe_numeric(data, col, top_k, hist_opts):
    try:
        hist, summary = numeric_distribution(data, col, *hist_opts)
    except ValueError as exc:
        raise KeyParamsError(f"{_OP}: {exc}") from exc
    return "numeric", hist, summary


def _describe_categorical(data, col, top_k, hist_opts):
    return "categorical", categorical_distribution(data, col, top_k), None


_DESCRIBE = {"numeric": _describe_numeric, "categorical": _describe_categorical}


def _describe_all(data, picked, kinds, top_k, hist_opts):
    """(figure items, histograms, numeric summaries, value counts) per picked column."""
    figures, hists, summaries, counts = [], [], [], []
    for col in picked:
        group, table, summary = _DESCRIBE[kinds[col]](data, col, top_k, hist_opts)
        figures.append((col, group, table))
        if summary is None:
            counts.append(table)
        else:
            hists.append(table)
            summaries.append(summary)
    return figures, hists, summaries, counts


def _add_tables(result, data, groups, picked, kinds, summaries, hists, counts) -> None:
    result.add_table(
        "columns", pd.DataFrame({"column": picked, "kind": [kinds[c] for c in picked]})
    )
    sizes = data[GROUP].value_counts()
    result.add_table(
        "groups", pd.DataFrame({GROUP: groups, "n_rows": [int(sizes[g]) for g in groups]})
    )
    if summaries:
        result.add_table("numeric_summary", pd.concat(summaries, ignore_index=True), "numeric")
        result.add_table("histograms", pd.concat(hists, ignore_index=True), "numeric")
    if counts:
        result.add_table("value_counts", pd.concat(counts, ignore_index=True), "categorical")


def _add_figures(result, figures, norm, cumulative, log_y) -> None:
    for i, (col, group, item) in enumerate(figures):
        if group == "numeric":
            fig = _histogram_figure(item, col, norm, cumulative, log_y)
        else:
            fig = _counts_figure(item, col)
        result.add_figure(col, fig, group, main=(i == 0))


def _add_vs_by(result: Result, df: pd.DataFrame, numeric_cols: list[str], by: str):
    corr = vs_by_correlations(df, numeric_cols, by)
    result.add_table("vs_by", corr, "numeric")
    if len(corr) == 1:
        result.metrics["pearson"] = float(corr["pearson"].iloc[0])
        result.metrics["spearman"] = float(corr["spearman"].iloc[0])
    for col in numeric_cols:
        sample = sample_scatter(df, col, by)
        if sample.empty:
            continue
        fig = px.scatter(sample, x=by, y=col, labels={by: by, col: col}, opacity=0.55)
        result.add_figure(f"{col} vs {by}", fig, "numeric")


def _y_column(norm: NormSpec, cumulative: bool) -> str:
    if cumulative:
        return "cumulative_count" if norm == "count" else "cumulative_share"
    return norm


def _y_title(norm: NormSpec, cumulative: bool) -> str:
    if cumulative:
        return "cumulative count" if norm == "count" else "cumulative share of rows"
    return {"count": "count", "density": "density", "share": "share of rows"}[norm]


@plotly_lock
def _histogram_figure(
    hist: pd.DataFrame,
    col: str,
    norm: NormSpec = "share",
    cumulative: bool = False,
    log_y: bool = False,
) -> go.Figure:
    y_col = _y_column(norm, cumulative)
    fig = go.Figure(
        [
            go.Bar(
                x=part["bin_left"],
                y=part[y_col],
                width=np.asarray(part["bin_right"] - part["bin_left"]),
                offset=0,
                name=str(group),
                opacity=0.55,
            )
            for group, part in hist.groupby(GROUP, sort=False)
        ]
    )
    fig.update_layout(
        barmode="overlay",
        xaxis_title=col,
        yaxis_title=_y_title(norm, cumulative),
        yaxis_type="log" if log_y else "linear",
    )
    return fig


@plotly_lock
def _counts_figure(table: pd.DataFrame, col: str):
    return px.bar(
        table,
        x="value",
        y="pct",
        color=GROUP,
        barmode="group",
        labels={"value": col, "pct": "% of the group's rows"},
    )


def _text(n_capped: int, absent: list[str]) -> str:
    lines = []
    if n_capped:
        lines.append(
            f"{n_capped} more eligible columns not shown (first {MAX_COLUMNS}); "
            "pick columns to see them."
        )
    if absent:
        lines.append(f"Not in test (shown as missing there): {absent}")
    return "\n".join(lines)


def _single_headline(
    col: str, kind: str, summaries: list[pd.DataFrame], counts: list[pd.DataFrame]
) -> str:
    rules = [
        (
            kind == "numeric" and bool(summaries),
            lambda: (
                f"{col}: numeric distribution "
                f"(mean {float(summaries[0]['mean'].iloc[0]):.2f}, "
                f"std {float(summaries[0]['std'].iloc[0]):.2f})"
            ),
        ),
        (
            kind == "categorical" and bool(counts),
            lambda: f"{col}: categorical distribution ({len(counts[0])} categories)",
        ),
        (True, lambda: f"{col}: distribution ({kind})"),
    ]
    return next(msg() for ok, msg in rules if ok)


def _headline(
    picked: list[str],
    kinds: dict[str, str],
    compare: str,
    by: str | None,
    by_label: bool,
    target: str | None,
    summaries: list[pd.DataFrame],
    counts: list[pd.DataFrame],
) -> str:
    if not picked:
        return ""
    if len(picked) == 1:
        return _single_headline(picked[0], kinds[picked[0]], summaries, counts)
    n_num = sum(k == "numeric" for k in kinds.values())
    n_cat = sum(k == "categorical" for k in kinds.values())
    contexts = [
        (compare == "train_vs_test", " across train vs test"),
        (by is not None, f" split by {by}"),
        (by_label and target is not None, f" split by {target}"),
        (True, ""),
    ]
    context = next(msg for ok, msg in contexts if ok)
    return f"{len(picked)} columns described ({n_num} numeric, {n_cat} categorical){context}"
