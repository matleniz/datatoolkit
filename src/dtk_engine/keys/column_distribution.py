"""Column distribution: histograms / value counts of picked columns, optionally
train vs test and split by label or by any column."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field

from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
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
from dtk_engine.ops.profile import HIST_BINS
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load

# Columns analysed when none are picked (the first eligible ones).
MAX_COLUMNS = 20


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    columns: list[str] = columns_field(
        f"Columns to describe (empty = every numeric / categorical / boolean "
        f"column, first {MAX_COLUMNS})"
    )
    compare: Literal["none", "train_vs_test"] = Field(
        default="none",
        description="train_vs_test: overlay `test` on `source` (same bins)",
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
    bins: int = Field(default=HIST_BINS, ge=2, le=200, description="Histogram bins")
    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=100,
        description="Categories kept per column, rest (other)",
    )
    target_bins: int = Field(
        default=TARGET_BINS,
        ge=2,
        le=20,
        description="Quantile bins of a numeric target / by column when continuous",
    )


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
    )


def distribution_result(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    target: str | None = None,
    by_label: bool = False,
    bins: int = HIST_BINS,
    top_k: int = TOP_K,
    target_bins: int = TARGET_BINS,
    by: str | None = None,
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.distribution``).

    ``target`` is only read with ``by_label`` (then it is not a described column).
    ``by`` splits by any column (top-k + other or quantile bins), independent of
    ``target``; mutually exclusive with ``by_label``.
    """
    op = "column_distribution"
    if by is not None and by_label:
        raise KeyParamsError(f"{op}: pass by or by_label, not both")
    if by is not None and by not in df.columns:
        raise KeyParamsError(f"{op}: by {by!r} not in the frame")
    if by_label:
        if target is None:
            raise KeyParamsError(f"{op}: by_label needs a target column")
        if target not in df.columns:
            raise KeyParamsError(f"{op}: target {target!r} not in the frame")
    exclude = [c for c in (by, target if by_label else None) if c is not None]
    picked, n_capped = pick_columns(
        df, columns or [], op, exclude=exclude, cap=MAX_COLUMNS
    )
    frames = {"train": df, "test": test} if test is not None else {"all": df}
    if by is not None:
        group_col, binner = by, by_binner(df[by], top_k, target_bins)
    elif by_label:
        group_col, binner = target, label_binner(df[target], target_bins)
    else:
        group_col, binner = None, None
    data = grouped_frame(frames, picked, group_col, binner)
    kinds = {c: value_kind(df[c]) for c in picked}
    by_numeric = by is not None and is_numeric(df[by])

    hists, summaries, counts, figures = [], [], [], []
    for col in picked:
        if kinds[col] == "numeric":
            hist, summary = numeric_distribution(data, col, bins)
            hists.append(hist)
            summaries.append(summary)
            figures.append((col, "numeric", _histogram_figure(hist, col)))
        else:
            table = categorical_distribution(data, col, top_k)
            counts.append(table)
            figures.append((col, "categorical", _counts_figure(table, col)))

    groups = group_order(data[GROUP])
    absent = [c for c in picked if c not in test.columns] if test is not None else []
    result = Result(
        metrics={
            "n_columns": len(picked),
            "n_numeric": sum(k == "numeric" for k in kinds.values()),
            "n_categorical": sum(k == "categorical" for k in kinds.values()),
            "n_columns_capped": n_capped,
            "n_groups": len(groups),
            "compare": "train_vs_test" if test is not None else "none",
            "by_label": target if by_label else "none",
            "by": by if by is not None else "none",
            "n_missing_in_test": len(absent),
        },
        text=_text(n_capped, absent),
    )
    result.add_table(
        "columns", pd.DataFrame({"column": picked, "kind": [kinds[c] for c in picked]})
    )
    sizes = data[GROUP].value_counts()
    result.add_table(
        "groups",
        pd.DataFrame({GROUP: groups, "n_rows": [int(sizes[g]) for g in groups]}),
    )
    if summaries:
        result.add_table(
            "numeric_summary", pd.concat(summaries, ignore_index=True), "numeric"
        )
        result.add_table("histograms", pd.concat(hists, ignore_index=True), "numeric")
    if counts:
        result.add_table(
            "value_counts", pd.concat(counts, ignore_index=True), "categorical"
        )
    for col, group, fig in figures:
        result.add_figure(col, fig, group)
    if by_numeric:
        numeric_cols = [c for c in picked if kinds[c] == "numeric"]
        if numeric_cols:
            corr = vs_by_correlations(df, numeric_cols, by)
            result.add_table("vs_by", corr, "numeric")
            if len(corr) == 1:
                result.metrics["pearson"] = float(corr["pearson"].iloc[0])
                result.metrics["spearman"] = float(corr["spearman"].iloc[0])
            for col in numeric_cols:
                sample = sample_scatter(df, col, by)
                if sample.empty:
                    continue
                result.add_figure(
                    f"{col} vs {by}",
                    px.scatter(
                        sample,
                        x=by,
                        y=col,
                        labels={by: by, col: col},
                        opacity=0.55,
                    ),
                    "numeric",
                )
    return result


def _histogram_figure(hist: pd.DataFrame, col: str) -> go.Figure:
    fig = go.Figure(
        [
            go.Bar(
                x=part["bin_left"],
                y=part["share"],
                width=np.asarray(part["bin_right"] - part["bin_left"]),
                offset=0,
                name=str(group),
                opacity=0.55,
            )
            for group, part in hist.groupby(GROUP, sort=False)
        ]
    )
    fig.update_layout(barmode="overlay", xaxis_title=col, yaxis_title="share of rows")
    return fig


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
