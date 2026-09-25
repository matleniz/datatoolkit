"""Column distribution: histograms / value counts of picked columns, optionally
train vs test and split by label."""

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field

from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ops.columns import pick_columns, value_kind
from dtk_engine.ops.distribution import (
    GROUP,
    TARGET_BINS,
    TOP_K,
    categorical_distribution,
    group_order,
    grouped_frame,
    label_binner,
    numeric_distribution,
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
        description="Quantile bins of a numeric target when by_label",
    )


@key(
    id="column_distribution",
    title="Column distribution",
    category="analysis",
    description="Per picked column: histogram on shared bins + summary (numeric) or "
    "top-k value counts (categorical), side by side per group: train vs test "
    "and / or per label class.",
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
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.distribution``).

    ``target`` is only read with ``by_label`` (then it is not a described column).
    """
    op = "column_distribution"
    if by_label:
        if target is None:
            raise ValueError(f"{op}: by_label needs a target column")
        if target not in df.columns:
            raise ValueError(f"{op}: target {target!r} not in the frame")
    picked, n_capped = pick_columns(
        df, columns or [], op, exclude=[target] if by_label else [], cap=MAX_COLUMNS
    )
    frames = {"train": df, "test": test} if test is not None else {"all": df}
    binner = label_binner(df[target], target_bins) if by_label else None
    data = grouped_frame(frames, picked, target, binner)
    kinds = {c: value_kind(df[c]) for c in picked}

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
