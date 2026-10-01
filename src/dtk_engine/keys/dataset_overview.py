"""First look at one table: shape, memory, per-column profile, head."""

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.ops.profile import (
    category_summary,
    category_values,
    column_profile,
    columns_of_type,
    datetime_stats,
    hashable_frame,
    id_stats,
    numeric_histograms,
    numeric_stats,
    text_stats,
)
from dtk_engine.params import SourceParams
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import load


class Params(SourceParams):
    head_rows: int = Field(default=5, ge=1, le=1000, description="Rows shown in `head`")


@key(
    id="dataset_overview",
    title="Dataset overview",
    category="analysis",
    description=(
        "Shape, memory, per-column profile, head, then per-semantic-type tabs: "
        "numeric stats, category values, datetime, text, ids."
    ),
)
def run(params: Params) -> Result:
    return overview_result(load(params.source), params.head_rows)


def overview_result(df: pd.DataFrame, head_rows: int = 5) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.overview``)."""
    profile = column_profile(df)
    n_cells = df.size
    pct_missing = (
        round(100 * int(df.isna().sum().sum()) / n_cells, 2) if n_cells else 0.0
    )
    n_dups = int(hashable_frame(df).duplicated().sum())
    headline = (
        f"{len(df)} rows, {df.shape[1]} columns; "
        f"{pct_missing:.1f} % missing cells, {n_dups} duplicate row{'s' if n_dups != 1 else ''}"
    )
    result = Result(
        headline=headline,
        metrics={
            "rows": len(df),
            "cols": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "pct_missing_cells": pct_missing,
            "n_duplicate_rows": n_dups,
        },
    )
    result.add_table("columns", profile, group="Overview")
    result.add_table("head", df.head(head_rows), group="Overview")
    with plotly_lock:
        result.add_figure(
            "% missing per column",
            px.bar(
                profile, x="column", y="pct_missing", labels={"pct_missing": "% missing"}
            ),
            group="Overview",
            main=True,
        )
    _add_type_groups(
        result, df, dict(zip(profile["column"], profile["semantic_type"], strict=True))
    )
    return result


@plotly_lock
def _histogram_grid(hist: pd.DataFrame):
    hist = hist.assign(bin_mid=(hist["bin_left"] + hist["bin_right"]) / 2)
    fig = px.bar(
        hist,
        x="bin_mid",
        y="count",
        facet_col="column",
        facet_col_wrap=3,
        facet_row_spacing=0.08,
        height=260 * -(-hist["column"].nunique() // 3) + 60,
    )
    fig.update_xaxes(matches=None, showticklabels=True, title=None)
    fig.update_yaxes(matches=None, title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def _add_type_groups(
    result: Result, df: pd.DataFrame, semantic: dict[str, str]
) -> None:
    """One tab per semantic type that has columns (numeric, categorical, ...)."""

    numeric = numeric_stats(df, semantic)
    if not numeric.empty:
        result.add_table("numeric stats", numeric, group="Numeric")
        hists = numeric_histograms(df, semantic=semantic)
        with plotly_lock:
            result.add_figure(
                "distributions",
                _histogram_grid(hists),
                group="Numeric",
            )

    cat_columns = columns_of_type(df, "categorical", "boolean", semantic=semantic)
    if cat_columns:
        result.add_table(
            "category summary", category_summary(df, semantic), group="Categorical"
        )
        result.add_table(
            "category values", category_values(df, cat_columns), group="Categorical"
        )

    datetimes = datetime_stats(df, semantic)
    if not datetimes.empty:
        result.add_table("datetime stats", datetimes, group="Datetime")

    texts = text_stats(df, semantic)
    if not texts.empty:
        result.add_table("text stats", texts, group="Text")

    ids = id_stats(df, semantic)
    if not ids.empty:
        result.add_table("id stats", ids, group="IDs")
