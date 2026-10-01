"""Missing values: rates, per-row gaps, disguised sentinels, co-occurrence, test spikes."""

from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.missing import (
    DROP_PCT,
    REVIEW_PCT,
    cooccurrence,
    cooccurrence_pairs,
    missing_per_row,
    missing_rates,
    sentinel_counts,
    value_spikes_vs_train,
)
from dtk_engine.params import SourceParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import SourceSpec, load

SortBy = Literal["pct_missing", "n_missing", "column"]


class Params(SourceParams):
    columns: list[str] = columns_field(
        "Columns to check (empty = every column); rates, sentinels, co-occurrence "
        "and per-row counts are restricted to this pick"
    )
    test: SourceSpec | None = Field(
        default=None,
        description="Optional test source: enables the imputation-spike check",
    )
    target: str | None = column_field(
        None,
        "Target column of `source`: counts rows missing it (drop them first)",
    )
    sort: SortBy = Field(
        default="pct_missing",
        description="Sort the missing_rates table by pct_missing | n_missing | column",
    )
    threshold: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Only list columns with pct_missing >= this (0 = show all)",
    )


@key(
    id="missing_values",
    title="Missing values",
    category="analysis",
    description="Missing rate per column with drop advice, missing fields per row "
    "(spike detection), suspected sentinels (-999, 'N/A', 1900-01-01...), "
    "missingness co-occurrence, and constants imputed on the test set only.",
)
def run(params: Params) -> Result:
    test = load(params.test) if params.test is not None else None
    return missing_result(
        load(params.source),
        test,
        params.target,
        params.columns,
        params.sort,
        params.threshold,
    )


def missing_result(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    target: str | None = None,
    columns: list[str] | None = None,
    sort: SortBy = "pct_missing",
    threshold: float = 0.0,
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.missing``)."""
    work, test_work = _scoped(df, test, columns)
    rates = missing_rates(work)
    if threshold > 0:
        rates = rates[rates["pct_missing"] >= threshold].reset_index(drop=True)
    ascending = sort == "column"
    rates = rates.sort_values(sort, ascending=ascending, kind="stable").reset_index(
        drop=True
    )
    per_row = missing_per_row(work)
    sentinels = sentinel_counts(work)
    matrix = cooccurrence(work)
    pairs = cooccurrence_pairs(matrix)
    n_cells = work.size
    n_missing_cells = int(work.isna().sum().sum())
    metrics: dict = {
        "n_rows": len(work),
        "n_columns": work.shape[1],
        "n_columns_with_missing": int((work.isna().sum() > 0).sum()),
        "n_columns_drop_candidates": int(
            rates["recommendation"].str.startswith("drop").sum()
        ) if len(rates) else 0,
        "n_rows_with_missing": int((work.isna().any(axis=1)).sum()),
        "n_missing_cells": n_missing_cells,
        "pct_missing_cells": round(100 * n_missing_cells / n_cells, 2) if n_cells else 0.0,
        "n_spike_bins": int(per_row["spike"].sum()),
        "n_sentinel_columns": int(sentinels["column"].nunique()),
        "n_cooccurring_pairs": len(pairs),
        "threshold": threshold,
        "sort": sort,
    }
    issues = []
    if target is not None:
        if target not in df.columns:
            raise KeyParamsError(f"target {target!r} is not a column of the source")
        n_target = int(df[target].isna().sum())
        metrics["n_rows_missing_target"] = n_target
        if n_target:
            issues.append(
                f"{n_target} rows have no target ({target!r}): drop them first "
                "and record how many"
            )
    if metrics["n_spike_bins"]:
        ks = per_row.loc[per_row["spike"], "n_missing"].tolist()
        issues.append(
            f"spike in missing fields per row at k={ks}: a block of rows misses the "
            "same fields (failed batch or unjoined source?)"
        )
    spikes = None
    if test_work is not None:
        spikes = value_spikes_vs_train(work, test_work)
        metrics["n_test_spikes"] = len(spikes)
        for s in spikes.itertuples():
            issues.append(
                f"{s.column}: value {s.value} is {s.pct_test}% of test "
                f"({s.n_test} rows) vs {s.pct_train}% of train (imputed on test?)"
            )

    headline = _headline(metrics["n_columns_with_missing"], rates)
    result = Result(headline=headline, metrics=metrics, text="\n".join(issues))
    result.add_table("missing_rates", rates)
    result.add_table("missing_per_row", per_row)
    result.add_table("sentinels", sentinels)
    result.add_table("cooccurrence_pairs", pairs)
    if spikes is not None:
        result.add_table("test_value_spikes", spikes)
    drop_candidates = rates.loc[
        rates["recommendation"].str.startswith("drop"), "column"
    ].tolist()
    result.add_table(
        "suggested_steps",
        _suggested_steps(drop_candidates, target),
        kind="steps",
    )
    missing_cols = rates[rates["n_missing"] > 0].copy()
    if not missing_cols.empty:
        missing_cols = missing_cols.sort_values(
            "pct_missing", ascending=False, kind="stable"
        ).reset_index(drop=True)
        missing_cols["severity"] = missing_cols["pct_missing"].apply(_severity_label)
        missing_cols["pct_label"] = missing_cols["pct_missing"].apply(
            lambda p: f"{p:.0f}%" if p == int(p) else f"{p:.1f}%"
        )
        severity_order = [
            f"≥ {DROP_PCT:.0f} % (drop)",
            f"{REVIEW_PCT:.0f}–{DROP_PCT:.0f} % (indicator)",
            f"< {REVIEW_PCT:.0f} %",
        ]
        severity_colors = {
            f"≥ {DROP_PCT:.0f} % (drop)": "#ef4444",
            f"{REVIEW_PCT:.0f}–{DROP_PCT:.0f} % (indicator)": "#f59e0b",
            f"< {REVIEW_PCT:.0f} %": "#3b82f6",
        }
        fig_main = px.bar(
            missing_cols,
            x="pct_missing",
            y="column",
            color="severity",
            text="pct_label",
            orientation="h",
            color_discrete_map=severity_colors,
            category_orders={"severity": severity_order},
            labels={"pct_missing": "% missing", "column": "", "severity": ""},
        )
        fig_main.update_traces(textposition="outside", cliponaxis=False)
        fig_main.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(missing_cols["column"].tolist())),
        )
        fig_main.update_xaxes(ticksuffix="%")
        fig_main.update_layout(legend_title_text="")
    else:
        empty_df = pd.DataFrame(
            {"column": pd.Series([], dtype=str), "pct_missing": pd.Series([], dtype=float)}
        )
        fig_main = px.bar(
            empty_df,
            x="pct_missing",
            y="column",
            orientation="h",
            labels={"pct_missing": "% missing", "column": ""},
        )
        fig_main.update_layout(
            xaxis_title="% missing",
            yaxis_title="",
            annotations=[
                {
                    "text": "No missing values",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "showarrow": False,
                    "font": {"size": 14},
                }
            ],
        )

    fig_matrix = None
    if not missing_cols.empty:
        cols_with_missing = [c for c in missing_cols["column"] if c in work.columns]
        if cols_with_missing:
            sample_work = (
                work[cols_with_missing].sample(n=300, random_state=42).sort_index()
                if len(work) > 300
                else work[cols_with_missing]
            )
            matrix_mask = sample_work.isna().astype(int)
            fig_matrix = px.imshow(
                matrix_mask,
                zmin=0,
                zmax=1,
                color_continuous_scale=[
                    [0.0, "#f1f5f9"],
                    [0.5, "#f1f5f9"],
                    [0.5, "#ef4444"],
                    [1.0, "#ef4444"],
                ],
                aspect="auto",
                labels={"x": "Column", "y": "Row index", "color": ""},
            )
            fig_matrix.update_coloraxes(
                colorbar={
                    "tickvals": [0.25, 0.75],
                    "ticktext": ["Present", "Missing"],
                    "len": 0.4,
                }
            )
            fig_matrix.update_traces(
                hovertemplate="Column: %{x}<br>Row: %{y}<extra></extra>"
            )

    with plotly_lock:
        result.add_figure(
            "% missing per column",
            fig_main,
            main=True,
        )
        if fig_matrix is not None:
            result.add_figure(
                "Missingness matrix",
                fig_matrix,
            )
        result.add_figure(
            "Missing fields per row",
            px.bar(
                per_row,
                x="n_missing",
                y="n_rows",
                color="spike",
                labels={"n_missing": "missing fields in the row", "n_rows": "rows"},
            ),
        )
        if not matrix.empty:
            result.add_figure(
                "Missingness co-occurrence (Jaccard)",
                px.imshow(matrix, zmin=0, zmax=1, aspect="auto"),
            )
    return result


def _severity_label(pct: float) -> str:
    if pct >= DROP_PCT:
        return f"≥ {DROP_PCT:.0f} % (drop)"
    if pct >= REVIEW_PCT:
        return f"{REVIEW_PCT:.0f}–{DROP_PCT:.0f} % (indicator)"
    return f"< {REVIEW_PCT:.0f} %"


def _headline(n_missing_cols: int, rates: pd.DataFrame) -> str:
    if n_missing_cols == 0:
        return "No missing values"
    n_above_30 = int((rates["pct_missing"] >= 30.0).sum()) if len(rates) else 0
    worst_str = ""
    missing_rates_df = rates[rates["n_missing"] > 0]
    if not missing_rates_df.empty:
        worst = missing_rates_df.sort_values(
            "pct_missing", ascending=False, kind="stable"
        ).iloc[0]
        worst_col = worst["column"]
        worst_pct = worst["pct_missing"]
        pct_display = f"{worst_pct:.0f} %" if worst_pct >= 1 else f"{worst_pct:.1f} %"
        worst_str = f" ({worst_col} {pct_display})"

    has_have = "has" if n_missing_cols == 1 else "have"
    cols_str = f"{n_missing_cols} column{'s' if n_missing_cols > 1 else ''}"
    if n_above_30 > 0:
        return (
            f"{cols_str} {has_have} missing values; {n_above_30} above 30 %{worst_str}"
        )
    return f"{cols_str} {has_have} missing values; none above 30 %{worst_str}"


def _suggested_steps(drop_candidates: list[str], target: str | None) -> pd.DataFrame:
    fields = ["order", "op", "target", "params", "why"]
    if not drop_candidates:
        return pd.DataFrame(columns=fields)
    params: dict = {"threshold": DROP_PCT / 100}
    if target is not None:
        params["target"] = target
    return pd.DataFrame(
        [
            {
                "order": 1,
                "op": "drop_high_missing",
                "target": "both",
                "params": params,
                "why": f"drop the {len(drop_candidates)} column(s) >= {DROP_PCT:.0f}% "
                f"missing on train ({', '.join(drop_candidates)})",
            }
        ],
        columns=fields,
    )


def _scoped(
    df: pd.DataFrame, test: pd.DataFrame | None, columns: list[str] | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Restrict the analysis frame (and test) to ``columns`` when given."""
    if not columns:
        return df, test
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyParamsError(f"missing_values: columns not in the frame {absent}")
    work = df[list(dict.fromkeys(columns))]
    if test is None:
        return work, None
    shared = [c for c in work.columns if c in test.columns]
    return work, test[shared]
