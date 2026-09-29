"""Correlations: numeric correlation heatmap over picked columns + correlated pairs."""

from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
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
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import CsvSource, SourceSpec, load

# Columns in the matrix when none are picked (the first numeric ones).
MAX_COLUMNS = 50
# Heatmap cells carry their value up to this many columns.
ANNOTATE_MAX = 20


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
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
    )


def correlations_result(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    threshold: float = DEFAULT_THRESHOLD,
    target: str | None = None,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.correlations``)."""
    op = "correlations"
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

    headline = _headline(corr, pairs, threshold)
    result = Result(
        headline=headline,
        metrics={
            "method": method,
            "n_columns": len(picked),
            "n_columns_capped": n_capped,
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
        fig = px.imshow(
            corr,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f" if len(picked) <= ANNOTATE_MAX else False,
            aspect="auto",
        )
        result.add_figure(f"{method} correlation", fig, main=True)
    return result


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
    if len(pairs) > 0:
        return (
            f"{len(pairs)} collinear pair{'s' if len(pairs) > 1 else ''} "
            f"(|corr| >= {threshold}); strongest: {a} / {b} ({val:.2f})"
        )
    return f"Strongest pair: {a} / {b} ({val:.2f})"


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
