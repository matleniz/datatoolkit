"""Impute benchmark: which imputation strategy recovers a column best?"""

from typing import Literal

import pandas as pd
from pydantic import Field

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.impute_benchmark import benchmark, best_per_column
from dtk_engine.params import SourceParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import load

Strategy = Literal["median", "mean", "group_mean", "group_prev", "group_interp"]


class Params(SourceParams):
    columns: list[str] = Field(
        default=["Age"],  # runnable demo (Titanic train)
        min_length=1,
        description="Numeric columns to benchmark (one study each)",
        json_schema_extra=columns_field("", dtype="numeric").json_schema_extra,
    )
    strategies: list[Strategy] = Field(
        default=["median", "mean"],
        min_length=1,
        description="`impute` strategies to compare; the group ones need `by` "
        "(and `order` for group_prev / group_interp)",
    )
    by: str | None = column_field(
        None, "Group strategies: entity column (e.g. patient_id)", semantic="group_id"
    )
    order: str | None = column_field(
        None, "group_prev / group_interp: order of the entity's rows (e.g. age, a date)"
    )
    mask_fraction: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Share of each column's known values hidden then re-imputed",
    )
    seed: int = Field(default=0, description="Seed of the masking (deterministic)")


@key(
    id="impute_benchmark",
    title="Impute benchmark",
    category="analysis",
    description="Hide a seeded share of a column's known values, fill them back "
    "with each imputation strategy and compare coverage, RMSE, MAE and median "
    "absolute error on the same masked rows. Read-only.",
)
def run(params: Params) -> Result:
    df = load(params.source)
    columns = list(dict.fromkeys(params.columns))
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyParamsError(f"impute_benchmark: columns not in the source {absent}")
    try:
        table = benchmark(
            df,
            columns,
            list(dict.fromkeys(params.strategies)),
            params.by,
            params.order,
            params.mask_fraction,
            params.seed,
        )
    except ValueError as exc:
        raise KeyParamsError(str(exc)) from exc
    best = best_per_column(table)
    result = Result(
        headline=_headline(columns, best),
        metrics={
            "n_columns": len(columns),
            "n_strategies": len(set(params.strategies)),
            "mask_fraction": params.mask_fraction,
            "seed": params.seed,
            **{f"best_{c}": s for c, s in zip(best["column"], best["strategy"], strict=True)},
        },
        text="Error is computed on the masked cells a strategy filled; coverage "
        "(filled / masked) is separate, so compare RMSE only at similar coverage.",
    )
    result.add_table("benchmark", table)
    return result


def _headline(columns: list[str], best: pd.DataFrame) -> str:
    if best.empty:
        return "No strategy could fill the masked values"
    parts = [
        f"{r.column}: {r.strategy} (RMSE {r.rmse:.3g}, coverage {r.coverage:.0%})"
        for r in best.itertuples()
    ]
    return "Best by RMSE: " + "; ".join(parts)
