"""Preprocessing advisor: per-column recommendations, each a ready workspace step."""

from typing import Literal

import pandas as pd
from pydantic import Field

from dtk_engine.ops.advisor import advise
from dtk_engine.params import SourceParams, column_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import SourceSpec, load

ModelFamily = Literal["tree", "linear", "distance", "neural"]


class Params(SourceParams):
    test: SourceSpec | None = Field(
        default=None,
        description="Optional test source: columns only in train (leak), unseen "
        "categories, test-side sentinels and types",
    )
    model_family: ModelFamily | None = Field(
        default=None,
        description="tree | linear | distance | neural: decides scaling, skew and "
        "high-cardinality encoding (empty = advice for every family)",
    )
    target: str | None = column_field(
        None,
        "Target column of `source`: excluded from the features, checked "
        "for missing values, used to detect target-derived columns",
    )


@key(
    id="preprocessing_advisor",
    title="Preprocessing advisor",
    category="analysis",
    description="Per-column preprocessing plan: drops and leak warnings (ids, "
    "train-only or target-derived columns), sentinels, types, imputation "
    "(+ indicator), log1p for skew, scaler per model family, one-hot cost / "
    "min_frequency / ordinal hints. Each row is a transform step "
    "(op, target, params) ready to apply.",
)
def run(params: Params) -> Result:
    test = load(params.test) if params.test is not None else None
    return advisor_result(load(params.source), test, params.model_family, params.target)


def advisor_result(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    model_family: str | None = None,
    target: str | None = None,
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.advise``)."""
    recs, columns = advise(df, test, model_family, target)
    warnings = recs[recs["severity"] == "warning"]
    metrics: dict = {
        "model_family": model_family or "any",
        "n_columns": len(columns),
        "n_recommendations": len(recs),
        "n_warnings": len(warnings),
        "n_leak_warnings": int((recs["category"] == "leak").sum()),
        "n_columns_dropped": int((columns["action"] == "drop").sum()),
        "onehot_columns_produced": int(columns["onehot_columns"].sum()),
    }
    lines = [f"- {r.column}: {r.advice}" for r in warnings.itertuples()]
    text = "Warnings:\n" + "\n".join(lines) if lines else ""
    result = Result(metrics=metrics, text=text)
    result.add_table("recommendations", recs, kind="steps")
    result.add_table("columns", columns)
    return result
