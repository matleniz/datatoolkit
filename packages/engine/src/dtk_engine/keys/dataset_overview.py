"""First look at one table: shape, memory, per-column profile, head."""

import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.profile import column_profile
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    head_rows: int = Field(default=5, ge=1, le=1000, description="Rows shown in `head`")


@key(
    id="dataset_overview",
    title="Dataset overview",
    category="analysis",
    description="Shape, memory, per-column dtype / semantic type / missing / uniques, head.",
)
def run(params: Params) -> Result:
    df = load(params.source)
    profile = column_profile(df)
    n_cells = df.size
    result = Result(
        metrics={
            "rows": len(df),
            "cols": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
            "pct_missing_cells": (
                round(100 * int(df.isna().sum().sum()) / n_cells, 2) if n_cells else 0.0
            ),
            "n_duplicate_rows": int(df.duplicated().sum()),
        }
    )
    result.add_table("columns", profile)
    result.add_table("head", df.head(params.head_rows))
    result.add_figure(
        "% missing per column",
        px.bar(
            profile, x="column", y="pct_missing", labels={"pct_missing": "% missing"}
        ),
    )
    return result
