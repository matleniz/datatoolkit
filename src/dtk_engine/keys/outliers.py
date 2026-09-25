"""Outliers: per-column IQR / z-score, multivariate IsolationForest, action table."""

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.outliers import (
    ACTION_TABLE,
    IQR_K,
    Z_THRESHOLD,
    flagged_rows,
    isolation_forest,
    numeric_columns,
    univariate_outliers,
)
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    iqr_k: float = Field(
        default=IQR_K, gt=0, description="IQR fence multiplier (1.5 = Tukey)"
    )
    z_threshold: float = Field(
        default=Z_THRESHOLD, gt=0, description="Flag values with |z| above this"
    )
    contamination: float = Field(
        default=0.01,
        gt=0,
        le=0.5,
        description="Expected share of anomalous rows for the IsolationForest "
        "(explicit, no 'auto')",
    )
    random_state: int = Field(default=0, description="IsolationForest seed")


@key(
    id="outliers",
    title="Outliers",
    category="analysis",
    description="Per numeric column IQR fences and |z| counts, a multivariate "
    "IsolationForest flagging anomalous rows, and what to do with them "
    "(remove / clip / keep / transform).",
)
def run(params: Params) -> Result:
    return outliers_result(
        load(params.source),
        params.iqr_k,
        params.z_threshold,
        params.contamination,
        params.random_state,
    )


def outliers_result(
    df: pd.DataFrame,
    iqr_k: float = IQR_K,
    z_threshold: float = Z_THRESHOLD,
    contamination: float = 0.01,
    random_state: int = 0,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.outliers``)."""
    columns = numeric_columns(df)
    table = univariate_outliers(df, columns, iqr_k, z_threshold)
    scores = isolation_forest(df, columns, contamination, random_state)
    flagged = flagged_rows(df, scores, columns)
    result = Result(
        metrics={
            "n_rows": len(df),
            "n_numeric_columns": len(columns),
            "n_columns_with_iqr_outliers": int((table["n_iqr"] > 0).sum()),
            "n_columns_with_z_outliers": int((table["n_z"] > 0).sum()),
            "n_rows_flagged": int(scores["flagged"].sum()) if len(scores) else 0,
            "contamination": contamination,
        },
        text=ACTION_TABLE,
    )
    result.add_table("outliers_per_column", table)
    result.add_table("flagged_rows", flagged)
    if len(table):
        result.add_figure(
            "% outliers per column (IQR)",
            px.bar(
                table, x="column", y="pct_iqr", labels={"pct_iqr": "% outside fences"}
            ),
        )
    if len(scores):
        result.add_figure(
            "IsolationForest scores",
            px.histogram(
                scores, x="score", color="flagged", labels={"score": "anomaly score"}
            ),
        )
    return result
