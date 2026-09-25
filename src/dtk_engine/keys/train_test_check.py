"""Train vs test consistency: schema, missing rates, ranges, categories, leaks."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import Field

from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ops.compare import (
    auto_id_columns,
    categorical_drift,
    compare_columns,
    drift_columns,
    find_issues,
    histogram_pair,
    numeric_drift,
    overlap,
    schema_diff,
)
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load

# Overlaid train/test histograms for the most drifted numeric columns.
HISTOGRAM_TOP = 6


class Params(KeyParams):
    train: SourceSpec = CsvSource(path=TRAIN_CSV)
    test: SourceSpec = CsvSource(path=TEST_CSV)
    id_columns: list[str] | None = Field(
        default=None,
        description="Entity id columns checked for train/test overlap; "
        "null = auto (id_like / group_id columns)",
    )


@key(
    id="train_test_check",
    title="Train / test check",
    category="analysis",
    description="Schema, dtype, missing-rate, range and category mismatches between "
    "a train and a test table, plus row / id overlap (leaks).",
)
def run(params: Params) -> Result:
    return check_result(load(params.train), load(params.test), params.id_columns)


def check_result(
    train: pd.DataFrame, test: pd.DataFrame, id_columns: list[str] | None = None
) -> Result:
    """The key's Result on two DataFrames (shared with ``dtk_engine.api.check``)."""
    schema = schema_diff(train, test)
    columns = compare_columns(train, test)
    both = columns[columns["in_train"] & columns["in_test"]]
    if id_columns is None:
        id_columns, missing_ids = auto_id_columns(columns), []
    else:
        missing_ids = [c for c in id_columns if c not in set(both["column"])]
    overlap_table = overlap(train, test, id_columns)
    numeric_cols, categorical_cols = drift_columns(train, test, columns)
    num_drift = numeric_drift(train, test, numeric_cols)
    cat_drift = categorical_drift(train, test, categorical_cols)
    issues = find_issues(
        train, test, columns, overlap_table, missing_ids, num_drift, cat_drift
    )

    result = Result(
        metrics={
            "n_common": len(schema["common"]),
            "n_only_train": len(schema["only_train"]),
            "n_only_test": len(schema["only_test"]),
            "n_dtype_mismatch": int((both["dtype_train"] != both["dtype_test"]).sum()),
            "n_issues": len(issues),
            "n_errors": int((issues["severity"] == "error").sum()),
            "n_warnings": int((issues["severity"] == "warning").sum()),
            "n_drifted": int(
                issues.loc[
                    (issues["check"] == "drift") & (issues["severity"] == "warning"),
                    "column",
                ].nunique()
            ),
        }
    )
    result.add_table("issues", issues)
    result.add_table("columns", columns)
    result.add_table("overlap", overlap_table)
    result.add_table("numeric_drift", num_drift)
    result.add_table("categorical_drift", cat_drift)
    missing = both.melt(
        id_vars="column",
        value_vars=["pct_missing_train", "pct_missing_test"],
        var_name="side",
        value_name="pct_missing",
    )
    missing["side"] = missing["side"].str.removeprefix("pct_missing_")
    result.add_figure(
        "% missing train vs test",
        px.bar(
            missing,
            x="column",
            y="pct_missing",
            color="side",
            barmode="group",
            labels={"pct_missing": "% missing"},
        ),
    )
    ranked = num_drift.dropna(subset=["psi"]).sort_values("psi", ascending=False)
    for col in ranked["column"].head(HISTOGRAM_TOP):
        edges, dens_train, dens_test = histogram_pair(train[col], test[col])
        fig = go.Figure(
            [
                go.Bar(
                    x=edges[:-1],
                    y=dens,
                    width=np.diff(edges),
                    name=side,
                    offset=0,
                    opacity=0.55,
                )
                for side, dens in (("train", dens_train), ("test", dens_test))
            ]
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title=col,
            yaxis_title="share of rows",
        )
        result.add_figure(f"{col}: train vs test", fig)
    return result
