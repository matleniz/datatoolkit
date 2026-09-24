"""Train vs test consistency: schema, missing rates, ranges, categories, leaks."""

import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ops.compare import (
    auto_id_columns,
    compare_columns,
    find_issues,
    overlap,
    schema_diff,
)
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


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
    train, test = load(params.train), load(params.test)
    schema = schema_diff(train, test)
    columns = compare_columns(train, test)
    both = columns[columns["in_train"] & columns["in_test"]]
    if params.id_columns is None:
        id_columns, missing_ids = auto_id_columns(columns), []
    else:
        id_columns = params.id_columns
        missing_ids = [c for c in id_columns if c not in set(both["column"])]
    overlap_table = overlap(train, test, id_columns)
    issues = find_issues(train, test, columns, overlap_table, missing_ids)

    result = Result(
        metrics={
            "n_common": len(schema["common"]),
            "n_only_train": len(schema["only_train"]),
            "n_only_test": len(schema["only_test"]),
            "n_dtype_mismatch": int((both["dtype_train"] != both["dtype_test"]).sum()),
            "n_issues": len(issues),
            "n_errors": int((issues["severity"] == "error").sum()),
            "n_warnings": int((issues["severity"] == "warning").sum()),
        }
    )
    result.add_table("issues", issues)
    result.add_table("columns", columns)
    result.add_table("overlap", overlap_table)
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
    return result
