"""Missing values: rates, per-row gaps, disguised sentinels, co-occurrence, test spikes."""

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.missing import (
    cooccurrence,
    cooccurrence_pairs,
    missing_per_row,
    missing_rates,
    sentinel_counts,
    value_spikes_vs_train,
)
from dtk_engine.params import KeyParams, column_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    test: SourceSpec | None = Field(
        default=None,
        description="Optional test source: enables the imputation-spike check",
    )
    target: str | None = column_field(
        None,
        "Target column of `source`: counts rows missing it (drop them first)",
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
    return missing_result(load(params.source), test, params.target)


def missing_result(
    df: pd.DataFrame, test: pd.DataFrame | None = None, target: str | None = None
) -> Result:
    """The key's Result on DataFrames (shared with ``dtk_engine.api.missing``)."""
    rates = missing_rates(df)
    per_row = missing_per_row(df)
    sentinels = sentinel_counts(df)
    matrix = cooccurrence(df)
    pairs = cooccurrence_pairs(matrix)
    metrics: dict = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "n_columns_with_missing": int((rates["n_missing"] > 0).sum()),
        "n_columns_drop_candidates": int(
            rates["recommendation"].str.startswith("drop").sum()
        ),
        "n_rows_with_missing": int((df.isna().any(axis=1)).sum()),
        "n_spike_bins": int(per_row["spike"].sum()),
        "n_sentinel_columns": int(sentinels["column"].nunique()),
        "n_cooccurring_pairs": len(pairs),
    }
    issues = []
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"target {target!r} is not a column of the source")
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
    if test is not None:
        spikes = value_spikes_vs_train(df, test)
        metrics["n_test_spikes"] = len(spikes)
        for s in spikes.itertuples():
            issues.append(
                f"{s.column}: value {s.value} is {s.pct_test}% of test "
                f"({s.n_test} rows) vs {s.pct_train}% of train (imputed on test?)"
            )

    result = Result(metrics=metrics, text="\n".join(issues))
    result.add_table("missing_rates", rates)
    result.add_table("missing_per_row", per_row)
    result.add_table("sentinels", sentinels)
    result.add_table("cooccurrence_pairs", pairs)
    if test is not None:
        result.add_table("test_value_spikes", spikes)
    result.add_figure(
        "% missing per column",
        px.bar(
            rates[rates["n_missing"] > 0],
            x="column",
            y="pct_missing",
            labels={"pct_missing": "% missing"},
        ),
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
