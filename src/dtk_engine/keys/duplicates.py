"""Duplicate rows: exact, partial on identity columns, and conflicting values."""

import pandas as pd

from dtk_engine.ops.duplicates import (
    GROUP_COLUMN,
    conflicts,
    default_subset,
    duplicate_groups,
    exact_duplicates,
)
from dtk_engine.params import SourceParams, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import load

SAMPLE_ROWS = 20

ADVICE = (
    "Exact duplicates carry no information: drop them (drop_duplicates). "
    "For groups sharing an identity but disagreeing on values, sort by recency "
    "(timestamp / version) and keep the last row of each group; without a "
    "recency column, inspect the conflicting columns before choosing."
)


class Params(SourceParams):
    subset: list[str] | None = columns_field(
        "Identity columns for partial duplicates; "
        "null = auto (id_like / group_id columns)",
        nullable=True,
    )


@key(
    id="duplicates",
    title="Duplicates",
    category="analysis",
    description="Exact duplicate rows, partial duplicates on identity columns "
    "and conflicts (same key, different values), with cleaning advice.",
)
def run(params: Params) -> Result:
    return duplicates_result(load(params.source), params.subset)


def duplicates_result(df: pd.DataFrame, subset: list[str] | None = None) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.duplicates``)."""
    subset = default_subset(df) if subset is None else subset
    missing = [c for c in subset if c not in df.columns]
    if missing:
        raise ValueError(f"subset columns not in the frame: {missing}")
    n_rows = len(df)
    n_exact, exact_rows = exact_duplicates(df)
    result = Result(
        metrics={
            "rows": n_rows,
            "n_exact_duplicates": n_exact,
            "pct_exact_duplicates": round(100 * n_exact / n_rows, 2) if n_rows else 0.0,
            "subset": ", ".join(subset),
        }
    )
    result.add_table("exact duplicates", exact_rows.head(SAMPLE_ROWS))
    if subset:
        groups = duplicate_groups(df, subset)
        by_column, conflict_rows = conflicts(df, subset)
        n_groups = int(groups[GROUP_COLUMN].nunique())
        n_conflicts = int(conflict_rows[GROUP_COLUMN].nunique())
        result.metrics.update(
            {
                "n_partial_duplicate_rows": len(groups),
                "n_duplicate_groups": n_groups,
                "n_conflict_groups": n_conflicts,
            }
        )
        result.add_table("duplicate groups", groups.head(SAMPLE_ROWS))
        result.add_table("conflicting columns", by_column)
        result.add_table("conflicting rows", conflict_rows.head(SAMPLE_ROWS))
        result.text = ADVICE
    else:
        result.text = (
            "No identity columns detected: pass `subset` to look for partial "
            "duplicates and conflicts. " + ADVICE
        )
    return result
