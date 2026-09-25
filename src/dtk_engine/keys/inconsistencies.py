"""Inconsistencies: spelling variants, mixed types, ambiguous dates."""

import pandas as pd
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.consistency import (
    ambiguous_dates,
    mixed_types,
    text_columns,
    variants,
)
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    columns: list[str] | None = Field(
        default=None, description="Columns to check; null = all text columns"
    )


@key(
    id="inconsistencies",
    title="Inconsistencies",
    category="analysis",
    description="Text variants that merge after strip + lower-case, mixed "
    "number / string columns, ambiguous dd/mm vs mm/dd dates, and a suggested "
    "variant -> canonical mapping.",
)
def run(params: Params) -> Result:
    return inconsistencies_result(load(params.source), params.columns)


def inconsistencies_result(
    df: pd.DataFrame, columns: list[str] | None = None
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.inconsistencies``)."""
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in the frame: {missing}")
        df = df[columns]
    cols = text_columns(df)
    summary, mapping = variants(df, cols)
    mixed = mixed_types(df)
    dates = ambiguous_dates(df, cols)
    result = Result(
        metrics={
            "columns_checked": len(cols),
            "n_columns_with_variants": len(summary),
            "n_merged_variants": int(summary["n_merged"].sum()),
            "n_mixed_type_columns": len(mixed),
            "n_ambiguous_date_columns": len(dates),
        }
    )
    result.add_table("variants", summary)
    result.add_table("suggested mapping", mapping)
    result.add_table("mixed types", mixed)
    result.add_table("ambiguous dates", dates)
    if len(mapping):
        result.text = (
            "Apply the suggested mapping (variant -> canonical, the most frequent "
            "form) or normalise with strip + lower-case."
        )
    return result
