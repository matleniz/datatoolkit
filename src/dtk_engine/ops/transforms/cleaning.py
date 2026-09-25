"""Cleaning ops: drop, rename, cast, dedupe, text, dates, sentinels, filter, clip."""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from dtk_engine.transform_registry import TransformParams, transform


class DropColumnsParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Columns to drop")
    missing_ok: bool = Field(
        default=False, description="Ignore listed columns absent from the frame"
    )


@transform("drop_columns", params_model=DropColumnsParams, title="Drop columns")
def drop_columns(
    df: pd.DataFrame, params: DropColumnsParams, state: dict
) -> pd.DataFrame:
    """Remove the listed columns."""
    return df.drop(
        columns=params.columns, errors="ignore" if params.missing_ok else "raise"
    )
