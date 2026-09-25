"""Column picking shared by the column-selector keys (``params.columns_field``).

Pure pandas, no ``Result``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.profile import semantic_type

# Semantic types a key analyses by default ("every eligible column"): ids, free
# text and dates only get in when picked explicitly.
AUTO_TYPES = ("numeric", "categorical", "boolean")


def is_numeric(series: pd.Series) -> bool:
    """Numeric and not boolean."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def value_kind(series: pd.Series) -> str:
    """``numeric`` (histogram, moments) or ``categorical`` (value counts): a
    numeric column with few values (e.g. a 0/1 flag, a class 1-3) is categorical."""
    return "numeric" if semantic_type(series) == "numeric" else "categorical"


def auto_eligible(series: pd.Series) -> bool:
    return semantic_type(series) in AUTO_TYPES


def numeric_feature(series: pd.Series) -> bool:
    """Numeric, and not an id (row / entity ids correlate with nothing useful)."""
    return is_numeric(series) and semantic_type(series) not in ("id_like", "group_id")


def pick_columns(
    df: pd.DataFrame,
    columns: list[str],
    op: str,
    *,
    exclude: Iterable[str | None] = (),
    eligible: Callable[[pd.Series], bool] = auto_eligible,
    required: Callable[[pd.Series], bool] | None = None,
    requirement: str = "",
    cap: int | None = None,
) -> tuple[list[str], int]:
    """``(columns to analyse, n left out by the cap)``.

    ``columns`` empty: every ``eligible`` column but ``exclude``, first ``cap``.
    Otherwise the pick as given (not capped): absent or excluded columns, or
    columns failing ``required``, raise a ``KeyParamsError`` naming them.
    """
    excluded = {c for c in exclude if c is not None}
    if not columns:
        picked = [str(c) for c in df.columns if c not in excluded and eligible(df[c])]
        if not picked:
            raise KeyParamsError(f"{op}: no eligible column in the frame")
        if cap is not None and len(picked) > cap:
            return picked[:cap], len(picked) - cap
        return picked, 0
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyParamsError(f"{op}: columns not in the frame {absent}")
    clash = [c for c in columns if c in excluded]
    if clash:
        raise KeyParamsError(f"{op}: {clash} cannot be picked (it is the target)")
    if required is not None:
        bad = [c for c in columns if not required(df[c])]
        if bad:
            raise KeyParamsError(f"{op}: {bad} {requirement}")
    return list(dict.fromkeys(columns)), 0
