"""Small helpers shared by the analysis and transform ops."""

import numpy as np
import pandas as pd


def pct(count: float, total: float) -> float:
    """`count` as a percentage of `total`, rounded to 2 decimals (0.0 if empty)."""
    return round(100 * count / total, 2) if total else 0.0


def py(value):
    """numpy scalar -> plain Python value (JSON-safe)."""
    return value.item() if isinstance(value, np.generic) else value


def json_scalar(value):
    """Like `py`, and NaN -> None."""
    value = py(value)
    return None if isinstance(value, float) and np.isnan(value) else value


def require_numeric(df: pd.DataFrame, columns: list[str], op: str) -> None:
    bad = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{op} needs numeric columns, got non-numeric {bad}")
