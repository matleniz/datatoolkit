"""Scaling ops (standardize, normalize, fitted on train).

``scale`` stores a center and a scale per column, learned on train with the
same definitions as sklearn's scalers (``(x - center) / scale``, missing
values ignored when fitting and kept as missing).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from dtk_engine.ops._util import require_numeric as _numeric
from dtk_engine.params import columns_field
from dtk_engine.transform_registry import TransformParams, transform

# --- scale --------------------------------------------------------------------


class ScaleParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns", source="step", dtype="numeric", required=True, min_length=1
    )
    method: Literal["standard", "minmax", "robust", "maxabs"] = Field(
        default="standard",
        description="standard: mean 0, std 1; minmax: train range -> [0, 1]; "
        "robust: median 0, IQR 1 (outlier-proof); maxabs: train |max| -> 1 "
        "(keeps zeros and signs)",
    )


def _center_scale(s: pd.Series, method: str) -> tuple[float, float]:
    if method == "standard":
        center, scale = s.mean(), s.std(ddof=0)
    elif method == "minmax":
        center, scale = s.min(), s.max() - s.min()
    elif method == "robust":
        center, scale = s.median(), s.quantile(0.75) - s.quantile(0.25)
    else:  # maxabs
        center, scale = 0.0, s.abs().max()
    # A constant column is left unscaled (sklearn does the same).
    return float(center), float(scale) if scale != 0 else 1.0


def _fit_scale(df: pd.DataFrame, params: ScaleParams) -> dict:
    _numeric(df, params.columns, "scale")
    empty = [c for c in params.columns if df[c].isna().all()]
    if empty:
        raise ValueError(f"scale: columns entirely missing in the fit frame {empty}")
    return {
        col: dict(zip(("center", "scale"), _center_scale(df[col], params.method), strict=True))
        for col in params.columns
    }


@transform("scale", params_model=ScaleParams, fit=_fit_scale, title="Scale")
def scale(df: pd.DataFrame, params: ScaleParams, state: dict) -> pd.DataFrame:
    """Rescale numeric columns with train statistics (standard, minmax, robust, maxabs).

    Test values beyond the train range are not clipped (minmax can give < 0 or > 1).
    """
    _numeric(df, params.columns, "scale")
    out = df.copy()
    for col in params.columns:
        out[col] = (out[col].astype(float) - state[col]["center"]) / state[col]["scale"]
    return out


# --- log1p --------------------------------------------------------------------


class Log1pParams(TransformParams):
    columns: list[str] = columns_field(
        "Non-negative, right-skewed numeric columns",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )


@transform("log1p", params_model=Log1pParams, title="Log(1 + x)")
def log1p(df: pd.DataFrame, params: Log1pParams, state: dict) -> pd.DataFrame:
    """Replace x by log(1 + x) to tame a right skew (refuses negative values)."""
    _numeric(df, params.columns, "log1p")
    negative = {c: int((df[c] < 0).sum()) for c in params.columns}
    negative = {c: n for c, n in negative.items() if n}
    if negative:
        raise ValueError(
            f"log1p: negative values (rows per column) {negative}; "
            "shift or clip the column first"
        )
    out = df.copy()
    for col in params.columns:
        out[col] = np.log1p(out[col].astype(float))
    return out
