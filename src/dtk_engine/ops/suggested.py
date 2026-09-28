"""Smart per-column defaults for analysis keys (bins, log scale, top-k).

Computed by the engine from the data — fronts should read ``suggested_params``
from ``column_profiles`` (or resolve ``bins="auto"``) rather than hard-code.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops.advisor.numeric import SKEW_THRESHOLD
from dtk_engine.ops.profile import HIST_BINS, hashable

# Cap histogram bin counts (same band as the column_distribution key param).
BINS_MIN = 2
BINS_MAX = 200
# Default top-k (mirrors ``ops.distribution.TOP_K``; kept local to avoid a cycle).
TOP_K_DEFAULT = 10
# Integer columns with at most this many distinct values get one bin per value.
DISCRETE_INT_MAX_UNIQUE = 40
# Cardinality above this: keep the default top-k; below: show every category.
TOP_K_SHOW_ALL_MAX = 20
# Heavy right skew (and non-negative) -> suggest log_x.
LOG_SKEW_MIN_N = 5


def suggest_histogram_bins(values: pd.Series | np.ndarray) -> int:
    """Number of histogram bins: discrete ints get one bin per value; else
    Freedman–Diaconis (Sturges when IQR is 0), clamped to ``[BINS_MIN, BINS_MAX]``.
    """
    arr = _float_array(values)
    n = len(arr)
    if n < 2:
        return BINS_MIN
    if _is_discrete_int(arr):
        n_unique = len(np.unique(arr))
        return int(np.clip(n_unique, BINS_MIN, BINS_MAX))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    span = float(arr.max() - arr.min())
    if iqr > 0 and span > 0:
        width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        if width > 0:
            return int(np.clip(math.ceil(span / width), BINS_MIN, BINS_MAX))
    sturges = math.ceil(math.log2(n) + 1)
    return int(np.clip(sturges, BINS_MIN, BINS_MAX))


def suggest_log_scale(values: pd.Series | np.ndarray) -> bool:
    """True when the column is non-negative and heavily right-skewed (log_x)."""
    arr = _float_array(values)
    if len(arr) < LOG_SKEW_MIN_N:
        return False
    if float(arr.min()) < 0:
        return False
    skew = float(pd.Series(arr).skew())
    return not np.isnan(skew) and skew > SKEW_THRESHOLD


def suggest_top_k(series: pd.Series, default: int = TOP_K_DEFAULT) -> int:
    """Top-k categories: show all when cardinality is small, else ``default``."""
    n_unique = int(hashable(series).nunique(dropna=True))
    if n_unique <= 0:
        return default
    if n_unique <= TOP_K_SHOW_ALL_MAX:
        return max(1, n_unique)
    return default


def suggested_params(series: pd.Series, *, kind: str | None = None) -> dict:
    """Defaults a front can pre-fill for Distribution / Target on this column.

    Always includes ``top_k``. Numeric (or ``kind == "number"``) also gets
    ``bins`` and ``log_scale``.
    """
    out: dict = {"top_k": suggest_top_k(series)}
    numeric = kind in (None, "number", "numeric") and _looks_numeric(series)
    if kind in ("number", "numeric") or (kind is None and numeric):
        if numeric:
            values = pd.to_numeric(series, errors="coerce").dropna()
            out["bins"] = suggest_histogram_bins(values)
            out["log_scale"] = suggest_log_scale(values)
        else:
            out["bins"] = HIST_BINS
            out["log_scale"] = False
    return out


def resolve_bin_edges(
    values: np.ndarray,
    bins: int | str = "auto",
    bin_edges: list[float] | None = None,
) -> np.ndarray | None:
    """Histogram edges from explicit ``bin_edges``, an int count, or ``"auto"``.

    Empty ``values`` -> None. Discrete ints under auto use half-integer edges
    so each integer is its own bin.
    """
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("bin_edges needs at least two increasing edges")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing")
        return edges
    if len(values) == 0:
        return None
    if bins == "auto":
        if _is_discrete_int(values):
            uniq = np.unique(values)
            # Half-integer fences around each integer value.
            return np.concatenate([[uniq[0] - 0.5], uniq + 0.5])
        n_bins = suggest_histogram_bins(values)
        return np.histogram_bin_edges(values, bins=n_bins)
    return np.histogram_bin_edges(values, bins=int(bins))


def _float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return np.asarray(values, dtype=float)


def _is_discrete_int(arr: np.ndarray) -> bool:
    if len(arr) == 0:
        return False
    if not np.allclose(arr, np.round(arr)):
        return False
    return len(np.unique(arr)) <= DISCRETE_INT_MAX_UNIQUE


def _looks_numeric(series: pd.Series) -> bool:
    if pdt.is_bool_dtype(series):
        return False
    return bool(pdt.is_numeric_dtype(series))
