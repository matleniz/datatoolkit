"""Within-entity fills: a value computed from the rows of the same group.

The building blocks of the formula functions ``group_mean`` / ``group_prev`` /
``group_interp`` and of ``impute``'s group strategies. Each takes the values,
the group codes ``by`` (``group_codes``) and, when ordered, the order key, as
arrays of one frame, and returns one value per row computed only from that
row's group in that frame: stateless, so a test frame is filled from its own
rows.

A row with a missing group (or a missing order key) gets NaN and is never used
as a source. Vectorised (bincount / groupby ffill), no Python loop over groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_codes(series: pd.Series) -> np.ndarray:
    """Group ids as float codes (0..k-1), NaN where the group is missing."""
    codes = pd.factorize(series)[0].astype(float)
    codes[codes < 0] = np.nan
    return codes


def group_mean(x: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Mean of the group's observed ``x`` on every row of the group."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    has_group = ~np.isnan(by)
    if not has_group.any():
        return out
    k = int(np.nanmax(by)) + 1
    src = has_group & ~np.isnan(x)
    idx = by[src].astype(int)
    sums = np.bincount(idx, weights=x[src], minlength=k)
    counts = np.bincount(idx, minlength=k)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    out[has_group] = means[by[has_group].astype(int)]
    return out


def _sorted_positions(by: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Positions of the usable rows (group and order known), by (group, order);
    ties keep the frame order."""
    usable = np.flatnonzero(~np.isnan(by) & ~np.isnan(order))
    return usable[np.lexsort((order[usable], by[usable]))]


def group_prev(values: pd.Series, by: np.ndarray, order: np.ndarray) -> pd.Series:
    """Last observed value at or before each row, in ``order`` within its group.

    Any dtype (a category is carried forward too). Never backward: rows before
    the group's first observed value stay missing.
    """
    order = np.asarray(order, dtype=float)
    pos = _sorted_positions(by, order)
    filled = values.iloc[pos].groupby(by[pos]).ffill()
    out = values.where(pd.Series(False, index=values.index))  # all missing
    out.iloc[pos] = filled.to_numpy()
    return out


def group_interp(x: np.ndarray, by: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Linear interpolation in ``order`` between the group's nearest observed
    neighbours; observed rows keep their value, no extrapolation (NaN before the
    first / after the last observed row of the group)."""
    x = np.asarray(x, dtype=float)
    order = np.asarray(order, dtype=float)
    out = np.full(len(x), np.nan)
    pos = _sorted_positions(by, order)
    xs, os_, gs = x[pos], order[pos], by[pos]
    seen = np.where(np.isnan(xs), np.nan, np.arange(len(pos), dtype=float))
    by_group = pd.Series(seen).groupby(gs)
    lo, hi = by_group.ffill().to_numpy(), by_group.bfill().to_numpy()
    ok = ~np.isnan(lo) & ~np.isnan(hi)
    lo, hi = lo[ok].astype(int), hi[ok].astype(int)
    span = os_[hi] - os_[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(span > 0, (os_[ok] - os_[lo]) / span, 0.5)
    out[pos[ok]] = xs[lo] + frac * (xs[hi] - xs[lo])
    return out
