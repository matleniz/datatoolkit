"""Memo for the expensive, deterministic model fits of the analysis ops (MAT-210).

Studio re-runs the ``outliers`` / ``feature_selection`` keys whenever its grid
reloads or a selection toggles back, with the same frame and params: an
IsolationForest or RandomForest + L1 refit costs seconds where hashing the frame
costs milliseconds. Keyed by content (``pd.util.hash_pandas_object`` of the
columns the fit reads, plus their names / dtypes and the params), so there is
nothing to invalidate; see ``dtk_engine.sources._cache``.

Only for fits seeded by an explicit ``random_state``: a hit must equal a refit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import pandas as pd

from dtk_engine.sources._cache import FrameLRU, digest

_CACHE = FrameLRU(max_entries=64, max_bytes=128 * 1024 * 1024)


def frame_digest(df: pd.DataFrame) -> str | None:
    """Content hash of ``df`` (values, index, column names, dtypes); None when a
    cell is unhashable (lists, dicts): the caller then computes uncached."""
    try:
        rows = pd.util.hash_pandas_object(df, index=True).to_numpy()
    except TypeError:
        return None
    h = hashlib.sha256(rows.tobytes())
    h.update(repr([(str(c), str(t)) for c, t in df.dtypes.items()]).encode())
    return h.hexdigest()


def memo_frame(
    name: str,
    df: pd.DataFrame,
    params: Any,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """``compute()`` memoized on (``name``, content of ``df``, ``params``).

    Pass as ``df`` exactly the columns ``compute`` reads."""
    content = frame_digest(df)
    if content is None:
        return compute()
    key = digest(name, content, params)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    out = compute()
    _CACHE.put(key, out)
    return out
