"""Reader registry: @reader(kind) and load(spec) dispatch."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from dtk_engine.cache import LRU, digest, stamps
from dtk_engine.errors import SourceError

# Raw-source cache; see dtk_engine/cache.py for the invalidation reasoning.
_RAW_CACHE = LRU(max_entries=32, max_bytes=500 * 1024 * 1024)

_READERS: dict[str, Callable[[BaseModel], pd.DataFrame]] = {}


def reader(kind: str):
    """Register ``read(spec) -> pd.DataFrame`` for specs of this ``kind``."""

    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        if kind in _READERS:
            raise ValueError(f"duplicate reader kind {kind!r}")
        _READERS[kind] = fn
        return fn

    return decorator


def load(spec: BaseModel) -> pd.DataFrame:
    """Load a SourceSpec into a DataFrame; raise SourceError on failure."""
    kind = getattr(spec, "kind", None)
    try:
        read = _READERS[kind]
    except KeyError:
        raise SourceError(f"no reader for source kind {kind!r}") from None
    stamp = stamps([spec])  # None: no path (sql, dataset) or stat failed -> no caching
    key = stamp and digest(spec.model_dump_json(), stamp)
    return _RAW_CACHE.memo(key, lambda: read(spec))
