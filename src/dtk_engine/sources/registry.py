"""Reader registry: @reader(kind) and load(spec) dispatch."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from dtk_engine.errors import SourceError
from dtk_engine.sources._cache import FrameLRU, digest, stamps

# Raw-source cache; see sources/_cache.py for the invalidation reasoning.
_RAW_CACHE = FrameLRU(max_entries=32, max_bytes=500 * 1024 * 1024)

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
    if stamp is None:
        return read(spec)
    key = digest(spec.model_dump_json(), stamp)
    hit = _RAW_CACHE.get(key)
    if hit is not None:
        return hit
    df = read(spec)
    _RAW_CACHE.put(key, df)
    return df
