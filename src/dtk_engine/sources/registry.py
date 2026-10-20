"""Reader registry: @reader(kind) and load(spec) dispatch."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from dtk_engine.cache import LRU, digest, stamps
from dtk_engine.errors import SourceError, duplicate_columns_message

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


def string_columns(df: pd.DataFrame, where: str = "the frame") -> pd.DataFrame:
    """``df`` with every column label a str (``header=None`` and hand-built frames
    give ints); the contract speaks column names as strings. Unchanged if already so.

    Otherwise a shallow copy (no data copied, the caller's frame keeps its labels).
    Labels that collide once converted (``1`` and ``"1"``) raise SourceError.
    """
    if all(isinstance(c, str) for c in df.columns):
        return df
    names = [str(c) for c in df.columns]
    dups = [n for n in names if names.count(n) > 1]
    if dups:
        raise SourceError(
            "column labels collide once converted to strings: "
            + duplicate_columns_message(dups, where)
        )
    out = df.copy(deep=False)
    out.columns = names
    return out


def load(spec: BaseModel) -> pd.DataFrame:
    """Load a SourceSpec into a DataFrame; raise SourceError on failure."""
    kind = getattr(spec, "kind", None)
    try:
        read = _READERS[kind]
    except KeyError:
        raise SourceError(f"no reader for source kind {kind!r}") from None
    stamp = stamps([spec])  # None: no path (sql, dataset) or stat failed -> no caching
    key = stamp and digest(spec.model_dump_json(), stamp)
    return _RAW_CACHE.memo(key, lambda: string_columns(read(spec), getattr(spec, "path", "the source")))
