"""Bounded, thread-safe, content-addressed memoization for the whole engine.

Every cache (raw sources, replayed workspace frames and shapes, model fits,
per-column facts) keys its entries by *content*: the JSON of what was asked,
``os.stat`` (mtime_ns, size) of every file read, or a hash of the values.

Why there is no invalidation hook (save / delete / rename / duplicate a
workspace, ...): a stale hit is impossible by construction. Changing steps,
datasets, merges or label changes the JSON payload, hence the key; editing a
file changes its stat, hence the key. An entry for old content just becomes
unreachable and ages out of the LRU. A hook could not do anything the LRU does
not already do, so do not "fix" phantom staleness by wiring one up.

Every ``get`` / ``put`` deep-copies the value: the pipeline can hand back the
very object it was given (an empty step list returns its input), and frames and
dicts are mutable, so a shared reference would let a caller corrupt the cache.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import os
import stat as stat_mod
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

_MISS = object()


class LRU:
    """LRU bounded by entry count and total DataFrame ``memory_usage(deep=True)``."""

    def __init__(self, max_entries: int, max_bytes: float = float("inf")) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._data: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return default
            self._data.move_to_end(key)
        return copy.deepcopy(hit[0])

    def put(self, key: str, value: Any) -> None:
        is_frame = isinstance(value, pd.DataFrame)
        size = int(value.memory_usage(deep=True).sum()) if is_frame else 0
        if size > self.max_bytes:
            return  # would evict everything and still not fit
        stored = copy.deepcopy(value)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._data[key] = (stored, size)
            self._bytes += size
            while len(self._data) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, evicted) = self._data.popitem(last=False)
                self._bytes -= evicted

    def memo(self, key: str | None, compute: Callable[[], Any]) -> Any:
        """``compute()`` cached under ``key``; ``None`` = uncacheable, always computes."""
        if key is None:
            return compute()
        hit = self.get(key, _MISS)
        if hit is not _MISS:
            return hit
        value = compute()
        self.put(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def __len__(self) -> int:
        return len(self._data)


class _Unhashable(Exception):
    """A frame holds cells whose hash would be ambiguous."""


def _frame_hash(data: pd.DataFrame | pd.Series) -> str:
    is_frame = isinstance(data, pd.DataFrame)
    frame = data if is_frame else data.to_frame("")
    kinds = [pd.api.types.infer_dtype(frame.iloc[:, i]) for i in range(frame.shape[1])]
    if any(k.startswith("mixed") or k == "unknown-array" for k in kinds):
        raise _Unhashable  # hash_pandas_object would hash 1 and "1" alike
    try:
        rows = pd.util.hash_pandas_object(frame, index=is_frame).to_numpy()
    except TypeError:
        raise _Unhashable from None
    meta = [(str(c), str(t)) for c, t in frame.dtypes.items()]
    return hashlib.sha256(rows.tobytes()).hexdigest() + repr((meta, kinds))


def _canon(part: Any) -> Any:
    if isinstance(part, pd.DataFrame | pd.Series):
        return _frame_hash(part)
    if isinstance(part, list | tuple):
        return [_canon(p) for p in part]
    if isinstance(part, dict):
        return {k: _canon(v) for k, v in part.items()}
    return part


def digest(*parts: Any) -> str | None:
    """Stable hash of ``parts``: frames and series by content (values and dtypes,
    plus the column names and index of a frame; a series is a bare column, its
    index and name do not count), anything else by ``repr``. None when a frame
    holds cells that hash ambiguously (lists, dicts, mixed types): uncacheable."""
    try:
        return hashlib.sha256(repr(_canon(parts)).encode()).hexdigest()
    except _Unhashable:
        return None


def file_stamp(path: str) -> tuple[str, int, int] | None:
    """(abspath, mtime_ns, size) of a regular file; None when it cannot be trusted.

    A directory (partitioned parquet) keeps its mtime when a file inside is
    rewritten, and an unreadable path may fail later with a proper error: both
    fail open, i.e. the caller skips caching.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not stat_mod.S_ISREG(st.st_mode):
        return None
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def stamps(specs: Iterable[Any]) -> list[tuple[str, int, int]] | None:
    """File stamps of every spec; None if any spec has no path or is untrusted."""
    out = []
    for spec in specs:
        stamp = file_stamp(getattr(spec, "path", None) or "")
        if stamp is None:
            return None
        out.append(stamp)
    return out


def memoize(max_entries: int = 1024, max_bytes: float = float("inf")) -> Callable:
    """Decorator caching a pure function on its arguments (keyed by ``digest``)."""

    def decorate(fn: Callable) -> Callable:
        cache = LRU(max_entries, max_bytes)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cache.memo(digest(args, kwargs), lambda: fn(*args, **kwargs))

        wrapper.cache = cache
        return wrapper

    return decorate


_FITS = LRU(max_entries=64, max_bytes=128 * 1024 * 1024)


def memo_frame(name: str, df: pd.DataFrame, params: Any, compute: Callable) -> Any:
    """``compute()`` memoized on (``name``, content of ``df``, ``params``): for the
    seeded model fits Studio re-runs on every grid reload (MAT-210). Pass as
    ``df`` exactly the columns ``compute`` reads."""
    return _FITS.memo(digest(name, df, params), compute)
