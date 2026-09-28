"""Shared bounded LRU for DataFrames, plus the content-addressing helpers.

Both cache layers (raw sources in ``sources.registry.load``, replay results in
``workspace.replay_cache``) key entries by *content*: the JSON of what was asked
plus ``os.stat`` (mtime_ns, size) of every file read.

Why there is no invalidation hook (save / delete / rename / duplicate a
workspace, ...): a stale hit is impossible by construction. Changing steps,
datasets, merges or label changes the JSON payload, hence the key; editing a
file changes its stat, hence the key. An entry for old content just becomes
unreachable and ages out of the LRU. A hook could not do anything the LRU does
not already do, so do not "fix" phantom staleness by wiring one up.

Every ``get`` / ``put`` copies the frame: the pipeline can hand back the very
object it was given (an empty step list returns its input), and DataFrames are
mutable, so a shared reference would let a caller corrupt the cache.
"""

from __future__ import annotations

import hashlib
import os
import stat as stat_mod
import threading
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import pandas as pd


class FrameLRU:
    """LRU of DataFrames bounded by entry count and total ``memory_usage(deep=True)``."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._data: OrderedDict[str, tuple[pd.DataFrame, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> pd.DataFrame | None:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            self._data.move_to_end(key)
            frame = hit[0]
        return frame.copy()

    def put(self, key: str, frame: pd.DataFrame) -> None:
        size = int(frame.memory_usage(deep=True, index=True).sum())
        if size > self.max_bytes:
            return  # would evict everything and still not fit
        stored = frame.copy()
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._data[key] = (stored, size)
            self._bytes += size
            while len(self._data) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, evicted) = self._data.popitem(last=False)
                self._bytes -= evicted

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def __len__(self) -> int:
        return len(self._data)


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
        path = getattr(spec, "path", None)
        if path is None:
            return None
        stamp = file_stamp(path)
        if stamp is None:
            return None
        out.append(stamp)
    return out


def digest(*parts: Any) -> str:
    """Stable hash of JSON-able parts (str parts are taken as-is)."""
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode())
        h.update(b"\x00")
    return h.hexdigest()
