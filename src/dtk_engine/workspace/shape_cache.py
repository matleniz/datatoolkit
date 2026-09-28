"""Shape cache for workspace summaries: ``[rows, cols]`` content-addressed.

Sibling of ``replay_cache``: same ``digest`` / ``stamps`` keying from
``sources._cache``, but stores only the tiny shape tuple so
``list_workspace_summaries`` does not copy DataFrames on a hit (MAT-204).

No save / delete / rename invalidation hook: changing steps, datasets, merges
or a source file changes the key; stale entries become unreachable and age out
(see ``sources._cache``). Uncacheable sources (no trustworthy file stamp)
compute every time and never store.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable

from dtk_engine.sources._cache import digest, stamps
from dtk_engine.workspace.models import Workspace

_MAX_ENTRIES = 256
_CACHE: OrderedDict[str, list[int] | None] = OrderedDict()
_LOCK = threading.Lock()


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


def _key(ws: Workspace, role: str) -> str | None:
    sources = [ws.datasets.train.x, ws.datasets.train.y]
    if ws.datasets.test is not None:
        sources += [ws.datasets.test.x, ws.datasets.test.y]
    sources += [m.source for m in ws.merges]
    file_stamps = stamps(s for s in sources if s is not None)
    if file_stamps is None:
        return None
    payload = {
        "role": role,
        "n": len(ws.steps),
        "label": ws.label.model_dump(mode="json"),
        "merges": [m.model_dump(mode="json") for m in ws.merges],
        "datasets": ws.datasets.model_dump(mode="json"),
        "steps": [s.model_dump(mode="json") for s in ws.steps],
    }
    return digest(payload, file_stamps)


def cached_shape(
    ws: Workspace, role: str, compute: Callable[[], list[int] | None]
) -> list[int] | None:
    """``compute()`` memoized on workspace content for ``role``."""
    key = _key(ws, role)
    if key is None:
        return compute()
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    shape = compute()
    with _LOCK:
        _CACHE[key] = shape
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return shape
