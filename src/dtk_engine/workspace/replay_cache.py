"""Replay-result cache: the frame of a role after ``steps``, content-addressed.

Key = JSON of everything the result depends on (kind, role, labeled, label,
merges, datasets, the steps replayed) + ``os.stat`` of every file the datasets
and merges read. Stat costs microseconds where parse + replay costs seconds, so
a file edit invalidates the key without paying for a load. Works for unsaved
workspace dicts as well as saved workspaces: nothing here is name-addressed.

No invalidation hooks on save / delete / rename / duplicate: see the reasoning
in ``dtk_engine.sources._cache`` (stale entries become unreachable and age out).

Sources without a trustworthy file stamp (SQL, a nested dataset, a directory)
make the whole call uncacheable: it computes every time, never wrong.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dtk_engine.sources._cache import FrameLRU, digest, stamps
from dtk_engine.workspace.models import Step, Workspace

_CACHE = FrameLRU(max_entries=64, max_bytes=750 * 1024 * 1024)


def _key(
    ws: Workspace, kind: str, role: str, steps: list[Step], labeled: bool
) -> str | None:
    sources = [ws.datasets.train.x, ws.datasets.train.y]
    if ws.datasets.test is not None:
        sources += [ws.datasets.test.x, ws.datasets.test.y]
    sources += [m.source for m in ws.merges]
    file_stamps = stamps(s for s in sources if s is not None)
    if file_stamps is None:
        return None
    payload = {
        "kind": kind,
        "role": role,
        "n": len(steps),
        "labeled": labeled,
        "label": ws.label.model_dump(mode="json"),
        "merges": [m.model_dump(mode="json") for m in ws.merges],
        "datasets": ws.datasets.model_dump(mode="json"),
        "steps": [s.model_dump(mode="json") for s in steps],
    }
    return digest(payload, file_stamps)


def cached_frame(
    ws: Workspace,
    kind: str,
    role: str,
    steps: list[Step],
    labeled: bool,
    compute: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """``compute()`` memoized on the workspace content; ``kind`` tells apart the
    different frames one workspace yields (raw / replayed, with or without ``_rid``)."""
    key = _key(ws, kind, role, steps, labeled)
    if key is None:
        return compute()
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    frame = compute()
    _CACHE.put(key, frame)
    return frame
