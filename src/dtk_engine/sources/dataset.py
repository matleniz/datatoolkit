"""`dataset` reader: the current state of a workspace dataset.

X is loaded, y joined when the role has labels and ``labeled`` is true, then the
workspace steps targeting the role are replayed.
"""

from __future__ import annotations

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.ops.join import LabelJoinError, join_labels, merge_table
from dtk_engine.sources.registry import load, reader
from dtk_engine.sources.spec import DatasetSource


def labeled_frame(
    dataset,
    label,
    labeled: bool,
    merges: list | None = None,
    role: str = "train",
) -> pd.DataFrame:
    x = load(dataset.x)
    if dataset.target_column is not None:
        if dataset.target_column not in x.columns:
            raise SourceError(
                f"target column {dataset.target_column!r} not in X "
                f"(columns: {list(x.columns)})"
            )
        frame = x if labeled else x.drop(columns=dataset.target_column)
    elif dataset.y is None or not labeled:
        frame = x
    else:
        try:
            frame = join_labels(x, load(dataset.y), label.mode, label.key)
        except LabelJoinError as exc:
            raise SourceError(str(exc)) from exc

    if merges:
        for m in merges:
            if m.apply_to == "both" or m.apply_to == role:
                merge_df = load(m.source)
                frame = merge_table(frame, merge_df, key=m.key, columns=m.columns)

    return frame


@reader("dataset")
def read_dataset(spec: DatasetSource, store=None) -> pd.DataFrame:
    # Imported here: dtk_engine.workspace imports sources.spec (import cycle).
    from dtk_engine.workspace.store import JsonWorkspaceStore, WorkspaceNotFoundError

    store = store if store is not None else JsonWorkspaceStore()
    try:
        ws = store.get(spec.workspace)
    except WorkspaceNotFoundError:
        raise SourceError(f"workspace not found: {spec.workspace!r}") from None
    return workspace_frame(ws, spec.role, spec.labeled, version=spec.version)


def raw_workspace_frame(ws, role: str, labeled: bool = True) -> pd.DataFrame:
    """Raw frame for ``role`` with labels and merges applied (no steps replayed)."""
    dataset = getattr(ws.datasets, role)
    if dataset is None:
        raise SourceError(f"workspace {ws.name!r} has no {role} dataset")
    merges = getattr(ws, "merges", [])
    return labeled_frame(dataset, ws.label, labeled, merges=merges, role=role)


def workspace_frame(
    ws, role: str, labeled: bool = True, version: int | None = None
) -> pd.DataFrame:
    """State of ``role`` for a Workspace object at ``version`` (no store access).

    ``version`` (None = every saved step) replays only the first N steps, same
    semantics as ``workspace.inspect._frame_at``.
    """
    from dtk_engine.workspace.replay import needs_train, replay, resolve_version
    from dtk_engine.workspace.replay_cache import cached_frame

    n = resolve_version(len(ws.steps), version)
    steps = ws.steps[:n]

    def compute() -> pd.DataFrame:
        frame = raw_workspace_frame(ws, role, labeled)
        train = None
        if role == "test" and needs_train(steps):
            train = raw_workspace_frame(ws, "train", labeled)
        return replay(steps, role, frame, train)

    return cached_frame(ws, "replay", role, steps, labeled, compute)

