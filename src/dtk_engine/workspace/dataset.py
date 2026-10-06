"""`dataset` source kind: the current state of a workspace dataset, cached.

X is loaded, y joined when the role has labels and ``labeled`` is true, merges
applied, then the workspace steps targeting the role are replayed.

Replayed frames are content-addressed (see ``dtk_engine.cache``): the key is
the JSON of everything the result depends on + ``os.stat`` of every file the
datasets and merges read. Stat costs microseconds where parse + replay costs
seconds. Works for unsaved workspace dicts as well as saved workspaces: nothing
here is name-addressed. Sources without a trustworthy file stamp (SQL, a nested
dataset, a directory) make the call uncacheable: it computes every time.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pandas as pd
from pydantic import ValidationError

from dtk_engine.cache import LRU, digest, stamps
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.ops.join import LabelJoinError, join_labels, merge_table
from dtk_engine.sources.registry import load, reader
from dtk_engine.sources.spec import DatasetSource
from dtk_engine.workspace.models import Step, Workspace
from dtk_engine.workspace.replay import (
    needs_train,
    replay,
    resolve_version,
    validate_steps,
)
from dtk_engine.workspace.store import JsonWorkspaceStore, WorkspaceNotFoundError

_FRAMES = LRU(max_entries=64, max_bytes=750 * 1024 * 1024)


def workspace_key(
    ws: Workspace, kind: str, role: str, steps: list[Step], labeled: bool = True
) -> str | None:
    """Content key of what ``ws`` yields for ``role`` after ``steps``; ``kind``
    tells apart the frames / facts of one workspace (raw, replayed, shape...)."""
    d = ws.datasets
    sources = [d.train.x, d.train.y, *([d.test.x, d.test.y] if d.test else [])]
    file_stamps = stamps([s for s in sources if s is not None] + [m.source for m in ws.merges])
    if file_stamps is None:
        return None
    payload = {
        "kind": kind,
        "role": role,
        "labeled": labeled,
        "label": ws.label.model_dump(mode="json"),
        "merges": [m.model_dump(mode="json") for m in ws.merges],
        "datasets": d.model_dump(mode="json"),
        "steps": [s.model_dump(mode="json", exclude={"id", "note"}) for s in steps],
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
    """``compute()`` memoized on the workspace content (see ``workspace_key``)."""
    return _FRAMES.memo(workspace_key(ws, kind, role, steps, labeled), compute)


def raw_workspace_frame(ws, role: str, labeled: bool = True) -> pd.DataFrame:
    """Raw frame for ``role`` with labels and merges applied (no steps replayed)."""
    dataset = getattr(ws.datasets, role)
    if dataset is None:
        raise SourceError(f"workspace {ws.name!r} has no {role} dataset")
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
            frame = join_labels(x, load(dataset.y), ws.label.mode, ws.label.key)
        except LabelJoinError as exc:
            raise SourceError(str(exc)) from exc
    for m in getattr(ws, "merges", []):
        if m.apply_to in ("both", role):
            frame = merge_table(frame, load(m.source), key=m.key, columns=m.columns)
    return frame


def workspace_frame(
    ws, role: str, labeled: bool = True, version: int | None = None
) -> pd.DataFrame:
    """State of ``role`` for a Workspace object at ``version`` (no store access).

    ``version`` (None = every saved step) replays only the first N steps, same
    semantics as ``workspace.inspect._frame_at``.
    """
    steps = ws.steps[: resolve_version(len(ws.steps), version)]

    def compute() -> pd.DataFrame:
        frame = raw_workspace_frame(ws, role, labeled)
        train = None
        if role == "test" and needs_train(steps):
            train = raw_workspace_frame(ws, "train", labeled)
        return replay(steps, role, frame, train)

    return cached_frame(ws, "replay", role, steps, labeled, compute)


def parse_workspace(ws: dict) -> Workspace:
    """Workspace model + every step's op and params (nothing is read).

    Invalid shape or step params -> KeyParamsError; unknown op ->
    UnknownTransformError.
    """
    try:
        parsed = Workspace.model_validate(ws)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    validate_steps(parsed.steps)
    return parsed


def preview(ws: dict, role: str, head_rows: int = 5) -> dict:
    """Replay an unsaved workspace dict in memory (nothing is written to the store).

    Validates like ``parse_workspace``; returns ``{shape: [rows, cols], columns,
    head: records}`` for ``role`` ("train" | "test") with its steps applied.
    """
    parsed = parse_workspace(ws)
    if role not in ("train", "test"):
        raise KeyParamsError(f"role must be 'train' or 'test', got {role!r}")
    df = workspace_frame(parsed, role)
    return {
        "shape": [len(df), df.shape[1]],
        "columns": [str(c) for c in df.columns],
        "head": json.loads(df.head(head_rows).to_json(orient="records")),
    }


@reader("dataset")
def read_dataset(spec: DatasetSource) -> pd.DataFrame:
    try:
        ws = JsonWorkspaceStore().get(spec.workspace)
    except WorkspaceNotFoundError:
        raise SourceError(f"workspace not found: {spec.workspace!r}") from None
    return workspace_frame(ws, spec.role, spec.labeled, version=spec.version)
