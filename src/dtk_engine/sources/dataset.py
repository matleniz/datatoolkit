"""`dataset` reader: the current state of a workspace dataset.

X is loaded, y joined when the role has labels and ``labeled`` is true, then the
workspace steps targeting the role are replayed.
"""

from __future__ import annotations

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.ops.join import LabelJoinError, join_labels
from dtk_engine.sources.registry import load, reader
from dtk_engine.sources.spec import DatasetSource


def labeled_frame(dataset, label, labeled: bool) -> pd.DataFrame:
    x = load(dataset.x)
    if dataset.target_column is not None:
        if dataset.target_column not in x.columns:
            raise SourceError(
                f"target column {dataset.target_column!r} not in X "
                f"(columns: {list(x.columns)})"
            )
        return x if labeled else x.drop(columns=dataset.target_column)
    if dataset.y is None or not labeled:
        return x
    try:
        return join_labels(x, load(dataset.y), label.mode, label.key)
    except LabelJoinError as exc:
        raise SourceError(str(exc)) from exc


@reader("dataset")
def read_dataset(spec: DatasetSource, store=None) -> pd.DataFrame:
    # Imported here: dtk_engine.workspace imports sources.spec (import cycle).
    from dtk_engine.workspace.replay import needs_train, replay
    from dtk_engine.workspace.store import JsonWorkspaceStore, WorkspaceNotFoundError

    store = store if store is not None else JsonWorkspaceStore()
    try:
        ws = store.get(spec.workspace)
    except WorkspaceNotFoundError:
        raise SourceError(f"workspace not found: {spec.workspace!r}") from None
    dataset = getattr(ws.datasets, spec.role)
    if dataset is None:
        raise SourceError(f"workspace {ws.name!r} has no {spec.role} dataset")
    frame = labeled_frame(dataset, ws.label, spec.labeled)
    train = None
    if spec.role == "test" and needs_train(ws.steps):
        train = labeled_frame(ws.datasets.train, ws.label, spec.labeled)
    return replay(ws.steps, spec.role, frame, train)
