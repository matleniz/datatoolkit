"""The engine <-> front boundary. Everything in and out is plain JSON.

Contract surface (fronts call these; inputs/outputs are plain JSON)::

    list_keys / key_schema / run_key
    list_workspaces / list_workspace_summaries / get_workspace /
        save_workspace / delete_workspace / rename_workspace / duplicate_workspace
    source_columns
    list_transforms / transform_schema
    preview_workspace
    export_workspace
    workspace_rows / column_profiles / preview_step / align_report  (Studio grid)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePath

from pydantic import TypeAdapter, ValidationError

from . import keys  # noqa: F401  (registers every key)
from .errors import KeyParamsError, SourceError, UnknownTransformError
from .ops import transforms  # noqa: F401  (registers every transform op)
from .ops.columns import is_numeric
from .registry import all_keys, get_key
from .sources import SourceSpec, load
from .sources.dataset import workspace_frame
from .transform_registry import all_transforms, get_transform
from .workspace import JsonWorkspaceStore, Workspace
from .workspace import inspect as _inspect
from .workspace.export import export_workspace as _export
from .workspace.replay import validate_steps


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


def _file_summary(spec) -> dict:
    """Train/test file summary for the workspace manager (kind + basename + path)."""
    path = getattr(spec, "path", None)
    return {
        "kind": getattr(spec, "kind", None),
        "path": path,
        "file": PurePath(path.replace("\\", "/")).name if path else None,
    }


def _target_summary(ws: Workspace) -> str | None:
    train = ws.datasets.train
    if train.target_column is not None:
        return train.target_column
    if train.y is not None:
        y_path = getattr(train.y, "path", None)
        return PurePath(y_path.replace("\\", "/")).name if y_path else None
    return None


def _role_shape(ws: Workspace, role: str) -> list[int] | None:
    """``[rows, cols]`` of ``role`` after steps, or null if unloadable / absent."""
    if getattr(ws.datasets, role) is None:
        return None
    try:
        df = workspace_frame(ws, role)
    except (SourceError, KeyParamsError, UnknownTransformError, OSError, ValueError):
        return None
    return [int(df.shape[0]), int(df.shape[1])]


def _workspace_summary(store: JsonWorkspaceStore, name: str) -> dict:
    path = store.path_of(name)
    ws = store.get(name)
    train = ws.datasets.train
    test = ws.datasets.test
    return {
        "name": name,
        "mtime": _iso_mtime(path),
        "step_count": len(ws.steps),
        "target": _target_summary(ws),
        "train": {**_file_summary(train.x), "shape": _role_shape(ws, "train")},
        "test": (
            None
            if test is None
            else {**_file_summary(test.x), "shape": _role_shape(ws, "test")}
        ),
    }


def list_keys() -> list[dict]:
    """Registered keys; ``needs_target``: the key requires a ``target`` column."""
    return [
        {
            "id": k.id,
            "title": k.title,
            "category": k.category,
            "description": k.description,
            "needs_target": k.needs_target,
        }
        for k in all_keys()
    ]


def key_schema(key_id: str) -> dict:
    return get_key(key_id).params_model.model_json_schema()


def run_key(key_id: str, params: dict) -> dict:
    """Run a key; returns its Result as a JSON dict.

    Omitted params take the key's defaults, and ``source`` / ``test`` default
    to the shipped ``demo_data`` CSVs: ``run_key(id, {})`` really analyses the
    Titanic-like demo files (meant for demos and tests), it is not a no-op.
    Invalid params -> KeyParamsError; unknown key -> UnknownKeyError.
    """
    k = get_key(key_id)
    try:
        parsed = k.params_model.model_validate(params)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    return k.run(parsed).model_dump(mode="json")


def source_columns(spec: dict) -> list[dict]:
    """Columns of a source (``{name, dtype, numeric}`` each, file order): the
    options of a column-selector param (``x-dtk-widget``, see ``params.py``).

    Invalid spec -> KeyParamsError; unreadable source -> SourceError.
    """
    try:
        parsed = TypeAdapter(SourceSpec).validate_python(spec)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    df = load(parsed)
    return [
        {"name": str(c), "dtype": str(df[c].dtype), "numeric": is_numeric(df[c])}
        for c in df.columns
    ]


def list_transforms() -> list[dict]:
    """Registered transform ops; ``needs_target``: fit reads the column named by
    the op's ``target`` param (supervised op)."""
    return [
        {
            "op": t.op,
            "title": t.title,
            "description": t.description,
            "needs_target": t.needs_target,
        }
        for t in all_transforms()
    ]


def transform_schema(op: str) -> dict:
    """JSON Schema of the op's params; unknown op -> raises UnknownTransformError."""
    return get_transform(op).params_model.model_json_schema()


# Workspaces: stored under $DTK_HOME/workspaces (default ~/.datatoolkit).


def list_workspaces() -> list[dict]:
    store = JsonWorkspaceStore()
    return [store.get(name).model_dump(mode="json") for name in store.list()]


def list_workspace_summaries() -> list[dict]:
    """Lightweight list for the workspace manager (MAT-171).

    Each entry: ``name``, ``mtime`` (UTC ISO), ``step_count``, ``target``
    (train ``target_column``, else y basename, else null), ``train`` /
    ``test`` (``{kind, path, file, shape}``; ``test`` null when absent;
    ``shape`` is ``[rows, cols]`` after steps, or null if unloadable).
    Sorted by name. Fronts must not recompute this from full workspace dicts.
    """
    store = JsonWorkspaceStore()
    return [_workspace_summary(store, name) for name in store.list()]


def get_workspace(name: str) -> dict:
    """The workspace dict; unknown name -> raises WorkspaceNotFoundError."""
    return JsonWorkspaceStore().get(name).model_dump(mode="json")


def _parse_workspace(ws: dict) -> Workspace:
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


def save_workspace(workspace: dict) -> dict:
    """Validate (shape, step ops and params) and store (create or overwrite);
    returns the normalized dict."""
    parsed = _parse_workspace(workspace)
    JsonWorkspaceStore().save(parsed)
    return parsed.model_dump(mode="json")


def preview_workspace(ws: dict, role: str, head_rows: int = 5) -> dict:
    """Replay an unsaved workspace dict in memory (nothing is written to the store).

    Validates like ``save_workspace``; returns ``{shape: [rows, cols], columns,
    head: records}`` for ``role`` ("train" | "test") with its steps applied.
    """
    parsed = _parse_workspace(ws)
    if role not in ("train", "test"):
        raise KeyParamsError(f"role must be 'train' or 'test', got {role!r}")
    df = workspace_frame(parsed, role)
    return {
        "shape": [len(df), df.shape[1]],
        "columns": [str(c) for c in df.columns],
        "head": json.loads(df.head(head_rows).to_json(orient="records")),
    }


def delete_workspace(name: str) -> None:
    JsonWorkspaceStore().delete(name)


def rename_workspace(name: str, new_name: str) -> dict:
    """Rename a stored workspace; returns the normalized dict under ``new_name``.

    Unknown ``name`` -> WorkspaceNotFoundError; invalid / taken ``new_name`` ->
    KeyParamsError. Source paths and uploaded content-addressed files stay
    shared (not moved). Serialized with save/delete under the store lock.
    """
    return JsonWorkspaceStore().rename(name, new_name).model_dump(mode="json")


def duplicate_workspace(name: str, new_name: str) -> dict:
    """Copy a workspace under ``new_name`` (steps, variables, merges, sources).

    Source file refs stay shared (content-addressed uploads are not copied).
    Unknown ``name`` -> WorkspaceNotFoundError; invalid / taken ``new_name`` ->
    KeyParamsError.
    """
    return JsonWorkspaceStore().duplicate(name, new_name).model_dump(mode="json")


def export_workspace(name: str, out_dir: str, overwrite: bool = False) -> dict:
    """Write ``out_dir/processed/{train,test}.parquet`` + ``out_dir/manifest.json``
    (sources hashed, steps with fitted states, versions); returns the manifest.

    Unknown workspace -> WorkspaceNotFoundError; an existing export without
    ``overwrite`` or an output path that is a raw input -> KeyParamsError; a
    source or a step failing on the data -> SourceError; an unknown step op ->
    UnknownTransformError, invalid step params -> KeyParamsError.
    """
    return _export(name, out_dir, overwrite=overwrite)


# Studio grid (unsaved workspace dicts; nothing written to the store).


def workspace_rows(
    ws: dict,
    role: str,
    version: int | None = None,
    offset: int = 0,
    limit: int = 500,
    columns: list[str] | None = None,
) -> dict:
    """Paged rows for ``role`` at ``version`` (None = all steps).

    Returns ``{columns: [{name, dtype, kind}], rows: [{..., _rid}], total,
    version}``. ``kind`` is number|binary|text|date|identifier|bool (from
    ``ops.profile.semantic_type``, plus an ``*_id`` / ``Id`` name heuristic for
    high-distinctness text). ``_rid`` is the row's position in the raw
    frame, preserved through row-dropping steps. NaN -> null, datetimes -> ISO.

    Optional ``columns`` (non-empty list) restricts column meta and row cells
    to those names in that order; unknown names raise ``KeyParamsError``.
    """
    return _inspect.workspace_rows(
        _parse_workspace(ws),
        role,
        version=version,
        offset=offset,
        limit=limit,
        columns=columns,
    )


def column_profiles(
    ws: dict,
    role: str,
    version: int | None = None,
    columns: list[str] | None = None,
) -> dict:
    """Per-column profile for ``role`` at ``version`` (histograms, sentinels,
    IQR bounds, variants, dates-/numbers-as-text, skew).

    Optional ``columns`` (non-empty list) profiles only those names in that
    order; unknown names raise ``KeyParamsError``. ``None`` / empty = all.
    """
    return _inspect.column_profiles(
        _parse_workspace(ws), role, version=version, columns=columns
    )


def preview_step(ws: dict, step: dict, role: str) -> dict:
    """Append ``step``, replay in memory; return shape / column / cell diffs
    plus the step's fitted ``state`` and ``fitted_on``.

    Invalid step -> KeyParamsError / UnknownTransformError (like
    ``save_workspace``); a step failing on the data -> SourceError naming it.
    """
    return _inspect.preview_step(_parse_workspace(ws), step, role)


def align_report(ws: dict) -> dict:
    """Train / test column alignment after the workspace's steps.

    Per column: train/test ``{name, kind, samples}``, status
    (match|type_mismatch|value_mismatch|missing_in_test|extra_in_test|label),
    means, ``numbers_as_text``, and ``similar`` test-only names (difflib).
    On ``value_mismatch`` (categorical / label column with test-only values):
    ``only_in_test`` (``[{value, count}, ...]``), ``pct_test_rows_unseen``,
    ``near_match_hint``, and ``near_matches`` (``[{test, train}, ...]``).
    Those four fields are ``null`` / ``[]`` on other statuses.
    """
    return _inspect.align_report(_parse_workspace(ws))
