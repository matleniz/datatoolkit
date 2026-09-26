"""The engine <-> front boundary. Everything in and out is plain JSON."""

from __future__ import annotations

import json

from pydantic import TypeAdapter, ValidationError

from . import keys  # noqa: F401  (registers every key)
from .errors import KeyParamsError
from .ops import transforms  # noqa: F401  (registers every transform op)
from .ops.columns import is_numeric
from .registry import all_keys, get_key
from .sources import SourceSpec, load
from .sources.dataset import workspace_frame
from .transform_registry import all_transforms, get_transform
from .workspace import JsonWorkspaceStore, Workspace
from .workspace.export import export_workspace as _export
from .workspace.replay import validate_steps


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


def export_workspace(name: str, out_dir: str, overwrite: bool = False) -> dict:
    """Write ``out_dir/processed/{train,test}.parquet`` + ``out_dir/manifest.json``
    (sources hashed, steps with fitted states, versions); returns the manifest.

    Unknown workspace -> WorkspaceNotFoundError; an existing export without
    ``overwrite`` or an output path that is a raw input -> KeyParamsError; a
    source or a step failing on the data -> SourceError; an unknown step op ->
    UnknownTransformError, invalid step params -> KeyParamsError.
    """
    return _export(name, out_dir, overwrite=overwrite)
