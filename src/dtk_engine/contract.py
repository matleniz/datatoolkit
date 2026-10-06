"""The engine <-> front boundary. Everything in and out is plain JSON.

Contract surface (fronts call these; inputs/outputs are plain JSON)::

    list_keys / key_schema / run_key
    list_workspaces / list_workspace_summaries / get_workspace /
        save_workspace / delete_workspace / rename_workspace / duplicate_workspace
    source_columns
    list_transforms / transform_schema
    preview_workspace
    export_workspace
    describe_document / list_documents / add_document / remove_document /
        document_text / document_path  (workspace documents)
    workspace_rows / column_profiles / preview_step / align_report  (Studio grid)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path, PurePath

from pydantic import TypeAdapter, ValidationError

from . import keys  # noqa: F401  (registers every key)
from .cache import LRU
from .errors import KeyParamsError, SourceError, UnknownTransformError
from .ops import transforms  # noqa: F401  (registers every transform op)
from .ops.columns import is_numeric
from .registry import all_keys, get_key
from .sources import SourceSpec, load
from .transform_registry import get_transform, transform_catalog
from .workspace import JsonWorkspaceStore, Workspace
from .workspace import documents as _documents
from .workspace import inspect as _inspect
from .workspace.dataset import parse_workspace, preview, workspace_frame, workspace_key
from .workspace.documents import (  # noqa: F401  (the upload-dir guard, for http and agent)
    NotAnUploadError,
    detect_kind,
    resolve_upload,
    source_spec,
    upload_dir,
)
from .workspace.export import export_workspace as _export
from .workspace.models import (
    MEMORY_CHARS_MAX,
    MemoryEntry,
    Step,
    WorkspaceDocument,
    fill_document_ids,
    fill_memory_ids,
    fill_step_ids,
)
from .workspace.store import WorkspaceNotFoundError

_SHAPES = LRU(max_entries=256)


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
    """``[rows, cols]`` of ``role`` after steps, or null if unloadable / absent.

    Content-addressed (``_SHAPES``, MAT-204): first call may replay;
    later summaries with the same content hit the tiny shape tuple and skip
    ``workspace_frame`` (MAT-200).
    """
    if getattr(ws.datasets, role) is None:
        return None

    def compute() -> list[int] | None:
        try:
            df = workspace_frame(ws, role)
        except (SourceError, KeyParamsError, UnknownTransformError, OSError, ValueError):
            return None
        return [int(df.shape[0]), int(df.shape[1])]

    return _SHAPES.memo(workspace_key(ws, "shape", role, ws.steps), compute)


def _workspace_summary(store: JsonWorkspaceStore, name: str) -> dict:
    """Sidebar row: metadata + cached shape (MAT-171 / MAT-200 / MAT-204)."""
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
    return transform_catalog()


def transform_schema(op: str) -> dict:
    """JSON Schema of the op's params; unknown op -> raises UnknownTransformError."""
    return get_transform(op).params_model.model_json_schema()


# Workspaces: stored under $DTK_HOME/workspaces (default ~/.datatoolkit).


def list_workspaces() -> list[dict]:
    store = JsonWorkspaceStore()
    return [store.get(name).model_dump(mode="json") for name in store.list()]


def list_workspace_summaries() -> list[dict]:
    """Lightweight list for the workspace manager (MAT-171 / MAT-204).

    Each entry: ``name``, ``mtime`` (UTC ISO), ``step_count``, ``target``
    (train ``target_column``, else y basename, else null), ``train`` /
    ``test`` (``{kind, path, file, shape}``; ``test`` null when absent;
    ``shape`` is ``[rows, cols]`` after steps, or null if unloadable).
    Shape is content-addressed and cached: the first summaries call for a
    workspace may replay frames; later calls with unchanged content reuse the
    cached tuple without loading frames (MAT-200). Sorted by name. Fronts must
    not recompute this from full workspace dicts.
    """
    store = JsonWorkspaceStore()
    return [_workspace_summary(store, name) for name in store.list()]


def get_workspace(name: str) -> dict:
    """The workspace dict; unknown name -> raises WorkspaceNotFoundError."""
    return JsonWorkspaceStore().get(name).model_dump(mode="json")


_STEPS = TypeAdapter(list[Step])
_DOCUMENTS = TypeAdapter(list[WorkspaceDocument])
_MEMORY = TypeAdapter(list[MemoryEntry])


def _stored_field(name: str, field: str, adapter: TypeAdapter, fill) -> list[dict]:
    """One list field of the stored workspace (ids filled), without validating
    the rest or loading its sources: the cheap reads behind the agent."""
    path = JsonWorkspaceStore().path_of(name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise WorkspaceNotFoundError(name) from None
    try:
        items = fill(adapter.validate_python(raw.get(field) or []))
    except (ValidationError, ValueError) as exc:
        raise KeyParamsError(f"invalid {field} in workspace {name!r}: {exc}") from exc
    return [item.model_dump(mode="json") for item in items]


def workspace_steps(name: str) -> list[dict]:
    """The stored workspace's steps (ids filled), cheap: the agent's change digest."""
    return _stored_field(name, "steps", _STEPS, fill_step_ids)


def workspace_memory(name: str) -> list[dict]:
    """The stored workspace's agent memory ``[{id, text, kind, updated_at}]``
    (cheap read, datatoolkit-issues#179)."""
    return _stored_field(name, "memory", _MEMORY, fill_memory_ids)


def memory_summary(name: str) -> dict:
    """``{entries, chars, max_chars}``: what the agent's get_memory answers."""
    entries = workspace_memory(name)
    return {"entries": entries, "chars": sum(len(e["text"]) for e in entries), "max_chars": MEMORY_CHARS_MAX}


def save_workspace(workspace: dict) -> dict:
    """Validate (shape, step ops and params) and store (create or overwrite);
    returns the normalized dict. A document path not already in the stored
    workspace must be a file under the upload dir (KeyParamsError otherwise)."""
    parsed = parse_workspace(workspace)
    store = JsonWorkspaceStore()
    _check_new_documents(store, parsed)
    store.save(parsed)
    return parsed.model_dump(mode="json")


def _check_new_documents(store: JsonWorkspaceStore, ws: Workspace) -> None:
    if not ws.documents:
        return
    try:
        known = {d.path for d in store.get(ws.name).documents}
    except WorkspaceNotFoundError:
        known = set()
    for doc in ws.documents:
        if doc.path not in known:
            _documents.resolve_upload(doc.path)


def preview_workspace(ws: dict, role: str, head_rows: int = 5) -> dict:
    """Replay an unsaved workspace dict in memory (nothing is written to the store).

    Validates like ``save_workspace``; returns ``{shape: [rows, cols], columns,
    head: records}`` for ``role`` ("train" | "test") with its steps applied.
    """
    return preview(ws, role, head_rows)


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


def export_workspace(
    name: str, out_dir: str, overwrite: bool = False, formats: list[str] | None = None
) -> dict:
    """Write ``out_dir/processed/{train,test}.parquet`` + ``out_dir/manifest.json``
    (sources hashed, steps with fitted states, versions); returns the manifest.
    ``formats`` (default ``["parquet"]``) adds / picks ``csv``
    (``processed/*.csv``), ``ipynb`` and ``py`` (``code/pipeline.*``, a replay
    of the steps through the notebook door, notes as markdown / comments).

    Unknown workspace -> WorkspaceNotFoundError; an existing export without
    ``overwrite`` or an output path that is a raw input -> KeyParamsError; a
    source or a step failing on the data -> SourceError; an unknown step op ->
    UnknownTransformError, invalid step params -> KeyParamsError.
    """
    return _export(name, out_dir, overwrite=overwrite, formats=formats)


# Workspace documents (datatoolkit-issues#178): upload refs kept with a
# workspace, read through ``workspace.documents`` (path re-checked each read).


def describe_document(path: str, name: str | None = None) -> dict:
    """A document entry (no ``id``) for an uploaded file; NotAnUploadError
    (a KeyParamsError) when ``path`` is not a file under the upload dir."""
    return _documents.describe(path, name)


def list_documents(workspace: str) -> list[dict]:
    """The stored workspace's documents (cheap read: the rest is not validated)."""
    return _stored_field(workspace, "documents", _DOCUMENTS, fill_document_ids)


def document_summaries(workspace: str) -> list[dict]:
    """What an agent lists: ``{id, name, kind, mime, size, note}``, plus the
    ``source`` spec of a table document (no path otherwise)."""
    return [_documents.summary(d) for d in list_documents(workspace)]


def add_document(
    workspace: str, path: str, name: str | None = None, note: str | None = None
) -> dict:
    """Add an uploaded file to the stored workspace; ``{document, workspace}``."""
    entry = {**describe_document(path, name), "note": note or None}

    def change(ws: Workspace) -> Workspace:
        data = ws.model_dump(mode="json")
        data["documents"] = [*data["documents"], entry]
        return _validated(data)

    saved = JsonWorkspaceStore().update(workspace, change)
    return {"document": saved.documents[-1].model_dump(mode="json"), "workspace": saved.model_dump(mode="json")}


def remove_document(workspace: str, doc_id: str) -> dict:
    """Drop a document from the stored workspace (the upload stays); ``{workspace}``."""

    def change(ws: Workspace) -> Workspace:
        _documents.find([d.model_dump() for d in ws.documents], doc_id)
        return ws.model_copy(update={"documents": [d for d in ws.documents if d.id != doc_id]})

    return {"workspace": JsonWorkspaceStore().update(workspace, change).model_dump(mode="json")}


def _validated(data: dict) -> Workspace:
    try:
        return Workspace.model_validate(data)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc


def _document(workspace: str, doc_id: str) -> dict:
    return _documents.find(list_documents(workspace), doc_id)


def document_text(
    workspace: str, doc_id: str, offset: int = 0, max_chars: int | None = None
) -> dict:
    """``{id, name, kind, offset, text, total_chars, next_offset}`` of a text or
    PDF document (pages marked ``--- page N ---``); DocumentNotReadable (a
    KeyParamsError) for table / other, or a PDF without the ``pdf`` extra."""
    return _documents.text_slice(_document(workspace, doc_id), offset, max_chars)


def document_path(workspace: str, doc_id: str) -> tuple[Path, dict]:
    """The document's file (re-resolved under the upload dir) and its entry."""
    doc = _document(workspace, doc_id)
    return _documents.resolve_upload(doc["path"]), doc


# Studio grid (unsaved workspace dicts; nothing written to the store).


def workspace_rows(
    ws: dict,
    role: str,
    version: int | None = None,
    offset: int = 0,
    limit: int = 500,
    columns: list[str] | None = None,
    filter: dict | None = None,
    sort: list[dict] | None = None,
) -> dict:
    """Paged rows for ``role`` at ``version`` (None = all steps).

    Returns ``{columns: [{name, dtype, kind}], rows: [{..., _rid}], total,
    total_unfiltered, version}``. ``kind`` is number|binary|text|date|identifier|bool (from
    ``ops.profile.semantic_type``, plus an ``*_id`` / ``Id`` name heuristic for
    high-distinctness text). ``_rid`` is the row's position in the raw
    frame, preserved through row-dropping steps. NaN -> null, datetimes -> ISO.

    Optional ``columns`` (non-empty list) restricts column meta and row cells
    to those names in that order; unknown names raise ``KeyParamsError``.

    Optional view-only ``filter`` (exactly the ``filter_rows`` params:
    ``{conditions: [{column, op, value}], combine: "and"|"or"}``) and ``sort``
    (``[{column, desc}]``, stable, multi-key, NaN last) run on the full frame
    after the steps and before paging; they never alter the workspace. ``total``
    is the row count after the filter (what a pager uses), ``total_unfiltered``
    the count before it; ``_rid`` stays the raw-frame position. Unknown
    columns, bad ops or malformed specs raise ``KeyParamsError``.
    """
    return _inspect.workspace_rows(
        parse_workspace(ws),
        role,
        version=version,
        offset=offset,
        limit=limit,
        columns=columns,
        filter=filter,
        sort=sort,
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
        parse_workspace(ws), role, version=version, columns=columns
    )


def preview_step(ws: dict, step: dict, role: str) -> dict:
    """Append ``step``, replay in memory; return shape / column / cell diffs
    plus the step's fitted ``state`` and ``fitted_on``.

    Invalid step -> KeyParamsError / UnknownTransformError (like
    ``save_workspace``); a step failing on the data -> SourceError naming it.
    """
    return _inspect.preview_step(parse_workspace(ws), step, role)


def preview_steps(ws: dict, steps: list[dict], role: str) -> dict:
    """Dry run of a list of steps appended in memory (nothing saved, no undo
    entry): ``preview_step``'s diff for the whole list, plus ``steps``: each
    step's ``{op, state, fitted_on}``."""
    return _inspect.preview_steps(parse_workspace(ws), steps, role)


def evaluate(
    ws: dict,
    role: str,
    exprs: list[str],
    steps: list[dict] | None = None,
    where: str | None = None,
    version: int | None = None,
) -> dict:
    """Read-only statistics of formula expressions (``count, missing, mean,
    std, min, q25, median, q75, max, sum`` each) on ``role`` at ``version``, or
    after the draft ``steps``; ``where`` selects rows (non-zero, not NaN)."""
    return _inspect.evaluate(
        parse_workspace(ws), role, exprs, steps=steps, where=where, version=version
    )


def column_notes(ws: dict, role: str = "train", version: int | None = None) -> dict:
    """Column notes resolved at ``version``: ``{notes: {name: text}, keys:
    {name: origin key}}`` (``keys``: renamed columns only; write a note on
    column ``X`` under ``keys.get(X, X)`` in ``notes.columns``)."""
    return _inspect.column_notes(parse_workspace(ws), role, version)


def align_report(ws: dict) -> dict:
    """Train / test column alignment after the workspace's steps.

    Per column: train/test ``{name, kind, samples}``, status
    (match|type_mismatch|value_mismatch|missing_in_test|extra_in_test|label),
    means, ``numbers_as_text``, and ``similar`` test-only names (difflib).
    On ``value_mismatch`` (categorical / label column with test-only values):
    ``only_in_test`` (``[{value, count}, ...]``), ``pct_test_rows_unseen``,
    ``near_match_hint``, ``near_matches`` (``[{test, train}, ...]``), and
    ``blocking`` (True when ``near_matches`` is non-empty or
    ``pct_test_rows_unseen`` exceeds 50 — Studio "to decide"; otherwise
    informational for rare new categories). Those fields are ``null`` / ``[]``
    / ``False`` on other statuses.
    """
    return _inspect.align_report(parse_workspace(ws))
