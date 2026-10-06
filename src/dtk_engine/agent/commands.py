"""The UI command table: the single place the MCP UI tools come from.

Each ``CommandSpec`` is one command Studio understands, shapes exactly as
``parseCommand`` in ``datatoolkit-web/src/state/agentCommands.ts`` parses them
(enums mirror ``TOOL_IDS``, ``CHART_TYPES``, ``CHART_AGGS``, ``FILTER_OPS`` and
``VARIABLE_STATS`` there; ``tests/agent/test_web_contract.py`` checks them
against a web checkout). The server generates one tool per entry, and
``GET /api/ui/commands/schema`` publishes the table (``command_schemas``) so
Studio can test its parser against it: a new Studio command becomes a tool by
adding an entry here and nothing else.

Every command is acked ``{id, ok, error?, identity?}``; Studio applies it as one
undoable change (Undo in its toast) and highlights what it touched. Failures
Studio reports: ``bad_command: <why>``, ``busy``, ``frame_unavailable: <why>``,
``save_failed: <why>``, ``stale`` (``stale: <why>`` plus ``stale: [{id, reason}]``
for an id-based ``propose_steps``), ``rejected``; the bridge adds ``no_studio``
and ``timeout``. ``propose_steps`` by step id also carries ``base_steps``
(``{id: {op, target, params}}`` the agent last saw, filled by the tool layer):
Studio applies it while every targeted id exists unchanged (datatoolkit-issues#153).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_STR = {"type": "string"}
_NAME = {"type": "string", "minLength": 1}
_RID = {"type": "integer", "minimum": 0, "description": "Row id (the grid's rid)."}
_TARGET = {"type": "string", "enum": ["train", "test", "both"]}
_STEP = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "description": "Transform op id (see list_transforms)."},
        "target": {**_TARGET, "description": "Which dataset(s) the step applies to."},
        "params": {
            "type": "object",
            "description": "Op params; the schema comes from transform_schema {op}.",
        },
    },
    "required": ["op"],
}
_STEP_ID = {"type": "string", "description": "Step id (stable across edits)."}
_INDEX = {"type": "integer", "minimum": 0, "description": "Legacy: step position."}
# Studio's dock windows (web ``TOOL_IDS``).
WINDOWS = [
    "compare", "corr", "dist", "missing", "outliers", "target", "drift",
    "feature_selection", "chart", "dataset_overview", "duplicates",
    "inconsistencies", "preprocessing_advisor",
]
# Web ``VARIABLE_STATS`` (engine ``formula`` op ``_STATS``).
VARIABLE_STATS = ["mean", "median", "std", "min", "max", "q25", "q75", "count"]
# Web ``CHART_TYPES`` / ``CHART_AGGS`` (the ``chart`` key's params).
CHART_TYPES = [
    "histogram", "box", "violin", "bar", "count", "scatter", "line", "heatmap",
    "density_heatmap", "pie", "scatter_matrix",
]
CHART_AGGS = ["count", "mean", "sum", "median"]
# Web ``FILTER_OPS`` (the ``filter_rows`` op's condition ops).
FILTER_OPS = ["eq", "ne", "gt", "ge", "lt", "le", "isin", "notin", "isna", "notna"]
CHART_NAME_MAX = 64
NOTE_MAX = 4000  # engine ``workspace.models.NOTE_MAX``
MEMORY_ENTRY_MAX = 500  # engine ``workspace.models.MEMORY_ENTRY_MAX``
MEMORY_KINDS = ["fact", "decision", "preference", "todo"]
_MEMORY_ID = {"type": "string", "pattern": "^m[0-9a-z-]{1,32}$", "description": "Memory entry id."}

_COLUMN_OR_NULL = {"type": ["string", "null"], "minLength": 1}
_CHART_PARAMS = {
    "type": "object",
    "description": "Chart builder params (the chart key's params without source).",
    "properties": {
        "chart": {"type": "string", "enum": CHART_TYPES},
        **{k: _COLUMN_OR_NULL for k in ("x", "y", "color", "facet_row", "facet_col", "size")},
        "columns": {"type": "array", "items": _NAME},
        "agg": {"type": ["string", "null"], "enum": [*CHART_AGGS, None]},
        "trendline": {"type": "boolean"},
        "log_x": {"type": "boolean"},
        "log_y": {"type": "boolean"},
        "bins": {"type": "integer", "minimum": 2, "maximum": 200},
        "sample_size": {"type": ["integer", "null"], "minimum": 1},
    },
    "additionalProperties": False,
}
_SCALAR = {"type": ["string", "number", "boolean"]}
_CONDITION = {
    "type": "object",
    "description": (
        "isna / notna take no value; isin / notin a non-empty list; the others a scalar."
    ),
    "properties": {
        "column": _NAME,
        "op": {"type": "string", "enum": FILTER_OPS},
        "value": {"anyOf": [_SCALAR, {"type": "array", "minItems": 1, "items": _SCALAR}]},
    },
    "required": ["column", "op"],
}
_GRID_FILTER = {
    "type": ["object", "null"],
    "description": "Same shape as the filter_rows op's params; null clears the filter.",
    "properties": {
        "conditions": {"type": "array", "minItems": 1, "items": _CONDITION},
        "combine": {"type": "string", "enum": ["and", "or"]},
    },
    "required": ["conditions", "combine"],
}
_GRID_SORT = {
    "type": ["array", "null"],
    "description": "Sort keys, first one wins; null clears the sort.",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {"column": _NAME, "desc": {"type": "boolean"}},
        "required": ["column", "desc"],
    },
}


def _args(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    """An object schema. No top-level anyOf / oneOf (model APIs refuse them in a
    tool's input_schema): "at least one of" rules are in the description and
    Studio enforces them (``bad_command``)."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    return schema


@dataclass(frozen=True)
class CommandSpec:
    type: str
    tool_name: str
    description: str
    input_schema: dict  # the command's own fields (the tool adds ``session``)
    context_fill: tuple[str, ...] = ()  # fields filled from the UI context when absent
    destructive: bool = False  # may wait for the user's review in Studio


def command_schemas() -> dict[str, dict]:
    """``{type: {input_schema, destructive}}``: the published command contract."""
    return {
        spec.type: {"input_schema": spec.input_schema, "destructive": spec.destructive}
        for spec in UI_COMMANDS.values()
    }


_PROPOSE_STEPS = CommandSpec(
    type="propose_steps",
    tool_name="propose_steps",
    description=(
        "Propose workspace steps to the user in Studio. `ops` apply in order; target "
        "steps by `id` (from get_workspace / workspace_changes): an id-based proposal "
        "applies even if the user edited other steps meanwhile, and is acked `stale` "
        "(with `stale: [{id, reason}]`) only if a step it targets was removed or "
        "changed. `add` appends; the ack's `added_ids` are the new steps' ids. "
        "Destructive ops (remove, drop_columns, filter_rows, drop_low_variance, "
        "drop_correlated) wait for the user's review and return `pending: \"review\"`: "
        "poll get_command_status with the returned id. workspace and base_identity "
        "default to the Studio context. The ack's identity is the new frame's. An ok ack "
        "means Studio has saved the steps (it awaits the PUT before acking): "
        "export_workspace / get_workspace then see them."
    ),
    input_schema=_args(
        {
            "ops": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "description": (
                        "Exactly one of add / replace / remove; replace / remove take "
                        "the step `id` (index: legacy, position-based)."
                    ),
                    "properties": {
                        "add": _args({"step": _STEP}, ("step",)),
                        "replace": _args(
                            {"id": _STEP_ID, "index": _INDEX, "step": _STEP}, ("step",)
                        ),
                        "remove": _args({"id": _STEP_ID, "index": _INDEX}),
                    },
                },
            },
            "workspace": _STR,
            "base_identity": _STR,
        },
        ("ops",),
    ),
    context_fill=("workspace", "base_identity"),
    destructive=True,
)

_VIEW_COMMANDS = (
    CommandSpec(
        type="open_window",
        tool_name="open_window",
        description=(
            "Open an analysis window in Studio. params.column for dist / outliers / "
            "target, params.by for dist; other params are the window's key params. "
            "Studio has one window per tool: opening an open one updates it in place, "
            "and params you omit keep their current value. Pass params.by \"\" to "
            "remove the dist split."
        ),
        input_schema=_args(
            {"tool": {"type": "string", "enum": WINDOWS}, "params": {"type": "object"}},
            ("tool",),
        ),
    ),
    CommandSpec(
        type="select_columns",
        tool_name="select_columns",
        description="Select columns in the Studio grid (an empty list clears the selection).",
        input_schema=_args({"columns": {"type": "array", "items": _STR}}, ("columns",)),
    ),
    CommandSpec(
        type="set_view",
        tool_name="set_view",
        description=(
            "Switch the Studio grid to a dataset role and/or a step version "
            "(null = all steps). At least one of role / version. The ack's identity "
            "is the frame now shown."
        ),
        input_schema=_args(
            {
                "role": {"type": "string", "enum": ["train", "test"]},
                "version": {"type": ["integer", "null"], "minimum": 0},
            },
        ),
    ),
    CommandSpec(
        type="set_grid_view",
        tool_name="set_grid_view",
        description=(
            "Filter and/or sort the rows the Studio grid shows (at least one of filter / sort). View only: no step, "
            "the data identity is unchanged (make it a step with propose_steps "
            "filter_rows, same filter shape). A given field replaces, null clears, an "
            "omitted one is kept. Columns must be on the frame shown. The grid's row "
            "count after it is in get_ui_context (grid.total)."
        ),
        input_schema=_args(
            {"filter": _GRID_FILTER, "sort": _GRID_SORT},
        ),
    ),
)

_SELECTION_COMMANDS = (
    CommandSpec(
        type="pick_row",
        tool_name="pick_row",
        description=(
            "Select one row of the Studio grid by rid (the inspector shows it). The rid "
            "is not checked: a row not in the frame shows an empty inspector."
        ),
        input_schema=_args({"rid": _RID}, ("rid",)),
    ),
    CommandSpec(
        type="pick_cell",
        tool_name="pick_cell",
        description="Select one cell of the Studio grid (row rid, column on the frame shown).",
        input_schema=_args({"rid": _RID, "column": _NAME}, ("rid", "column")),
    ),
    CommandSpec(
        type="clear_selection",
        tool_name="clear_selection",
        description="Clear the Studio grid selection (columns, row and cell).",
        input_schema=_args(),
    ),
)

_SETTING_COMMANDS = (
    CommandSpec(
        type="set_target",
        tool_name="set_target",
        description=(
            "Set the target column Studio's analyses use (null clears it). The column "
            "must be on the frame shown."
        ),
        input_schema=_args({"column": _COLUMN_OR_NULL}, ("column",)),
    ),
    CommandSpec(
        type="set_dist_by",
        tool_name="set_dist_by",
        description=(
            "Split the Distribution window by a column (null clears the split). The "
            "column must be on the frame shown."
        ),
        input_schema=_args({"by": _COLUMN_OR_NULL}, ("by",)),
    ),
    CommandSpec(
        type="set_tool_params",
        tool_name="set_tool_params",
        description=(
            "Tune an analysis window's params (merged over its current ones); keys and "
            "values are checked against the window key's schema (key_schema). column "
            "(dist / outliers / target only) picks the per-column params, default the "
            "focused column."
        ),
        input_schema=_args(
            {
                "tool": {"type": "string", "enum": WINDOWS},
                "params": {"type": "object", "minProperties": 1},
                "column": _NAME,
            },
            ("tool", "params"),
        ),
    ),
)

_WORKSPACE_COMMANDS = (
    CommandSpec(
        type="set_note",
        tool_name="set_note",
        description=(
            "Write a note on a step (by id), a column (its name at the latest "
            "version; the note follows later renames) or the whole workspace, to "
            "keep a finding next to what it justifies (e.g. a study's result on the "
            "column it decided). text \"\" deletes the note. One undoable change. "
            "workspace defaults to the Studio context. Read notes with get_notes. An ok ack "
            "means Studio has saved the note (it awaits the PUT before acking)."
        ),
        input_schema=_args(
            {
                "kind": {"type": "string", "enum": ["step", "column", "workspace"]},
                "step_id": {**_STEP_ID, "description": "kind step: the step id."},
                "column": {**_NAME, "description": "kind column: the column name."},
                "text": {"type": "string", "maxLength": NOTE_MAX},
                "workspace": _STR,
            },
            ("kind", "text"),
        ),
        context_fill=("workspace",),
    ),
    CommandSpec(
        type="keep_attachment",
        tool_name="keep_attachment",
        description=(
            "Keep a chat attachment in the workspace as a reference document (it then "
            "outlives the chat: list_documents / read_document in any session). Only "
            "when the user asks. One undoable change; the ack's document_id is the new "
            "document's id. workspace defaults to the Studio context. An ok ack means "
            "Studio has saved it (it awaits the PUT before acking): list_documents sees it."
        ),
        input_schema=_args(
            {
                "attachment_id": {**_NAME, "description": "Attachment id (list_attachments)."},
                "note": {"type": "string", "maxLength": NOTE_MAX},
                "workspace": _STR,
            },
            ("attachment_id",),
        ),
        context_fill=("workspace",),
    ),
    CommandSpec(
        type="remember",
        tool_name="remember",
        description=(
            "Save a short fact in the workspace memory, which you get at the start of "
            "every later chat on this workspace (data quirks, decisions, the user's "
            "preferences, todos). One fact per entry, short; pass memory_id to rewrite "
            "an entry (update rather than duplicate). One undoable change; the ack's "
            "memory_id is the entry's id. Acked `bad_command: memory full` over the "
            "caps (forget first). workspace defaults to the Studio context. Read the "
            "whole memory with get_memory. An ok ack means Studio has saved the entry "
            "(it awaits the PUT before acking). Ids are never reused in a workspace."
        ),
        input_schema=_args(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": MEMORY_ENTRY_MAX},
                "kind": {"type": "string", "enum": MEMORY_KINDS},
                "memory_id": _MEMORY_ID,
                "workspace": _STR,
            },
            ("text",),
        ),
        context_fill=("workspace",),
    ),
    CommandSpec(
        type="forget",
        tool_name="forget",
        description=(
            "Remove one entry from the workspace memory (it became wrong or done). "
            "One undoable change. workspace defaults to the Studio context. An ok ack "
            "means Studio has saved the removal (it awaits the PUT before acking)."
        ),
        input_schema=_args({"memory_id": _MEMORY_ID, "workspace": _STR}, ("memory_id",)),
        context_fill=("workspace",),
    ),
    CommandSpec(
        type="add_variable",
        tool_name="add_variable",
        description=(
            "Declare a workspace variable (`@name` in a formula) = stat(column), saved "
            "with the workspace. The name is unique; the column must be on the frame "
            "shown. A formula step proposed with propose_steps must carry the variable "
            "in its own params.variables."
        ),
        input_schema=_args(
            {
                "name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
                "stat": {"type": "string", "enum": VARIABLE_STATS},
                "column": _NAME,
            },
            ("name", "stat", "column"),
        ),
    ),
    CommandSpec(
        type="draft_chart",
        tool_name="draft_chart",
        description=(
            "Fill Studio's chart builder (opening it if needed), merged over the "
            "current draft; nothing is saved. Columns must be on the frame shown."
        ),
        input_schema=_args({"params": {**_CHART_PARAMS, "minProperties": 1}}, ("params",)),
    ),
    CommandSpec(
        type="add_chart",
        tool_name="add_chart",
        description=(
            "Save a chart in the workspace (what the builder's Save does). The name is "
            "unique; params are the builder's with defaults filled in and must be "
            "drawable (x / y where the chart type needs them)."
        ),
        input_schema=_args(
            {"name": {**_NAME, "maxLength": CHART_NAME_MAX}, "params": _CHART_PARAMS},
            ("name", "params"),
        ),
    ),
    CommandSpec(
        type="edit_step",
        tool_name="edit_step",
        description=(
            "Open an applied step in Studio's step editor (what clicking its card does); "
            "nothing changes in the pipeline. `busy` while a review waits or the editor "
            "is open on another step."
        ),
        input_schema=_args({"index": {"type": "integer", "minimum": 0}}, ("index",)),
    ),
    CommandSpec(
        type="fill_editor",
        tool_name="fill_editor",
        description=(
            "Review-first alternative to propose_steps: fill Studio's step editor and "
            "let the user preview and Apply. With no editor open, op is required and "
            "opens it on a new step; else params are merged into the open one (an "
            "edited step keeps its op). Param keys are checked against transform_schema "
            "{op}. At least one of op / params / target. Nothing is applied; `busy` while a "
            "review waits."
        ),
        input_schema=_args(
            {"op": _NAME, "params": {"type": "object"}, "target": _TARGET},
        ),
    ),
)

UI_COMMANDS: dict[str, CommandSpec] = {
    spec.type: spec
    for spec in (
        _PROPOSE_STEPS, *_VIEW_COMMANDS, *_SELECTION_COMMANDS, *_SETTING_COMMANDS,
        *_WORKSPACE_COMMANDS,
    )
}
