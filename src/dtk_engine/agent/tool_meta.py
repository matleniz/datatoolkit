"""Display title and MCP behaviour hints of every tool: the single source of truth.

``build_tools`` looks each tool up here by name (a tool without an entry fails
at build time); ``server._tool`` turns the entry into the SDK's ``Tool.title``
and ``ToolAnnotations``. All hints are advisory and per tool, not per argument.
Every tool is engine-local (no network): ``open_world_hint`` is always False.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolMeta:
    title: str
    read_only: bool = False
    destructive: bool = False  # may overwrite or delete (meaningful when not read-only)
    idempotent: bool = False  # a repeat with the same arguments has no further effect


def _ro(title: str) -> ToolMeta:
    return ToolMeta(title, read_only=True, idempotent=True)


def _ui(title: str) -> ToolMeta:
    """Changes only what Studio shows (view state), nothing saved."""
    return ToolMeta(title, idempotent=True)


TOOL_META: dict[str, ToolMeta] = {
    # Read-only
    "list_keys": _ro("List analysis keys"),
    "key_schema": _ro("Get a key's parameter schema"),
    "run_key": _ro("Run an analysis key"),
    "list_transforms": _ro("List transform ops"),
    "transform_schema": _ro("Get a transform's parameter schema"),
    "list_workspaces": _ro("List workspaces"),
    "get_workspace": _ro("Get a workspace"),
    "get_rows": _ro("Get dataset rows"),
    "get_profiles": _ro("Get column profiles"),
    "preview_step": _ro("Preview a step (dry run)"),
    "preview_steps": _ro("Preview several steps (dry run)"),
    "evaluate": _ro("Evaluate formula statistics"),
    "get_notes": _ro("Get notes"),
    "get_memory": _ro("Get workspace memory"),
    "align_report": _ro("Train/test alignment report"),
    "source_columns": _ro("Get a source's columns"),
    "get_ui_context": _ro("Get what the user sees in Studio"),
    "list_attachments": _ro("List chat attachments"),
    "read_attachment": _ro("Read a chat attachment"),
    "list_documents": _ro("List workspace documents"),
    "read_document": _ro("Read a workspace document"),
    "get_command_status": _ro("Get a Studio command's status"),
    # Writes files outside the workspace: overwrite replaces a previous export
    "export_workspace": ToolMeta("Export a workspace", destructive=True, idempotent=True),
    # Writes to the workspace through Studio (one undoable change each)
    "propose_steps": ToolMeta("Propose workspace steps", destructive=True),  # replace / remove
    "set_note": ToolMeta("Write a note", destructive=True, idempotent=True),  # replaces / deletes
    "keep_attachment": ToolMeta("Keep an attachment as a document"),
    "remember": ToolMeta("Save a memory entry", destructive=True),  # memory_id rewrites
    "forget": ToolMeta("Forget a memory entry", destructive=True, idempotent=True),
    "add_variable": ToolMeta("Declare a workspace variable", idempotent=True),
    "add_chart": ToolMeta("Save a chart", idempotent=True),
    # Studio view state only
    "open_window": _ui("Open an analysis window"),
    "select_columns": _ui("Select grid columns"),
    "set_view": _ui("Switch grid dataset or version"),
    "set_grid_view": _ui("Filter or sort the grid"),
    "pick_row": _ui("Select a grid row"),
    "pick_cell": _ui("Select a grid cell"),
    "clear_selection": _ui("Clear the grid selection"),
    "set_target": _ui("Set the target column"),
    "set_dist_by": _ui("Set the distribution split"),
    "set_tool_params": _ui("Set an analysis window's parameters"),
    "draft_chart": _ui("Fill the chart builder"),
    "edit_step": _ui("Open a step in the editor"),
    "fill_editor": _ui("Fill the step editor"),
}
