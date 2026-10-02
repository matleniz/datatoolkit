"""The UI command table: the single place the MCP UI tools come from.

Each ``CommandSpec`` is one command Studio understands (shapes as parsed by
``datatoolkit-web/src/state/agentCommands.ts``). The server generates one tool
per entry, so a new Studio command becomes a tool by adding an entry here and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

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
_WINDOWS = [
    "compare", "corr", "dist", "missing", "outliers", "target", "drift",
    "feature_selection", "chart",
]


@dataclass(frozen=True)
class CommandSpec:
    type: str
    tool_name: str
    description: str
    input_schema: dict
    context_fill: tuple[str, ...] = ()  # fields filled from the UI context when absent


UI_COMMANDS: dict[str, CommandSpec] = {
    spec.type: spec
    for spec in (
        CommandSpec(
            type="propose_steps",
            tool_name="propose_steps",
            description=(
                "Propose workspace steps to the user in Studio. `ops` apply in order. "
                "Destructive ops (remove, drop_columns, filter_rows, drop_low_variance, "
                "drop_correlated) wait for the user's review and return `pending: \"review\"`: "
                "poll get_command_status with the returned id. workspace and base_identity "
                "default to the Studio context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ops": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "description": "Exactly one of add / replace / remove.",
                            "properties": {
                                "add": {
                                    "type": "object",
                                    "properties": {"step": _STEP},
                                    "required": ["step"],
                                },
                                "replace": {
                                    "type": "object",
                                    "properties": {
                                        "index": {"type": "integer", "minimum": 0},
                                        "step": _STEP,
                                    },
                                    "required": ["index", "step"],
                                },
                                "remove": {
                                    "type": "object",
                                    "properties": {"index": {"type": "integer", "minimum": 0}},
                                    "required": ["index"],
                                },
                            },
                        },
                    },
                    "workspace": {"type": "string"},
                    "base_identity": {"type": "string"},
                    "session": {"type": "string"},
                },
                "required": ["ops"],
            },
            context_fill=("workspace", "base_identity"),
        ),
        CommandSpec(
            type="open_window",
            tool_name="open_window",
            description=(
                "Open an analysis window in Studio. params.column for dist / outliers / "
                "target, params.by for dist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": _WINDOWS},
                    "params": {"type": "object"},
                    "session": {"type": "string"},
                },
                "required": ["tool"],
            },
        ),
        CommandSpec(
            type="select_columns",
            tool_name="select_columns",
            description="Select columns in the Studio grid.",
            input_schema={
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "session": {"type": "string"},
                },
                "required": ["columns"],
            },
        ),
        CommandSpec(
            type="set_view",
            tool_name="set_view",
            description=(
                "Switch the Studio grid to a dataset role and/or a step version "
                "(null = all steps). At least one of role / version."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["train", "test"]},
                    "version": {"type": ["integer", "null"], "minimum": 0},
                    "session": {"type": "string"},
                },
                "anyOf": [{"required": ["role"]}, {"required": ["version"]}],
            },
        ),
    )
}
