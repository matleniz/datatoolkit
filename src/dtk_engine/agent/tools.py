"""Agent tools: the static contract tools plus one tool per UI command.

``build_tools(port)`` is transport-agnostic; ``server.py`` turns each
``ToolSpec`` into an MCP tool. Every handler runs ``policy.check_args`` on its
arguments first and frames its output with ``policy.frame``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import anyio.to_thread

from dtk_engine import contract
from dtk_engine.agent import policy
from dtk_engine.agent.commands import UI_COMMANDS, CommandSpec
from dtk_engine.agent.ports import UiPort
from dtk_engine.errors import KeyParamsError

COMMAND_TIMEOUT = 30.0
_NO_CONTEXT = "no Studio context: pass workspace"
_NO_CONTEXT_NOTE = "no Studio context: key defaults (demo data) used"
# UI-command fields filled from the Studio context: field -> context key.
_CONTEXT_KEYS = {"workspace": "workspace", "base_identity": "identity"}

Handler = Callable[[dict], Awaitable[dict]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Handler
    read_only: bool


def _schema(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    return schema


_STR = {"type": "string"}
_ROLE = {"type": "string", "enum": ["train", "test"]}
_SESSION = {"type": "string", "description": "Studio session (default: most recent)."}
_VIEW = {
    "workspace": {**_STR, "description": "Default: the workspace open in Studio."},
    "role": {**_ROLE, "description": "Default: the role shown in Studio, else train."},
    "version": {
        "type": "integer",
        "minimum": 0,
        "description": "Steps replayed (default: the version shown in Studio, else all).",
    },
    "session": _SESSION,
}
_COLUMNS = {"type": "array", "items": _STR, "description": "Restrict to these columns."}


def _need(args: dict, name: str) -> Any:
    if args.get(name) is None:
        raise KeyParamsError(f"missing argument: {name}")
    return args[name]


def _with_note(framed: dict, note: str) -> dict:
    return {**framed, "note": "; ".join(n for n in (framed.get("note"), note) if n)}


async def _compute(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Contract calls can take seconds (keys): keep them off the event loop."""
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


@dataclass(frozen=True)
class _View:
    workspace: str
    role: str
    version: int | None
    identity: str | None


class _Tools:
    def __init__(self, port: UiPort) -> None:
        self.port = port

    # -- context ---------------------------------------------------------
    async def _view(self, args: dict, *, use_version: bool = True) -> _View:
        """Workspace / role / version from the args, else the Studio context."""
        ctx = await self.port.get_context(args.get("session")) or {}
        name = args.get("workspace")
        same = bool(ctx.get("workspace")) and name in (None, ctx["workspace"])
        if name is None:
            if not same:
                raise KeyParamsError(_NO_CONTEXT)
            name = ctx["workspace"]
        role = args.get("role") or (ctx.get("role") if same else None) or "train"
        on_screen = same and role == ctx.get("role")
        version = args.get("version") if use_version else None
        if version is None and use_version and on_screen:
            version = ctx.get("version")
        effective = version if version is not None else ctx.get("latest")
        identity = ctx.get("identity") if on_screen and effective == ctx.get("version") else None
        return _View(name, role, version, identity)

    # -- keys / transforms -----------------------------------------------
    async def list_keys(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.list_keys))

    async def key_schema(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.key_schema, _need(args, "key")))

    async def _fill_source(self, key: str, params: dict, session: str | None) -> tuple[dict, str | None, str]:
        """``(params, identity, note)``: ``source`` / ``test`` from the Studio context."""
        props = (await _compute(contract.key_schema, key)).get("properties", {})
        if "source" not in props or "source" in params:
            return params, None, ""
        ctx = await self.port.get_context(session)
        name = (ctx or {}).get("workspace")
        if not name:
            return params, None, _NO_CONTEXT_NOTE
        frame = {"workspace": name, "role": ctx.get("role") or "train", "version": ctx.get("version")}
        params = {**params, "source": {"kind": "dataset", **frame}}
        if "test" in props and "test" not in params and await self._has_test(name):
            params["test"] = {"kind": "dataset", **frame, "role": "test"}
        return params, ctx.get("identity"), ""

    @staticmethod
    async def _has_test(name: str) -> bool:
        ws = await _compute(contract.get_workspace, name)
        return (ws.get("datasets") or {}).get("test") is not None

    async def run_key(self, args: dict) -> dict:
        key = _need(args, "key")
        params, identity, note = await self._fill_source(
            key, dict(args.get("params") or {}), args.get("session")
        )
        include = bool(args.get("include_figures"))
        result = await _compute(contract.run_key, key, params)
        framed = policy.frame(policy.compact_result(result, include), identity=identity)
        return _with_note(framed, note) if note else framed

    async def list_transforms(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.list_transforms))

    async def transform_schema(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.transform_schema, _need(args, "op")))

    # -- workspaces ------------------------------------------------------
    async def list_workspaces(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.list_workspace_summaries))

    async def get_workspace(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.get_workspace, _need(args, "name")))

    async def _workspace_dict(self, view: _View) -> dict:
        return await _compute(contract.get_workspace, view.workspace)

    async def get_rows(self, args: dict) -> dict:
        limit = policy.row_limit(args.get("limit"))
        view = await self._view(args)
        ws = await self._workspace_dict(view)
        data = await _compute(
            contract.workspace_rows, ws, view.role, version=view.version,
            offset=int(args.get("offset") or 0), limit=limit, columns=args.get("columns"),
        )
        return policy.frame(data, identity=view.identity)

    async def get_profiles(self, args: dict) -> dict:
        view = await self._view(args)
        ws = await self._workspace_dict(view)
        data = await _compute(
            contract.column_profiles, ws, view.role, version=view.version,
            columns=args.get("columns"),
        )
        return policy.frame(data, identity=view.identity)

    async def preview_step(self, args: dict) -> dict:
        step = _need(args, "step")
        view = await self._view(args, use_version=False)
        ws = await self._workspace_dict(view)
        data = await _compute(contract.preview_step, ws, step, view.role)
        return policy.frame(data, identity=view.identity)

    async def align_report(self, args: dict) -> dict:
        view = await self._view(args, use_version=False)
        ws = await self._workspace_dict(view)
        return policy.frame(await _compute(contract.align_report, ws))

    async def source_columns(self, args: dict) -> dict:
        return policy.frame(await _compute(contract.source_columns, _need(args, "spec")))

    # -- Studio ----------------------------------------------------------
    async def get_ui_context(self, args: dict) -> dict:
        ctx = await self.port.get_context(args.get("session"))
        framed = policy.frame(ctx, identity=(ctx or {}).get("identity"))
        return framed if ctx else _with_note(framed, "no Studio context published")

    async def get_command_status(self, args: dict) -> dict:
        cid = str(_need(args, "id"))
        status = await self.port.command_status(cid)
        if status is None:
            raise KeyParamsError(f"no recent command {cid!r}")
        return policy.frame(status, identity=status.get("identity"))

    def ui_command(self, spec: CommandSpec) -> Handler:
        async def handler(args: dict) -> dict:
            policy.check_command(spec.type)
            session = args.get("session")
            cmd = {k: v for k, v in args.items() if k != "session"}
            missing = [f for f in spec.context_fill if cmd.get(f) is None]
            if missing:
                ctx = await self.port.get_context(session) or {}
                for field in missing:
                    value = ctx.get(_CONTEXT_KEYS[field])
                    if value is not None:
                        cmd[field] = value
            ack = await self.port.send_command({**cmd, "type": spec.type}, session, COMMAND_TIMEOUT)
            return policy.frame(ack, identity=ack.get("identity"))

        return handler


def _checked(handler: Handler) -> Handler:
    async def run(args: dict) -> dict:
        policy.check_args(args)
        return await handler(args)

    return run


def build_tools(port: UiPort) -> list[ToolSpec]:
    t = _Tools(port)
    session_only = _schema({"session": _SESSION})
    static: list[tuple[str, str, dict, Handler, bool]] = [
        ("list_keys", "List the analysis keys (id, title, category, needs_target).",
         _schema(), t.list_keys, True),
        ("key_schema", "JSON Schema of a key's params.",
         _schema({"key": _STR}, ("key",)), t.key_schema, True),
        ("run_key",
         ("Run an analysis key. params: see key_schema {key}. Without params.source it "
          "analyses the data on screen in Studio (else the demo data). Figures are "
          "reduced to titles unless include_figures."),
         _schema({
             "key": _STR,
             "params": {"type": "object", "description": "Key params (see key_schema)."},
             "include_figures": {"type": "boolean"},
             "session": _SESSION,
         }, ("key",)), t.run_key, True),
        ("list_transforms", "List the transform ops usable in workspace steps.",
         _schema(), t.list_transforms, True),
        ("transform_schema", "JSON Schema of a transform op's params.",
         _schema({"op": _STR}, ("op",)), t.transform_schema, True),
        ("list_workspaces", "Summaries of the saved workspaces.",
         _schema(), t.list_workspaces, True),
        ("get_workspace", "A saved workspace (datasets, steps, ...).",
         _schema({"name": _STR}, ("name",)), t.get_workspace, True),
        ("get_rows",
         (f"Rows of a workspace dataset at a version (default {policy.DEFAULT_ROWS}, "
          f"max {policy.MAX_ROWS})."),
         _schema({
             **_VIEW, "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": policy.MAX_ROWS},
             "columns": _COLUMNS,
         }), t.get_rows, True),
        ("get_profiles", "Per-column profiles (histograms, sentinels, skew, ...).",
         _schema({**_VIEW, "columns": _COLUMNS}), t.get_profiles, True),
        ("preview_step",
         "Dry-run a step on top of the workspace steps; nothing is saved.",
         _schema({
             "step": {
                 "type": "object",
                 "description": "{op, target?, params?}; params: see transform_schema {op}.",
             },
             "workspace": _VIEW["workspace"], "role": _VIEW["role"], "session": _SESSION,
         }, ("step",)), t.preview_step, True),
        ("align_report", "Train / test column alignment after the workspace's steps.",
         _schema({"workspace": _VIEW["workspace"], "session": _SESSION}),
         t.align_report, True),
        ("source_columns", "Column names and kinds of a source spec.",
         _schema({"spec": {"type": "object"}}, ("spec",)), t.source_columns, True),
        ("get_ui_context",
         "What the user sees in Studio: workspace, role, version, selection, windows.",
         session_only, t.get_ui_context, True),
        ("get_command_status",
         "Final ack of a UI command (poll after propose_steps returned pending: review).",
         _schema({"id": _STR}, ("id",)), t.get_command_status, True),
    ]
    tools = [ToolSpec(n, d, s, _checked(h), ro) for n, d, s, h, ro in static]
    tools.extend(
        ToolSpec(
            c.tool_name, c.description, _with_session(c.input_schema),
            _checked(t.ui_command(c)), False,
        )
        for c in UI_COMMANDS.values()
    )
    return tools


def _with_session(schema: dict) -> dict:
    """A UI command's schema plus ``session`` (which Studio tab gets it)."""
    return {**schema, "properties": {**schema.get("properties", {}), "session": _SESSION}}
