"""Agent tools: the static contract tools plus one tool per UI command.

``build_tools(port)`` is transport-agnostic; ``server.py`` turns each
``ToolSpec`` into an MCP tool. Every handler runs ``policy.check_args`` on its
arguments first and frames its output with ``policy.frame``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import anyio.to_thread

from dtk_engine import contract
from dtk_engine.agent import attachments, policy
from dtk_engine.agent.commands import UI_COMMANDS, CommandSpec
from dtk_engine.agent.digest import TRACKER, frame_identity
from dtk_engine.agent.ports import UiPort
from dtk_engine.errors import KeyParamsError

COMMAND_TIMEOUT = 30.0
_NO_CONTEXT = "no Studio context: pass workspace"
_NO_CONTEXT_NOTE = "no Studio context: key defaults (demo data) used"
# UI-command fields filled from the Studio context: field -> context key.
_CONTEXT_KEYS = {"workspace": "workspace", "base_identity": "identity"}

Handler = Callable[[dict], Awaitable[dict]]
# Steps the agent had seen before the current tool call ({id: content}), set
# by ``_Tools.tracked`` before it records the new state: propose_steps sends
# them as ``base_steps``.
_SEEN_BEFORE: ContextVar[dict[str, dict] | None] = ContextVar("seen_before", default=None)


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
_DETAIL = {"type": "boolean", "description": "Full result (default: compact)."}


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
    def tracked(self, handler: Handler) -> Handler:
        """Every result: ``identity`` filled (else the frame shown in Studio) and
        ``workspace_changes`` when the steps changed since the agent last saw them."""

        async def run(args: dict) -> dict:
            ctx = await self.port.get_context(args.get("session")) or {}
            changes = await self._changes(args, ctx)
            out = await handler(args)
            if not isinstance(out, dict):
                return out
            if out.get("identity") is None and ctx.get("identity"):
                out = {**out, "identity": ctx["identity"]}
            if changes:
                out = {**out, "workspace_changes": changes}
            return out

        return run

    async def _changes(self, args: dict, ctx: dict) -> dict | None:
        name = args.get("workspace") or args.get("name") or ctx.get("workspace")
        if not isinstance(name, str) or not name:
            return None
        key = (ctx.get("session") or "", name)
        try:
            steps = await _compute(contract.workspace_steps, name)
        except (KeyError, KeyParamsError, OSError, ValueError):
            return None
        _SEEN_BEFORE.set(TRACKER.seen(key))
        diff = TRACKER.changes(key, steps) or {}
        commands = await self._settled(key)
        if commands:
            diff["commands"] = commands
        if not diff:
            return None
        identity = ctx.get("identity") if ctx.get("workspace") == name else None
        return {"workspace": name, "identity": identity, **diff}

    async def _settled(self, key: tuple[str, str]) -> list[dict]:
        """The agent's proposals that left review since (applied / why not)."""
        out = []
        for cid in sorted(TRACKER.pending(key)):
            status = await self.port.command_status(cid)
            if status is None or status.get("ok") is None:
                continue
            TRACKER.settle(key, cid)
            out.append({"id": cid, "status": "applied" if status["ok"] else status.get("error")})
        return out

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
        keys = await _compute(contract.list_keys)
        if not args.get("detail"):
            keys = policy.compact_catalog(keys, ("id", "title", "category", "needs_target"))
        return policy.frame(keys)

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
        if identity is None:
            identity = await _source_identity(params.get("source"))
        framed = policy.frame(policy.compact_result(result, include), identity=identity)
        return _with_note(framed, note) if note else framed

    async def list_transforms(self, args: dict) -> dict:
        ops = await _compute(contract.list_transforms)
        if not args.get("detail"):
            ops = policy.compact_catalog(ops, ("op", "title", "needs_target"))
        return policy.frame(ops)

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
        return policy.frame(data, identity=_identity(view, ws))

    async def get_profiles(self, args: dict) -> dict:
        view = await self._view(args)
        ws = await self._workspace_dict(view)
        data = await _compute(
            contract.column_profiles, ws, view.role, version=view.version,
            columns=args.get("columns"),
        )
        if not args.get("columns") and not args.get("detail"):
            data = policy.compact_profiles(data)
        return policy.frame(data, identity=_identity(view, ws))

    async def preview_step(self, args: dict) -> dict:
        # Same default as Studio applies to a proposed step.
        step = {"target": "both", **_need(args, "step")}
        view = await self._view(args, use_version=False)
        ws = await self._workspace_dict(view)
        data = await _compute(contract.preview_step, ws, step, view.role)
        compact = policy.compact_preview(data, bool(args.get("detail")))
        return policy.frame(compact, identity=_identity(view, ws))

    async def align_report(self, args: dict) -> dict:
        view = await self._view(args, use_version=False)
        ws = await self._workspace_dict(view)
        data = await _compute(contract.align_report, ws)
        if not args.get("detail"):
            data = policy.compact_align(data)
        return policy.frame(data, identity=_identity(view, ws))

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

    # -- attachments (read-only) -----------------------------------------
    async def _attachments(self, args: dict) -> tuple[str, list[dict]]:
        session = args.get("session")
        if session is None:
            session = ((await self.port.get_context(None)) or {}).get("session")
        if session is None:
            raise KeyParamsError("no Studio session: pass session")
        return session, await self.port.list_attachments(session)

    async def list_attachments(self, args: dict) -> dict:
        _, found = await self._attachments(args)
        return policy.frame(found)

    async def read_attachment(self, args: dict) -> dict:
        att_id = str(_need(args, "id"))
        _, found = await self._attachments(args)
        att = next((a for a in found if a["id"] == att_id), None)
        if att is None:
            raise KeyParamsError(f"no attachment {att_id!r} in this session")
        return await _compute(
            attachments.read_text, att, int(args.get("offset") or 0), args.get("max_chars")
        )

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
            key = None
            if spec.type == "propose_steps" and isinstance(cmd.get("workspace"), str):
                ctx = await self.port.get_context(session) or {}
                key = (ctx.get("session") or "", cmd["workspace"])
                cmd = _with_base_steps(cmd)
            ack = await self.port.send_command({**cmd, "type": spec.type}, session, COMMAND_TIMEOUT)
            if key is not None:
                await _after_proposal(key, ack)
            return policy.frame(ack, identity=ack.get("identity"))

        return handler


def _identity(view: _View, ws: dict) -> str:
    """Studio's identity when the frame is on screen, else computed."""
    return view.identity or frame_identity(ws, view.role, view.version)


async def _source_identity(source: Any) -> str | None:
    if not isinstance(source, dict) or source.get("kind") != "dataset":
        return None
    try:
        ws = await _compute(contract.get_workspace, source.get("workspace"))
    except (KeyError, KeyParamsError, OSError, ValueError):
        return None
    return frame_identity(ws, source.get("role") or "train", source.get("version"))


def _targeted_ids(ops: Any) -> list[str]:
    ids = []
    for op in ops if isinstance(ops, list) else []:
        for kind in ("replace", "remove"):
            body = op.get(kind) if isinstance(op, dict) else None
            if isinstance(body, dict) and isinstance(body.get("id"), str):
                ids.append(body["id"])
    return ids


def _with_base_steps(cmd: dict) -> dict:
    """``base_steps``: what the agent last saw of each step id it targets."""
    seen = _SEEN_BEFORE.get() or {}
    base = {i: seen[i] for i in _targeted_ids(cmd.get("ops")) if i in seen}
    if not base or "base_steps" in cmd:
        return cmd
    return {**cmd, "base_steps": base}


async def _after_proposal(key: tuple[str, str], ack: dict) -> None:
    """The agent's own applied edit is not a user change: observe it now."""
    if ack.get("pending") == "review" and ack.get("id"):
        TRACKER.add_pending(key, str(ack["id"]))
    elif ack.get("ok"):
        with contextlib.suppress(KeyError, KeyParamsError, OSError, ValueError):
            TRACKER.observe(key, await _compute(contract.workspace_steps, key[1]))


def _checked(handler: Handler) -> Handler:
    async def run(args: dict) -> dict:
        policy.check_args(args)
        return await handler(args)

    return run


def build_tools(port: UiPort) -> list[ToolSpec]:
    t = _Tools(port)
    session_only = _schema({"session": _SESSION})
    static: list[tuple[str, str, dict, Handler, bool]] = [
        ("list_keys",
         "List the analysis keys (id, title, category, needs_target; descriptions with detail).",
         _schema({"detail": _DETAIL}), t.list_keys, True),
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
        ("list_transforms",
         "List the transform ops usable in workspace steps (op, title; descriptions with detail).",
         _schema({"detail": _DETAIL}), t.list_transforms, True),
        ("transform_schema", "JSON Schema of a transform op's params.",
         _schema({"op": _STR}, ("op",)), t.transform_schema, True),
        ("list_workspaces", "Summaries of the saved workspaces.",
         _schema(), t.list_workspaces, True),
        ("get_workspace",
         "A saved workspace (datasets, steps with their stable `id`, ...).",
         _schema({"name": _STR}, ("name",)), t.get_workspace, True),
        ("get_rows",
         (f"Rows of a workspace dataset at a version (default {policy.DEFAULT_ROWS}, "
          f"max {policy.MAX_ROWS})."),
         _schema({
             **_VIEW, "offset": {"type": "integer", "minimum": 0},
             "limit": {"type": "integer", "minimum": 1, "maximum": policy.MAX_ROWS},
             "columns": _COLUMNS,
         }), t.get_rows, True),
        ("get_profiles",
         ("Per-column profiles (kind, missing, distinct, top values, sentinels, skew, ...). "
          "Without columns: compact (no histograms); detail: true for everything."),
         _schema({**_VIEW, "columns": _COLUMNS, "detail": _DETAIL}), t.get_profiles, True),
        ("preview_step",
         (f"Dry-run a step on top of the workspace steps; nothing is saved. Changed "
          f"cells, removed rids and the fitted state are cut to {policy.PREVIEW_ITEMS} "
          f"items ({policy.PREVIEW_DETAIL_ITEMS} with detail); changed_total and "
          "elided give the full counts."),
         _schema({
             "step": {
                 "type": "object",
                 "description": "{op, target?, params?}; params: see transform_schema {op}.",
             },
             "detail": {"type": "boolean", "description": "Larger samples (default false)."},
             "workspace": _VIEW["workspace"], "role": _VIEW["role"], "session": _SESSION,
         }, ("step",)), t.preview_step, True),
        ("align_report",
         ("Train / test column alignment after the workspace's steps; matching columns "
          "are listed by name only unless detail."),
         _schema({"workspace": _VIEW["workspace"], "session": _SESSION, "detail": _DETAIL}),
         t.align_report, True),
        ("source_columns", "Column names and kinds of a source spec.",
         _schema({"spec": {"type": "object"}}, ("spec",)), t.source_columns, True),
        ("get_ui_context",
         "What the user sees in Studio: workspace, role, version, selection, windows.",
         session_only, t.get_ui_context, True),
        ("list_attachments",
         ("Files the user attached to the chat (id, name, path, size, kind, columns). "
          "Read-only: tables are read with the usual tools and a source spec on their "
          "path; text with read_attachment."),
         session_only, t.list_attachments, True),
        ("read_attachment",
         (f"Text of an attached file of kind text, framed as data (at most "
          f"{policy.max_response_chars()} characters per call; continue at next_offset)."),
         _schema({
             "id": _STR, "offset": {"type": "integer", "minimum": 0},
             "max_chars": {"type": "integer", "minimum": 1}, "session": _SESSION,
         }, ("id",)), t.read_attachment, True),
        ("get_command_status",
         "Final ack of a UI command (poll after propose_steps returned pending: review).",
         _schema({"id": _STR}, ("id",)), t.get_command_status, True),
    ]
    tools = [ToolSpec(n, d, s, t.tracked(_checked(h)), ro) for n, d, s, h, ro in static]
    tools.extend(
        ToolSpec(
            c.tool_name, c.description, _with_session(c.input_schema),
            t.tracked(_checked(t.ui_command(c))), False,
        )
        for c in UI_COMMANDS.values()
    )
    return tools


def _with_session(schema: dict) -> dict:
    """A UI command's schema plus ``session`` (which Studio tab gets it)."""
    return {**schema, "properties": {**schema.get("properties", {}), "session": _SESSION}}
