"""HTTP API over the JSON contract (optional extra ``api``).

Thin FastAPI routes under ``/api`` that call ``dtk_engine.contract``; no
per-key logic. Install with ``pip install 'dtk-engine[api]'`` (or
``uv sync --extra api``) then run ``dtk-api``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path, PurePath
from typing import Literal

# FastAPI is an optional dependency; keep the import inside this module so
# ``import dtk_engine`` still works without the ``api`` extra.
from fastapi import APIRouter, Depends, FastAPI, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from starlette.requests import HTTPConnection
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from dtk_engine import contract
from dtk_engine.agent.attachments import (
    NotAnUploadError,
    UnknownAttachmentError,
    describe_upload,
    registry,
    upload_dir,
)
from dtk_engine.agent.commands import command_schemas
from dtk_engine.errors import (
    KeyParamsError,
    SourceError,
    UnknownKeyError,
    UnknownTransformError,
    message_from_validation_details,
    validation_error_details,
)
from dtk_engine.ui_bridge import (
    UiBridge,
    allowed_hosts,
    clear_runtime,
    host_name,
    write_runtime,
)
from dtk_engine.workspace import WorkspaceNotFoundError


def _cors_origins() -> list[str]:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra = os.environ.get("DTK_CORS_ORIGINS", "")
    for part in extra.split(","):
        origin = part.strip()
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def _save_upload(name: str, data: bytes, root: Path | None = None) -> Path:
    """Write an uploaded file under ``root`` and return its absolute path.

    Stored as ``<root>/<content hash>/<file name>``: same-name uploads with
    other contents never overwrite a file a saved workspace still reads.
    """
    digest = hashlib.sha256(data).hexdigest()[:12]
    base = PurePath(name.replace("\\", "/")).name
    if base in ("", ".", ".."):
        base = "upload"
    path = (root or upload_dir()) / digest / base
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.resolve()


def _error_body(exc: BaseException) -> dict:
    """JSON error payload: ``{type, message}`` plus ``details`` for validation.

    When the failure is a pydantic ``ValidationError`` (as ``exc.details`` or
    ``exc.__cause__``), ``message`` is the concise joined inner issues — never
    the full pydantic dump — and ``details`` is ``[{loc, msg, type}, ...]``.
    """
    details = getattr(exc, "details", None)
    cause = exc.__cause__
    if details is None and isinstance(cause, ValidationError):
        details = validation_error_details(cause)
    if details is not None:
        return {
            "type": type(exc).__name__,
            "message": message_from_validation_details(details),
            "details": details,
        }
    return {"type": type(exc).__name__, "message": str(exc)}


async def _body(request: Request, *required: str, message: str = "") -> dict:
    """JSON object body holding every ``required`` key, else a ``KeyParamsError``."""
    body = await request.json()
    if not isinstance(body, dict) or any(key not in body for key in required):
        raise KeyParamsError(message or f"body must include {' and '.join(required)}")
    return body


# Read-only compute endpoints run the (sync, CPU-bound) engine call in the
# threadpool: on the event loop, one slow key (feature_selection: tens of
# seconds) would stall every other request, previews included (MAT-212).
# Store writes stay on the loop, so they remain serialized.
keys_router, transforms_router, workspaces_router, studio_router, uploads_router = (
    APIRouter(prefix="/api") for _ in range(5)
)


@keys_router.get("/keys")
def get_keys() -> list[dict]:
    return contract.list_keys()


@keys_router.get("/keys/{key_id}/schema")
def get_key_schema(key_id: str) -> dict:
    return contract.key_schema(key_id)


@keys_router.post("/keys/{key_id}/run")
async def post_run_key(key_id: str, request: Request) -> dict:
    body = await _body(request, "params", message="body must be {params: ...}")
    return await run_in_threadpool(contract.run_key, key_id, body["params"])


@transforms_router.get("/transforms")
def get_transforms() -> list[dict]:
    return contract.list_transforms()


@transforms_router.get("/transforms/{op}/schema")
def get_transform_schema(op: str) -> dict:
    return contract.transform_schema(op)


@workspaces_router.get("/workspaces")
def get_workspaces() -> list[dict]:
    return contract.list_workspaces()


@workspaces_router.get("/workspaces/summaries")
def get_workspace_summaries() -> list[dict]:
    # Registered before /workspaces/{name} so "summaries" is not a name.
    return contract.list_workspace_summaries()


@workspaces_router.get("/workspaces/{name}")
def get_one_workspace(name: str) -> dict:
    return contract.get_workspace(name)


@workspaces_router.put("/workspaces/{name}")
async def put_workspace(name: str, request: Request) -> dict:
    body = await _body(request, message="workspace body must be a JSON object")
    return contract.save_workspace({**body, "name": name})


@workspaces_router.delete("/workspaces/{name}", status_code=204)
def delete_one_workspace(name: str) -> Response:
    contract.delete_workspace(name)
    return Response(status_code=204)


@workspaces_router.post("/workspaces/{name}/rename")
async def post_rename(name: str, request: Request) -> dict:
    body = await _body(request, "new_name", message="body must be {new_name: ...}")
    return contract.rename_workspace(name, body["new_name"])


@workspaces_router.post("/workspaces/{name}/duplicate")
async def post_duplicate(name: str, request: Request) -> dict:
    body = await _body(request, "new_name", message="body must be {new_name: ...}")
    return contract.duplicate_workspace(name, body["new_name"])


@workspaces_router.post("/workspaces/{name}/export")
async def post_export(name: str, request: Request) -> dict:
    body = await _body(request, "out_dir")
    return contract.export_workspace(
        name, body["out_dir"], overwrite=bool(body.get("overwrite", False))
    )


@studio_router.post("/source/columns")
async def post_source_columns(request: Request) -> list[dict]:
    body = await _body(request, "spec", message="body must be {spec: ...}")
    return await run_in_threadpool(contract.source_columns, body["spec"])


@studio_router.post("/workspace/preview")
async def post_preview(request: Request) -> dict:
    body = await _body(request, "workspace")
    return await run_in_threadpool(
        contract.preview_workspace, body["workspace"], body.get("role", "train"),
        head_rows=int(body.get("head_rows", 5)),
    )


@studio_router.post("/workspace/rows")
async def post_rows(request: Request) -> dict:
    body = await _body(request, "workspace")
    return await run_in_threadpool(
        contract.workspace_rows, body["workspace"], body.get("role", "train"),
        version=body.get("version"), offset=int(body.get("offset", 0)),
        limit=int(body.get("limit", 500)), columns=body.get("columns"),
        filter=body.get("filter"), sort=body.get("sort"),
    )


@studio_router.post("/workspace/profiles")
async def post_profiles(request: Request) -> dict:
    body = await _body(request, "workspace")
    return await run_in_threadpool(
        contract.column_profiles, body["workspace"], body.get("role", "train"),
        version=body.get("version"), columns=body.get("columns"),
    )


@studio_router.post("/workspace/preview-step")
async def post_preview_step(request: Request) -> dict:
    body = await _body(request, "workspace", "step")
    return await run_in_threadpool(
        contract.preview_step, body["workspace"], body["step"], body.get("role", "train")
    )


@studio_router.post("/workspace/align")
async def post_align(request: Request) -> dict:
    body = await _body(request, "workspace")
    return await run_in_threadpool(contract.align_report, body["workspace"])


@uploads_router.put("/uploads/{filename}")
async def put_upload(filename: str, request: Request) -> dict:
    data = await request.body()
    path = _save_upload(filename, data)
    return {"path": str(path)}


# -- UI bridge (/api/ui): Studio <-> agent relay, guarded by token/Origin/Host --


class _UiGuardError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


async def _ui_guard_response(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, _UiGuardError)
    body = {"type": "UiBridgeError", "message": str(exc), "details": None}
    return JSONResponse(status_code=exc.status, content=body)


def _bridge(request: HTTPConnection) -> UiBridge:
    return request.app.state.ui_bridge


def _presented_token(request: HTTPConnection) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token", "")


_HTTP_SCHEME = {"ws": "http", "wss": "https"}


def _check_access(request: HTTPConnection, token: str, *, need_origin: bool = False) -> None:
    """Host (DNS rebinding), then Origin, then token; shared by ``/api/ui`` and ``/mcp``.

    ``need_origin``: a browser always sends ``Origin`` on a WebSocket, and no
    CORS preflight protects one, so the terminal socket refuses a missing one.
    """
    if host_name(request.headers.get("host", "")) not in allowed_hosts():
        raise _UiGuardError(403, "host not allowed")
    origin = request.headers.get("origin")
    if origin is None and need_origin:
        raise _UiGuardError(403, "origin required")
    if origin is not None:
        scheme = _HTTP_SCHEME.get(request.url.scheme, request.url.scheme)
        same = f"{scheme}://{request.headers.get('host', '')}"
        if origin != same and origin not in _cors_origins():
            raise _UiGuardError(403, "origin not allowed")
    presented = _presented_token(request).encode()
    if not secrets.compare_digest(presented, token.encode()):
        raise _UiGuardError(401, "missing or invalid UI token")


def _ui_guard(request: Request) -> None:
    """Router dependency for ``/api/ui``."""
    _check_access(request, _bridge(request).token)


class _McpGuard:
    """Pure-ASGI wrapper applying the ``/api/ui`` checks in front of the ``/mcp`` app.

    The SDK app lives in ``app.state.mcp_app`` while the lifespan runs (its
    session manager runs once per instance). Serves ``/mcp`` and ``/mcp/``
    alike (the inner app only knows ``/mcp``) so clients never see a redirect.
    """

    def __init__(self, bridge: UiBridge) -> None:
        self.bridge = bridge

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        inner = getattr(scope["app"].state, "mcp_app", None)
        try:
            _check_access(Request(scope), self.bridge.token)
            if inner is None:
                raise _UiGuardError(503, "MCP endpoint not running")
        except _UiGuardError as exc:
            body = {"type": "UiBridgeError", "message": str(exc), "details": None}
            await JSONResponse(status_code=exc.status, content=body)(scope, receive, send)
            return
        await inner({**scope, "path": "/mcp"}, receive, send)


_MAX_COMMAND_TIMEOUT = 120.0

ui_router = APIRouter(prefix="/api/ui", dependencies=[Depends(_ui_guard)])


class UiContext(BaseModel):
    """Studio's published view state; only ``session`` is required."""

    model_config = ConfigDict(extra="allow")
    session: str


class UiAck(BaseModel):
    """Final ack ``{id, ok, error?, identity?}`` or interim ``{id, pending: "review"}``."""

    id: str
    ok: bool | None = None
    pending: Literal["review"] | None = None
    error: str | None = None
    identity: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> UiAck:
        if (self.ok is None) == (self.pending is None):
            raise ValueError("an ack sets exactly one of ok / pending")
        return self


def _ui_error(status: int, type_: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"type": type_, "message": message, "details": None}
    )


@ui_router.put("/context", status_code=204)
async def put_ui_context(context: UiContext, request: Request) -> Response:
    _bridge(request).put_context(context.model_dump(exclude_unset=True))
    return Response(status_code=204)


@ui_router.get("/context", response_model=None)
async def get_ui_context(request: Request, session: str | None = None):
    context = _bridge(request).get_context(session)
    if context is None:
        return _ui_error(404, "NoUiContext", "no UI context published")
    return context


async def _sse_stream(
    request: Request, bridge: UiBridge, session: str, ping_interval: float
) -> AsyncIterator[str]:
    queue = bridge.add_listener(session)
    try:
        yield ": connected\n\n"
        while not await request.is_disconnected():
            try:
                cmd = await asyncio.wait_for(queue.get(), ping_interval)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            if cmd is None:  # bridge closing
                return
            event, data = cmd if isinstance(cmd, tuple) else ("command", cmd)
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
    finally:
        bridge.remove_listener(session, queue)


@ui_router.get("/events")
async def get_ui_events(request: Request, session: str) -> StreamingResponse:
    stream = _sse_stream(request, _bridge(request), session, request.app.state.ui_ping_interval)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@ui_router.post("/ack", response_model=None)
async def post_ui_ack(ack: UiAck, request: Request):
    if not _bridge(request).ack(ack.model_dump()):
        return _ui_error(404, "UnknownCommand", f"no pending command {ack.id!r}")
    return {"ok": True}


@ui_router.post("/commands")
async def post_ui_command(request: Request) -> dict:
    body = await _body(request, "type", message="body must be {type: ..., ...}")
    session = request.query_params.get("session") or body.pop("session", None)
    try:
        timeout = float(request.query_params.get("timeout") or body.pop("timeout", 30.0))
    except (TypeError, ValueError):
        raise KeyParamsError("timeout must be a number") from None
    body.pop("timeout", None)
    body.pop("session", None)
    return await _bridge(request).send_command(
        body, session=session, timeout=min(max(timeout, 0.0), _MAX_COMMAND_TIMEOUT)
    )


@ui_router.get("/sessions")
async def get_ui_sessions(request: Request) -> list[dict]:
    """Known Studio sessions (``dtk-mcp doctor``: is a Studio tab listening?)."""
    return _bridge(request).sessions()


@ui_router.get("/commands/schema")
async def get_ui_command_schema() -> dict:
    """``{type: {input_schema, destructive}}``: the UI commands agents may send.

    Declared before ``/commands/{cid}`` so it is not read as a command id.
    """
    return command_schemas()


@ui_router.get("/commands/{cid}", response_model=None)
async def get_ui_command(cid: str, request: Request):
    status = _bridge(request).command_status(cid)
    if status is None:
        return _ui_error(404, "UnknownCommand", f"no recent command {cid!r}")
    return status


# -- in-Studio agent chat (/api/ui/agent): protocol in docs/agent-chat-protocol.md --

_NO_AGENT_EXTRA = "extra agent not installed (uv sync --extra agent-sdk)"


class AgentSession(BaseModel):
    session: str


class AgentSend(AgentSession):
    text: str
    attachments: list[str] = []


class AgentAttach(AgentSession):
    path: str


class AgentConfig(AgentSession):
    pack: str
    model: str | None = None


class AgentPermission(AgentSession):
    id: str
    allow: bool


def _agent_hub(request: Request):
    """The app's ``AgentHub`` (built in the lifespan), or None without the extra."""
    return getattr(request.app.state, "agent_hub", None)


@ui_router.get("/agent")
def get_agent(request: Request, session: str | None = None) -> dict:
    hub = _agent_hub(request)
    if hub is None:
        return {
            "available": False, "pack": None, "reason": _NO_AGENT_EXTRA, "running": False,
            "usage": {"input_tokens": 0, "output_tokens": 0}, "max_tokens": None,
        }
    return hub.status(session)


@ui_router.post("/agent/send", status_code=202, response_model=None)
async def post_agent_send(body: AgentSend, request: Request):
    hub = _agent_hub(request)
    if hub is None:
        return _ui_error(503, "NoAgent", _NO_AGENT_EXTRA)
    from dtk_engine.agent.attachments import UnknownAttachmentError
    from dtk_engine.agent.chat import AgentBusyError, NoAgentError

    await run_in_threadpool(hub.unavailable, body.session)  # first auth probe may spawn the CLI
    try:
        turn = hub.send(body.session, body.text, body.attachments)
    except UnknownAttachmentError as exc:
        return _ui_error(422, "UnknownAttachment", str(exc))
    except NoAgentError as exc:
        return _ui_error(503, "NoAgent", str(exc))
    except AgentBusyError as exc:
        return _ui_error(409, "AgentBusy", str(exc))
    return {"turn": turn}


@ui_router.get("/agent/options")
async def get_agent_options(request: Request, refresh: bool = False) -> dict:
    hub = _agent_hub(request)
    if hub is None:
        return {"available": False, "reason": _NO_AGENT_EXTRA, "default": None, "packs": []}
    return await hub.options(refresh)


@ui_router.post("/agent/config", response_model=None)
async def post_agent_config(body: AgentConfig, request: Request):
    hub = _agent_hub(request)
    if hub is None:
        return _ui_error(503, "NoAgent", _NO_AGENT_EXTRA)
    from dtk_engine.agent.chat import AgentBusyError, ConfigError, NoAgentError

    try:
        return await hub.configure(body.session, body.pack, body.model)
    except NoAgentError as exc:
        return _ui_error(503, "NoAgent", str(exc))
    except AgentBusyError as exc:
        return _ui_error(409, "AgentBusy", str(exc))
    except ConfigError as exc:
        return _ui_error(422, exc.code, str(exc))


@ui_router.post("/agent/cancel")
async def post_agent_cancel(body: AgentSession, request: Request) -> dict:
    hub = _agent_hub(request)
    return {"cancelled": bool(hub is not None and await hub.cancel(body.session))}


@ui_router.post("/agent/permission", response_model=None)
async def post_agent_permission(body: AgentPermission, request: Request):
    hub = _agent_hub(request)
    if hub is None or not hub.reply(body.session, body.id, body.allow):
        return _ui_error(404, "UnknownPermission", f"no pending permission {body.id!r}")
    return {"ok": True}


# -- agent terminal (WS /api/ui/terminal): a CLI in a PTY, docs/agent-chat-protocol.md --

_NO_TERMINAL = "extra agent not installed, or no POSIX PTY here (uv sync --extra agent)"


async def ui_terminal(websocket: WebSocket) -> None:
    """Accepted first, then refused with a 44xx code, so the browser sees why."""
    await websocket.accept()
    try:
        _check_access(websocket, _bridge(websocket).token, need_origin=True)
    except _UiGuardError as exc:
        await websocket.close(4401 if exc.status == 401 else 4403, str(exc))
        return
    terminals = getattr(websocket.app.state, "terminals", None)
    if terminals is None:
        await websocket.close(4403, _NO_TERMINAL)
        return
    from dtk_engine.agent.terminal import serve

    await serve(websocket, terminals, dict(websocket.query_params))


# -- chat attachments (read-only): protocol in docs/agent-chat-protocol.md --


@ui_router.post("/agent/attachments", response_model=None)
async def post_agent_attachment(body: AgentAttach, request: Request):
    try:  # file checks off the loop; registering (and its SSE event) on it
        described = await run_in_threadpool(describe_upload, body.path)
    except NotAnUploadError as exc:
        return _ui_error(422, "NotAnUpload", str(exc))
    return registry(_bridge(request)).register(body.session, described)


@ui_router.get("/agent/attachments", response_model=None)
async def get_agent_attachments(session: str, request: Request):
    return registry(_bridge(request)).list(session)


@ui_router.delete("/agent/attachments/{att_id}", response_model=None)
async def delete_agent_attachment(att_id: str, session: str, request: Request):
    try:
        registry(_bridge(request)).remove(session, att_id)
    except UnknownAttachmentError as exc:
        return _ui_error(404, "UnknownAttachment", str(exc))
    return {"removed": True}


_ERROR_STATUS: dict[type[Exception], int] = {
    **dict.fromkeys((UnknownKeyError, UnknownTransformError, WorkspaceNotFoundError), 404),
    **dict.fromkeys((KeyParamsError, SourceError), 422),
    Exception: 500,
}


async def _error_response(_request: Request, exc: Exception) -> JSONResponse:
    status = next(code for t, code in _ERROR_STATUS.items() if isinstance(exc, t))
    return JSONResponse(status_code=status, content=_error_body(exc))


def create_app(*, ui_ping_interval: float = 15.0) -> FastAPI:
    """Build the FastAPI app (CORS, contract routes under ``/api``, UI bridge ``/api/ui``).

    One ``UiBridge`` per app, at ``app.state.ui_bridge`` (``.token`` is the
    per-run token the Studio launcher hands to the front).
    """

    bridge = UiBridge()
    try:  # the optional extra ``agent`` adds ``/mcp`` and the chat hub; without it, neither
        from dtk_engine.agent.packs.chat_packs import hub_from_env
        from dtk_engine.agent.policy import AuditLog
        from dtk_engine.agent.server import mcp_http_app
    except ImportError:
        mcp_http_app = None
    try:  # terminal packs: extra ``agent`` and a POSIX PTY
        from dtk_engine.agent.terminal import Terminals
    except ImportError:
        Terminals = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if mcp_http_app is not None:
                audit = AuditLog()
                app.state.mcp_app, mcp_lifespan = mcp_http_app(bridge, audit)
                await stack.enter_async_context(mcp_lifespan)
                # Built here, not at import: ``dtk-api --agent`` sets DTK_AGENT_PACK first.
                app.state.agent_hub = hub_from_env(bridge, audit)
                stack.push_async_callback(app.state.agent_hub.close)
            if Terminals is not None:
                stack.push_async_callback(app.state.terminals.close)
            yield
            app.state.ui_bridge.close()

    application = FastAPI(title="dtk-api", docs_url=None, redoc_url=None, lifespan=lifespan)
    application.state.ui_bridge = bridge
    application.state.ui_ping_interval = ui_ping_interval
    if Terminals is not None:
        application.state.terminals = Terminals()
    application.add_middleware(
        CORSMiddleware, allow_origins=_cors_origins(), allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )
    for exc_type in _ERROR_STATUS:
        application.add_exception_handler(exc_type, _error_response)
    application.add_exception_handler(_UiGuardError, _ui_guard_response)
    for router in (
        keys_router, transforms_router, workspaces_router, studio_router, uploads_router,
        ui_router,
    ):
        application.include_router(router)
    if mcp_http_app is not None:
        guarded = _McpGuard(bridge)
        application.router.routes.extend(Route(path, guarded) for path in ("/mcp", "/mcp/"))
    application.add_api_websocket_route("/api/ui/terminal", ui_terminal)
    return application


app = create_app()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``dtk-api [--host HOST] [--port PORT] [--agent [PACK]] [--terminal]``."""
    parser = argparse.ArgumentParser(prog="dtk-api", description="datatoolkit HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--agent", nargs="?", const="agent-sdk", metavar="PACK",
        help="enable the in-Studio agent chat (pack: agent-sdk (default), api-anthropic, "
        "api-openai or stub; same as DTK_AGENT_PACK)",
    )
    parser.add_argument(
        "--terminal", action="store_true",
        help="allow agent CLIs (claude, gemini, opencode) in a Studio terminal "
        "(same as DTK_AGENT_TERMINAL=1)",
    )
    args = parser.parse_args(argv)
    if args.agent:
        os.environ["DTK_AGENT_PACK"] = args.agent
    if args.terminal:
        os.environ["DTK_AGENT_TERMINAL"] = "1"
    import uvicorn

    # Runtime file: lets the Vite dev server and dtk-mcp find this run's token.
    host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url_host = f"[{host}]" if ":" in host else host
    write_runtime(f"http://{url_host}:{args.port}", app.state.ui_bridge.token)
    try:
        # SSE streams never end by themselves: cap the graceful wait so Ctrl-C exits.
        uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=2)
    finally:
        clear_runtime()


if __name__ == "__main__":
    main()
