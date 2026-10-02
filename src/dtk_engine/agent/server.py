"""The MCP server: tools from ``tools.py`` over the SDK's low-level ``Server``.

``build_server`` is shared by the stdio ``dtk-mcp`` (``RemoteUiPort``) and the
``/mcp`` mount of ``dtk-api`` (``mcp_http_app`` with the in-process bridge).
"""

from __future__ import annotations

import json
import time
from contextlib import AbstractAsyncContextManager
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError
from starlette.applications import Starlette

from dtk_engine.agent import policy
from dtk_engine.agent.ports import LocalUiPort, UiPort
from dtk_engine.agent.tools import ToolSpec, build_tools
from dtk_engine.errors import (
    KeyParamsError,
    message_from_validation_details,
    validation_error_details,
)
from dtk_engine.ui_bridge import UiBridge

CONTEXT_URI = "studio://context"
MCP_PATH = "/mcp"


def _error_body(exc: BaseException) -> dict:
    """``{type, message, details}`` like the HTTP API's error body."""
    details = getattr(exc, "details", None)
    cause = exc.__cause__
    if details is None and isinstance(cause, ValidationError):
        details = validation_error_details(cause)
    if details is not None:
        message = message_from_validation_details(details)
    else:
        message = str(exc)
    return {"type": type(exc).__name__, "message": message, "details": details}


def _text_result(payload: Any, *, is_error: bool = False) -> types.CallToolResult:
    text = json.dumps(payload, default=str)
    return types.CallToolResult(content=[types.TextContent(text=text)], is_error=is_error)


def _tool(spec: ToolSpec) -> types.Tool:
    return types.Tool(
        name=spec.name,
        description=spec.description,
        input_schema=spec.input_schema,
        annotations=types.ToolAnnotations(read_only_hint=True) if spec.read_only else None,
    )


def build_server(port: UiPort, audit: policy.AuditLog) -> Server:
    specs = {spec.name: spec for spec in build_tools(port)}
    tools = [_tool(spec) for spec in specs.values()]

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = params.arguments or {}
        started = time.perf_counter()
        spec = specs.get(params.name)
        try:
            if spec is None:
                raise KeyParamsError(f"unknown tool: {params.name}")
            framed = await spec.handler(args)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a tool error
            body = _error_body(exc)
            audit.record(
                params.name, args, status="error", error=body["message"],
                duration_ms=_ms(started),
            )
            return _text_result(body, is_error=True)
        audit.record(
            params.name, args, status="ok", identity=framed.get("identity"),
            duration_ms=_ms(started),
        )
        return _text_result(framed)

    async def list_resources(ctx: Any, params: Any) -> types.ListResourcesResult:
        resource = types.Resource(
            uri=CONTEXT_URI,
            name="studio-context",
            description="What the user sees in Studio (the latest published UI context).",
            mime_type="application/json",
        )
        return types.ListResourcesResult(resources=[resource])

    async def read_resource(
        ctx: Any, params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        if str(params.uri) != CONTEXT_URI:
            raise ValueError(f"unknown resource: {params.uri}")
        context = await port.get_context(None)
        content = types.TextResourceContents(
            uri=CONTEXT_URI, mime_type="application/json", text=json.dumps(context)
        )
        return types.ReadResourceResult(contents=[content])

    return Server(
        "dtk",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def mcp_http_app(
    bridge: UiBridge, audit: policy.AuditLog | None = None
) -> tuple[Starlette, AbstractAsyncContextManager[None]]:
    """``(app, lifespan)``: the streamable-HTTP app (route ``/mcp``) and its session manager.

    The host app must enter ``lifespan`` and guard the app itself: the SDK's own
    DNS-rebinding protection is disabled so ``http.py`` keeps a single allow-list.
    """
    server = build_server(LocalUiPort(bridge), audit or policy.AuditLog())
    app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return app, server.session_manager.run()
