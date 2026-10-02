"""``/mcp`` mounted in ``dtk-api``: guard (token / Origin / Host), both paths, optional extra."""

from __future__ import annotations

import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from dtk_engine.http import create_app

TOKEN = "mcp-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "t", "version": "0"},
    },
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    monkeypatch.delenv("DTK_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("DTK_UI_ALLOWED_HOSTS", raising=False)


@pytest.fixture
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="error", timeout_graceful_shutdown=1
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_guard_rejects_before_the_sdk(server, path):
    def post(**headers):
        return httpx.post(server + path, json=INIT, headers={**MCP_HEADERS, **headers})

    assert post().status_code == 401
    assert post(Authorization="Bearer wrong").status_code == 401
    assert post(**AUTH, Origin="http://evil.example").status_code == 403
    assert post(**AUTH, Host="evil.example").status_code == 403
    ok = post(**AUTH)
    assert ok.status_code == 200
    assert "dtk" in ok.text


@pytest.mark.anyio
@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
async def test_tools_listed_with_token(server, path):
    http = create_mcp_http_client(headers=AUTH)
    async with Client(streamable_http_client(server + path, http_client=http)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"run_key", "propose_steps", "get_ui_context"} <= names
        out = await client.call_tool("list_keys", {})
        assert not out.is_error


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_no_redirect_for_either_path(server):
    for path in ("/mcp", "/mcp/"):
        response = httpx.post(server + path, json=INIT, headers={**MCP_HEADERS, **AUTH})
        assert response.status_code == 200


def test_app_without_agent_extra_has_no_mcp(monkeypatch):
    monkeypatch.setitem(sys.modules, "dtk_engine.agent.server", None)
    app = create_app()
    assert not any(getattr(r, "path", "").startswith("/mcp") for r in app.routes)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/keys").status_code == 200
        assert client.post("/mcp", json=INIT, headers=AUTH).status_code == 404
