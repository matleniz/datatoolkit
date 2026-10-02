"""Stdio ``dtk-mcp``: the CLI entry and ``RemoteUiPort`` against a real engine."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn
from mcp import Client, StdioServerParameters

from dtk_engine.agent import cli
from dtk_engine.agent.ports import RemoteUiPort
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import write_runtime

TOKEN = "cli-test-token"

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DTK_HOME", str(h))
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.delenv("DTK_UI_RUNTIME_FILE", raising=False)
    return h


@pytest.fixture
def engine():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app()
    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", timeout_graceful_shutdown=1)
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    url = f"http://127.0.0.1:{port}"
    write_runtime(url, TOKEN)
    yield app, url
    srv.should_exit = True
    thread.join(5)


def test_main_defaults_to_serve(monkeypatch):
    calls = []
    monkeypatch.setitem(cli.SUBCOMMANDS, "serve", ("x", lambda args: calls.append(args.command)))
    cli.main([])
    cli.main(["serve"])
    assert calls == [None, "serve"]


async def test_no_runtime_file_means_no_studio():
    port = RemoteUiPort()
    assert await port.get_context(None) is None
    assert await port.command_status("c1") is None
    ack = await port.send_command({"type": "set_view", "id": "c9"}, None, 1.0)
    assert ack == {"id": "c9", "ok": False, "error": "no_studio"}


async def test_remote_port_against_running_engine(engine):
    app, url = engine
    port = RemoteUiPort()
    assert await port.get_context(None) is None  # engine up, nothing published
    await asyncio.to_thread(
        httpx.put, url + "/api/ui/context", json={"session": "s1", "workspace": "w"},
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert (await port.get_context(None))["workspace"] == "w"
    ack = await port.send_command({"type": "set_view", "role": "test"}, None, 1.0)
    assert ack["error"] == "no_studio"

    # A fake Studio on the SSE route: reads one command, acks it over HTTP.
    commands: list[dict] = []
    threading.Thread(target=_studio, args=(url, commands), daemon=True).start()
    deadline = time.time() + 5
    while not app.state.ui_bridge.has_listener("s1") and time.time() < deadline:
        await asyncio.sleep(0.02)
    ack = await port.send_command({"type": "set_view", "role": "test"}, "s1", 5.0)
    assert ack["identity"] == "w|test|0"
    assert commands[0]["type"] == "set_view"
    assert (await port.command_status(ack["id"]))["ok"] is True


def _studio(url: str, commands: list[dict]) -> None:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with httpx.stream(
        "GET", url + "/api/ui/events", params={"session": "s1"}, headers=headers, timeout=10
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                cmd = json.loads(line[6:])
                commands.append(cmd)
                httpx.post(
                    url + "/api/ui/ack",
                    json={"id": cmd["id"], "ok": True, "identity": "w|test|0"},
                    headers=headers,
                )
                return


async def test_stdio_server_end_to_end(home):
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "dtk_engine.agent.cli"],
        env={"DTK_HOME": str(home), "PATH": ""},
    )
    async with Client(params) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"list_keys", "propose_steps"} <= names
        out = await client.call_tool("propose_steps", {"ops": [{"remove": {"index": 0}}]})
        assert json.loads(out.content[0].text)["data"]["error"] == "no_studio"
