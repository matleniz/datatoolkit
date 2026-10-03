"""Terminal packs: the WebSocket guard, a fake CLI on a real PTY, clean stops."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dtk_engine.agent import options, terminal
from dtk_engine.http import create_app, main

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX PTY only")

TOKEN = "term-token"
ORIGIN = {"origin": "http://testserver"}
FAKE_CLI = """\
import json, os, signal, sys
def size(*_):
    c, r = os.get_terminal_size(0)
    print(f"size:{c}x{r}", flush=True)
signal.signal(signal.SIGWINCH, size)
print("argv:" + json.dumps(sys.argv[1:]), flush=True)
print(f"tty:{os.isatty(0)} ctty:{os.getsid(0) == os.getpid()}", flush=True)
size()
print(f"pid:{os.getpid()}", flush=True)
for line in sys.stdin:
    if line.strip() == "exit":
        sys.exit(3)
    print("got:" + line.strip(), flush=True)
"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    monkeypatch.setenv("DTK_AGENT_TERMINAL", "1")
    monkeypatch.setenv("DTK_UI_ALLOWED_HOSTS", "testserver")  # TestClient's socket Host
    monkeypatch.delenv("DTK_AGENT_PACK", raising=False)


@pytest.fixture
def fake_cli(tmp_path) -> Path:
    path = tmp_path / "fake_cli.py"
    path.write_text(FAKE_CLI)
    return path


@pytest.fixture
def client(fake_cli):
    app = create_app()

    def argv_for(pack, directory, model):
        return [sys.executable, str(fake_cli), pack.id, str(directory), *([model] if model else [])]

    app.state.terminals.argv_for = argv_for
    with TestClient(app) as c:
        yield c


def _url(pack: str = "claude-code", **params: str) -> str:
    query = {"session": "s1", "pack": pack, "token": TOKEN, **params}
    return "/api/ui/terminal?" + "&".join(f"{k}={v}" for k, v in query.items())


def _until(ws, marker: str) -> str:
    """PTY output until ``marker`` shows up (control frames are skipped)."""
    out = ""
    while marker not in out:
        message = ws.receive()
        if message.get("bytes") is not None:
            out += message["bytes"].decode(errors="replace")
        elif message["type"] == "websocket.close":
            raise AssertionError(f"closed before {marker!r}: {out!r}")
    return out


def _pid(ws) -> int:
    out = _until(ws, "pid:")
    while not re.search(r"pid:\d+\s", out):
        out += _until(ws, "\n")
    return int(re.search(r"pid:(\d+)", out).group(1))


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _close_code(client, url: str, headers: dict | None = ORIGIN) -> int:
    headers = dict(headers or {})  # TestClient adds to the dict
    with client.websocket_connect(url, headers=headers) as ws, pytest.raises(
        WebSocketDisconnect
    ) as info:
        ws.receive_text()
    return info.value.code


@pytest.mark.parametrize(
    ("url", "headers", "code"),
    [
        (_url(token="wrong"), ORIGIN, 4401),
        (_url(), None, 4403),  # Origin is required on the socket
        (_url(), {"origin": "http://evil.example"}, 4403),
        (_url("stub"), ORIGIN, 4404),
        (_url("nope"), ORIGIN, 4404),
        (_url(model="--dangerously-skip-permissions"), ORIGIN, 4422),
        (_url(cols="0"), ORIGIN, 4422),
        (_url(session=""), ORIGIN, 4422),
    ],
)
def test_refusals_close_with_contract_codes(client, url, headers, code):
    assert _close_code(client, url, headers) == code


def test_terminal_off_is_refused(client, monkeypatch):
    monkeypatch.delenv("DTK_AGENT_TERMINAL")
    assert _close_code(client, _url()) == 4403


def test_bad_host_is_refused(client):
    assert _close_code(client, _url(), {**ORIGIN, "host": "evil.example"}) == 4403


def test_cli_on_a_pty_echo_resize_and_exit(client, tmp_path):
    with client.websocket_connect(_url(model="sonnet", cols="100", rows="30"), headers=dict(ORIGIN)) as ws:
        started = json.loads(ws.receive_text())
        assert started["type"] == "started"
        assert started["pack"] == "claude-code" and started["model"] == "sonnet"
        directory = tmp_path / "home" / "agent" / "terminal" / "claude-code"
        assert started["command"][-3:] == ["claude-code", str(directory), "sonnet"]
        out = _until(ws, "pid:")
        assert "tty:True ctty:True" in out and "size:100x30" in out
        assert (directory / "dtk.mcp.json").is_file()  # the pack's dtk-only config
        ws.send_bytes(b"hello\n")
        assert "got:hello" in _until(ws, "got:hello")
        ws.send_text(json.dumps({"type": "resize", "cols": 80, "rows": 24}))
        assert "size:80x24" in _until(ws, "size:80x24")
        ws.send_text("not json")  # ignored
        ws.send_bytes(b"exit\n")
        message = ws.receive()
        while message.get("bytes") is not None:
            message = ws.receive()
        assert json.loads(message["text"]) == {"type": "exit", "code": 3}
        with pytest.raises(WebSocketDisconnect) as info:
            ws.receive_text()
        assert info.value.code == 1000
    assert len(client.app.state.terminals) == 0


def test_closing_the_socket_stops_the_cli(client):
    with client.websocket_connect(_url(), headers=dict(ORIGIN)) as ws:
        ws.receive_text()
        pid = _pid(ws)
    assert _gone(pid)
    deadline = time.monotonic() + 5
    while len(client.app.state.terminals) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(client.app.state.terminals) == 0


def test_one_terminal_per_session_and_pack(client):
    with client.websocket_connect(_url(), headers=dict(ORIGIN)) as ws:
        ws.receive_text()
        _until(ws, "pid:")
        assert _close_code(client, _url()) == 4409
        with client.websocket_connect(_url("gemini"), headers=dict(ORIGIN)) as other:
            assert json.loads(other.receive_text())["pack"] == "gemini"


def test_missing_cli_sends_error_then_1011(client):
    client.app.state.terminals.argv_for = lambda pack, d, m: ["dtk-no-such-cli"]
    with client.websocket_connect(_url(), headers=dict(ORIGIN)) as ws:
        error = json.loads(ws.receive_text())
        assert error["type"] == "error" and "dtk-no-such-cli" in error["message"]
        with pytest.raises(WebSocketDisconnect) as info:
            ws.receive_text()
    assert info.value.code == 1011


@pytest.mark.anyio
async def test_shutdown_stops_every_cli(fake_cli, tmp_path):
    terminals = terminal.Terminals(lambda pack, d, m: [sys.executable, str(fake_cli)])
    one = terminals.open("s1", "claude-code", None, (80, 24))
    two = terminals.open("s2", "opencode", None, (80, 24))
    pids = [one.proc.pid, two.proc.pid]
    await terminals.close()
    assert len(terminals) == 0
    assert all(_gone(pid) for pid in pids)


@pytest.mark.anyio
async def test_stop_escalates_to_sigkill(tmp_path, monkeypatch):
    stubborn = tmp_path / "stubborn.py"
    stubborn.write_text(
        "import signal, time\n"
        "for s in (signal.SIGHUP, signal.SIGTERM):\n"
        "    signal.signal(s, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setattr(terminal, "HUP_GRACE", 0.2)
    monkeypatch.setattr(terminal, "TERM_GRACE", 0.3)
    term = terminal.Terminal([sys.executable, str(stubborn)], tmp_path, (80, 24))
    deadline = time.monotonic() + 5
    out = b""
    while b"ready" not in out and time.monotonic() < deadline:
        out += term.read() or b""
        await asyncio.sleep(0.02)
    assert await term.stop() == -9
    assert term.fd is None


def test_real_pack_argv_has_the_model_flag(tmp_path):
    directory = tmp_path / "cfg"
    claude = terminal.build_argv(terminal.terminal_pack("claude-code"), directory, "opus")
    assert claude[0] == "claude" and claude[-2:] == ["--model", "opus"]
    assert "--strict-mcp-config" in claude
    gemini = terminal.build_argv(terminal.terminal_pack("gemini"), directory, "gemini-3-pro")
    assert gemini[-2:] == ["-m", "gemini-3-pro"]
    opencode = terminal.build_argv(terminal.terminal_pack("opencode"), directory, None)
    assert opencode == ["opencode"]


def test_dtk_api_terminal_flag_sets_the_env(monkeypatch):
    monkeypatch.delenv("DTK_AGENT_TERMINAL")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    main(["--terminal"])
    assert options.terminal_enabled()
