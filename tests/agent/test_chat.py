"""In-Studio agent chat: hub turns over the stub pack, HTTP routes, SSE frames."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from dtk_engine.agent.chat import AgentBusyError, AgentHub, NoAgentError, Pack
from dtk_engine.agent.packs.chat_packs import hub_from_env, tool_result_fields
from dtk_engine.agent.packs.stub import StubAdapter
from dtk_engine.agent.packs.stub import chat_pack as stub_pack
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio
TOKEN = "chat-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    for name in ("DTK_AGENT_PACK", "DTK_AGENT_MAX_TOKENS", "DTK_UI_REVIEW_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


class Studio:
    """A fake Studio tab: one listener; acks commands, collects agent events."""

    def __init__(self, bridge: UiBridge, session: str = "s1", ack: dict | None = None):
        self.bridge, self.session = bridge, session
        self.queue = bridge.add_listener(session)
        self.events: list[dict] = []
        self.commands: list[dict] = []
        self.ack = ack or {"ok": True, "identity": "ws|train|v1|abcd"}

    async def until_done(self, timeout: float = 10.0) -> list[dict]:
        async def pump():
            while True:
                item = await self.queue.get()
                if isinstance(item, tuple):
                    self.events.append(item[1])
                    if item[1]["type"] == "done":
                        return
                else:
                    self.commands.append(item)
                    self.bridge.ack({"id": item["id"], **self.ack})

        await asyncio.wait_for(pump(), timeout)
        events, self.events = self.events, []
        return events


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _hub(bridge: UiBridge, pack: Pack | None = None, **kw) -> AgentHub:
    return AgentHub(bridge, pack if pack is not None else stub_pack(), **kw)


# -- hub over the stub pack --------------------------------------------------


async def test_echo_turn_events_and_usage_totals():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(UiBridge("t"))
    hub.bridge = bridge
    assert hub.send("s1", "hello") == "t1"
    events = await studio.until_done()
    assert _types(events) == ["user_message", "assistant_delta", "usage", "done"]
    assert all(e["turn"] == "t1" for e in events)
    assert events[1]["text"] == "stub: hello"
    assert events[2] == {
        "type": "usage", "turn": "t1", "input_tokens": 10, "output_tokens": 5,
        "total_input_tokens": 10, "total_output_tokens": 5,
    }
    assert events[3]["stop_reason"] == "end_turn"
    assert hub.send("s1", "again") == "t2"
    usage = next(e for e in await studio.until_done() if e["type"] == "usage")
    assert (usage["total_input_tokens"], usage["total_output_tokens"]) == (20, 10)
    assert hub.status("s1")["usage"] == {"input_tokens": 20, "output_tokens": 10}


async def test_add_a_step_goes_through_the_real_bridge_with_context_fill():
    bridge = UiBridge("t")
    bridge.put_context({"session": "s1", "workspace": "parkinson", "identity": "id-1"})
    studio, hub = Studio(bridge), _hub(bridge)
    hub.send("s1", "please add a step")
    events = await studio.until_done()
    assert _types(events) == [
        "user_message", "tool_call", "tool_result", "assistant_delta", "usage", "done",
    ]
    (cmd,) = studio.commands
    assert cmd["type"] == "propose_steps"
    assert cmd["workspace"] == "parkinson" and cmd["base_identity"] == "id-1"
    assert cmd["ops"] == [{"add": {"step": {
        "op": "scale", "target": "both", "params": {"columns": ["age"]},
    }}}]
    call, result = events[1], events[2]
    assert call["name"] == "propose_steps" and result["id"] == call["id"]
    assert result["ok"] is True and result["command"] == cmd["id"]
    assert result["identity"] == "ws|train|v1|abcd"


async def test_review_pending_is_reported_on_the_tool_result():
    bridge = UiBridge("t")
    studio = Studio(bridge, ack={"pending": "review"})
    hub = _hub(bridge)
    hub.send("s1", "add a step")
    result = next(e for e in await studio.until_done() if e["type"] == "tool_result")
    assert result["ok"] is True and result["pending"] == "review" and result["command"]


async def test_drop_column_is_a_destructive_proposal_held_for_review():
    bridge = UiBridge("t")
    bridge.put_context({"session": "s1", "workspace": "parkinson", "identity": "id-1"})
    studio = Studio(bridge, ack={"pending": "review"})
    hub = _hub(bridge)
    hub.send("s1", "please drop time_since_intake_off")
    events = await studio.until_done()
    (cmd,) = studio.commands
    assert cmd["ops"] == [{"add": {"step": {
        "op": "drop_columns", "target": "both",
        "params": {"columns": ["time_since_intake_off"]},
    }}}]
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is True and result["pending"] == "review"
    assert result["command"] == cmd["id"]
    reply = next(e for e in events if e["type"] == "assistant_delta")
    assert reply["text"] == "stub: waiting for your review in Studio"
    assert bridge.command_status(cmd["id"]) == {"id": cmd["id"], "ok": None, "pending": "review"}


async def test_permission_denied_and_allowed():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge)

    async def answer(allow: bool):
        while True:
            item = await studio.queue.get()
            if isinstance(item, tuple):
                studio.events.append(item[1])
                if item[1]["type"] == "permission_request":
                    assert hub.reply("s1", item[1]["id"], allow)
                    assert not hub.reply("s1", item[1]["id"], allow)  # answered once
                    return
            else:
                studio.commands.append(item)

    hub.send("s1", "permission please")
    await asyncio.wait_for(answer(False), 5)
    events = await studio.until_done()
    assert _types(events) == [
        "user_message", "tool_call", "permission_request", "tool_result",
        "assistant_delta", "usage", "done",
    ]
    assert events[3] == {**events[3], "ok": False, "error": "denied", "id": events[1]["id"]}
    assert events[2]["tool"] == "propose_steps" and events[2]["summary"]
    assert studio.commands == []

    hub.send("s1", "permission please")
    await asyncio.wait_for(answer(True), 5)
    events = await studio.until_done()
    assert next(e for e in events if e["type"] == "tool_result")["ok"] is True
    assert len(studio.commands) == 1


async def test_permission_times_out_as_denied(monkeypatch):
    monkeypatch.setenv("DTK_UI_REVIEW_TIMEOUT", "0.05")
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge)
    hub.send("s1", "permission")
    events = await studio.until_done()
    assert next(e for e in events if e["type"] == "tool_result")["error"] == "denied"


class SlowAdapter(StubAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def send(self, text: str) -> str | None:
        await self.cancelled.wait()
        return "interrupted"

    async def cancel(self) -> None:
        self.cancelled.set()


def _pack(create) -> Pack:
    return Pack("test", lambda: "p", lambda: "m", lambda: None, create)


async def test_busy_then_cancel():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge, _pack(SlowAdapter))
    hub.send("s1", "long")
    await asyncio.sleep(0.01)
    with pytest.raises(AgentBusyError):
        hub.send("s1", "second")
    assert hub.status("s1")["running"] is True
    assert await hub.cancel("s1") is True
    events = await studio.until_done()
    assert events[-1]["stop_reason"] == "cancelled"
    assert await hub.cancel("s1") is False
    assert hub.status("s1")["running"] is False


class StuckAdapter(StubAdapter):
    async def send(self, text: str) -> str | None:
        await asyncio.Event().wait()
        return None


async def test_cancel_kills_a_stuck_turn(monkeypatch):
    monkeypatch.setattr("dtk_engine.agent.chat.CANCEL_GRACE", 0.05)
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge, _pack(StuckAdapter))
    hub.send("s1", "x")
    await asyncio.sleep(0.01)
    assert await hub.cancel("s1")
    assert (await studio.until_done())[-1]["stop_reason"] == "cancelled"


class BrokenAdapter(StubAdapter):
    async def send(self, text: str) -> str | None:
        raise RuntimeError("cli exploded")


async def test_pack_error_ends_the_turn():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge, _pack(BrokenAdapter))
    hub.send("s1", "x")
    events = await studio.until_done()
    assert _types(events) == ["user_message", "error", "usage", "done"]
    assert events[1] == {**events[1], "message": "cli exploded", "code": "pack_error"}
    assert events[-1]["stop_reason"] == "error"


async def test_max_tokens_cap(monkeypatch):
    monkeypatch.setenv("DTK_AGENT_MAX_TOKENS", "12")
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge)
    assert hub.status(None)["max_tokens"] == 12
    hub.send("s1", "one")
    events = await studio.until_done()
    assert _types(events)[-3:] == ["error", "usage", "done"]
    assert events[-3]["code"] == "max_tokens" and events[-1]["stop_reason"] == "max_tokens"
    hub.send("s1", "two")  # over the cap: no adapter call at all
    events = await studio.until_done()
    assert _types(events) == ["user_message", "error", "usage", "done"]
    assert events[2]["input_tokens"] == 0


async def test_sessions_are_separate_and_close():
    bridge = UiBridge("t")
    a, b = Studio(bridge, "a"), Studio(bridge, "b")
    hub = _hub(bridge)
    hub.send("a", "x")
    hub.send("b", "y")
    assert (await a.until_done())[1]["text"] == "stub: x"
    assert (await b.until_done())[1]["text"] == "stub: y"
    await hub.close()
    assert hub.status("a")["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_no_pack_and_env_selection(monkeypatch):
    bridge = UiBridge("t")
    hub = hub_from_env(bridge)
    status = hub.status(None)
    assert status["available"] is False and status["pack"] is None
    assert "DTK_AGENT_PACK" in status["reason"]
    with pytest.raises(NoAgentError):
        hub.send("s1", "x")
    monkeypatch.setenv("DTK_AGENT_PACK", "nope")
    assert "unknown DTK_AGENT_PACK" in hub_from_env(bridge).status(None)["reason"]
    monkeypatch.setenv("DTK_AGENT_PACK", "stub")
    status = hub_from_env(bridge).status(None)
    assert status["available"] is True and status["pack"] == "stub"
    assert status["model"] == "stub" and status["max_tokens"] is None
    monkeypatch.setenv("DTK_AGENT_PACK", "agent-sdk")
    assert hub_from_env(bridge).status(None)["pack"] == "agent-sdk"


def test_session_tools_are_the_tools_taking_session():
    hub = _hub(UiBridge("t"))
    assert {"propose_steps", "run_key", "get_ui_context"} <= hub.session_tools
    assert "list_keys" not in hub.session_tools


def test_tool_result_fields():
    assert tool_result_fields({"type": "X", "message": "bad"}, is_error=True) == {
        "ok": False, "error": "bad",
    }
    ack = {"identity": "i2", "data": {"id": "c1", "ok": False, "error": "stale"}}
    assert tool_result_fields(ack, is_error=False) == {
        "ok": False, "error": "stale", "command": "c1", "identity": "i2",
    }
    big = tool_result_fields({"identity": None, "data": {"rows": ["x" * 500]}}, is_error=False)
    assert big["ok"] and len(big["summary"]) == 200 and "identity" not in big
    assert tool_result_fields("plain", is_error=False) == {"ok": True, "summary": "plain"}


@contextlib.contextmanager
def _serve():
    """The app on a real socket (SSE needs one); yields its base URL."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(
        create_app(), host="127.0.0.1", port=port, log_level="error",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _read_until(lines, predicate):
    for line in lines:
        if predicate(line):
            return line
    raise AssertionError("stream ended")


# -- HTTP routes -------------------------------------------------------------


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setenv("DTK_AGENT_PACK", "stub")
    with TestClient(create_app(), base_url="http://localhost") as c:
        yield c


def test_status_route_and_guard(stub_client):
    assert stub_client.get("/api/ui/agent").status_code == 401
    body = stub_client.get("/api/ui/agent", headers=AUTH).json()
    assert body["available"] is True and body["pack"] == "stub"
    assert body["running"] is False and body["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_send_permission_and_cancel_routes(stub_client):
    sent = stub_client.post("/api/ui/agent/send", json={"session": "s", "text": "hi"}, headers=AUTH)
    assert sent.status_code == 202 and sent.json() == {"turn": "t1"}
    bad = stub_client.post(
        "/api/ui/agent/permission", json={"session": "s", "id": "p9", "allow": True},
        headers=AUTH,
    )
    assert bad.status_code == 404 and bad.json()["type"] == "UnknownPermission"
    cancel = stub_client.post("/api/ui/agent/cancel", json={"session": "zz"}, headers=AUTH)
    assert cancel.json() == {"cancelled": False}
    assert stub_client.post("/api/ui/agent/send", json={"session": "s"}, headers=AUTH).status_code == 422


def test_send_without_pack_is_503():
    with TestClient(create_app(), base_url="http://localhost") as client:
        resp = client.post("/api/ui/agent/send", json={"session": "s", "text": "x"}, headers=AUTH)
        assert resp.status_code == 503
        assert resp.json()["type"] == "NoAgent" and "DTK_AGENT_PACK" in resp.json()["message"]
        assert client.get("/api/ui/agent", headers=AUTH).json()["available"] is False


def test_agent_events_on_the_sse_stream(monkeypatch):
    monkeypatch.setenv("DTK_AGENT_PACK", "stub")
    with _serve() as url, httpx.Client(timeout=5) as client:
        stream = client.stream("GET", f"{url}/api/ui/events", params={"session": "s1"}, headers=AUTH)
        with stream as resp:
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            threading.Thread(target=lambda: httpx.post(
                f"{url}/api/ui/agent/send", json={"session": "s1", "text": "hey"},
                headers=AUTH,
            )).start()
            seen = []
            while not seen or seen[-1]["type"] != "done":
                _read_until(lines, lambda line: line == "event: agent")
                data = _read_until(lines, lambda line: line.startswith("data: "))
                seen.append(json.loads(data[6:]))
    assert _types(seen) == ["user_message", "assistant_delta", "usage", "done"]
