"""agent-sdk pack without a CLI: detection, options, permission gate, event translation."""

from __future__ import annotations

import asyncio
import json
import stat

import pytest

claude_agent_sdk = pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from dtk_engine.agent.chat import AgentHub, ChatSession
from dtk_engine.agent.commands import UI_COMMANDS
from dtk_engine.agent.packs import agent_sdk
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    for name in (
        "ANTHROPIC_API_KEY", "DTK_AGENT_CLI", "DTK_AGENT_MODEL", "DTK_AGENT_MAX_TURNS",
        "DTK_AGENT_MAX_TOKENS", "DTK_AGENT_COMPACT_AT", *agent_sdk._PROVIDER_ENVS,
    ):
        monkeypatch.delenv(name, raising=False)
    agent_sdk._cli_auth.cache_clear()
    yield
    agent_sdk._cli_auth.cache_clear()


def _fake_cli(tmp_path, status: dict | str) -> str:
    path = tmp_path / "claude"
    body = status if isinstance(status, str) else json.dumps(status)
    path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{body}\nEOF\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


# -- detection ---------------------------------------------------------------


def test_api_key_needs_no_cli(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("DTK_AGENT_CLI", "/nonexistent/claude")
    assert agent_sdk.detect() is None
    assert agent_sdk.provider() == "Anthropic API (ANTHROPIC_API_KEY)"


def test_cli_login_is_used(tmp_path, monkeypatch):
    status = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "pro",
              "email": "someone@example.com"}
    monkeypatch.setenv("DTK_AGENT_CLI", _fake_cli(tmp_path, status))
    assert agent_sdk.detect() is None
    provider = agent_sdk.provider()
    assert provider == "Claude (claude.ai login, pro, via the claude CLI)"
    assert "example.com" not in provider


def test_cli_logged_out(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_AGENT_CLI", _fake_cli(tmp_path, {"loggedIn": False}))
    assert "not logged in" in agent_sdk.detect()


def test_cli_without_auth_status_lets_the_first_turn_tell(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_AGENT_CLI", _fake_cli(tmp_path, "unknown command"))
    assert agent_sdk.detect() is None


def test_no_cli_anywhere(monkeypatch):
    monkeypatch.setattr(agent_sdk.shutil, "which", lambda _name: None)
    monkeypatch.setattr(agent_sdk, "_bundled_cli", lambda: None)
    assert "claude CLI not found" in agent_sdk.detect()


def test_model_and_turns_from_env(monkeypatch):
    assert agent_sdk.model() is None and agent_sdk._max_turns() == agent_sdk.DEFAULT_MAX_TURNS
    monkeypatch.setenv("DTK_AGENT_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("DTK_AGENT_MAX_TURNS", "3")
    assert agent_sdk.model() == "claude-sonnet-5-5" and agent_sdk._max_turns() == 3


# -- adapter with a fake client ------------------------------------------------


class FakeClient:
    def __init__(self, options) -> None:
        self.options = options
        self.script: list = []
        self.queries: list[str] = []
        self.interrupted = 0
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def query(self, text: str) -> None:
        self.queries.append(text)

    async def receive_response(self):
        for message in self.script:
            await asyncio.sleep(0)
            yield message

    async def interrupt(self) -> None:
        self.interrupted += 1

    async def disconnect(self) -> None:
        self.connected = False


class Listener:
    def __init__(self, bridge: UiBridge, session: str = "s1") -> None:
        self.queue = bridge.add_listener(session)

    def events(self) -> list[dict]:
        out = []
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if isinstance(item, tuple):
                out.append(item[1])
        return out


@pytest.fixture
def fake(monkeypatch):
    clients: list[FakeClient] = []

    def make(options):
        clients.append(FakeClient(options))
        return clients[-1]

    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", make)
    return clients


async def _started(fake, bridge: UiBridge, **hub_kw):
    hub = AgentHub(bridge, agent_sdk.chat_pack(), **hub_kw)
    adapter = agent_sdk.SdkAdapter()
    chat = ChatSession(hub, "s1")
    await adapter.start(chat)
    chat.adapter = adapter
    return hub, chat, adapter, fake[-1]


async def test_options_lock_the_agent_down(fake, tmp_path):
    _, _, _, client = await _started(fake, UiBridge("t"))
    options = client.options
    assert client.connected
    assert options.tools == [] and options.allowed_tools == []
    assert options.setting_sources == [] and options.strict_mcp_config is True
    assert list(options.mcp_servers) == ["dtk"]
    server = options.mcp_servers["dtk"]
    assert server["type"] == "sdk" and server["instance"] is not None
    assert options.include_partial_messages is True
    assert options.permission_mode == "default"
    assert str(tmp_path / "home") in str(options.cwd)
    assert "no shell" in options.system_prompt


async def test_can_use_tool_allows_dtk_only_and_pins_session(fake):
    _, _, adapter, _ = await _started(fake, UiBridge("t"))
    denied = await adapter._can_use_tool("Bash", {"command": "ls"}, None)
    assert isinstance(denied, PermissionResultDeny)
    denied = await adapter._can_use_tool("mcp__other__x", {}, None)
    assert isinstance(denied, PermissionResultDeny)
    allowed = await adapter._can_use_tool(
        "mcp__dtk__propose_steps", {"ops": [], "session": "other-tab"}, None
    )
    assert isinstance(allowed, PermissionResultAllow)
    assert allowed.updated_input == {"ops": [], "session": "s1"}
    allowed = await adapter._can_use_tool("mcp__dtk__list_keys", {}, None)
    assert allowed.updated_input == {}


async def test_every_ui_command_tool_is_allowed_with_session_pinned(fake):
    _, _, adapter, client = await _started(fake, UiBridge("t"))
    for spec in UI_COMMANDS.values():
        allowed = await adapter._can_use_tool(f"mcp__dtk__{spec.tool_name}", {"x": 1}, None)
        assert isinstance(allowed, PermissionResultAllow), spec.tool_name
        assert allowed.updated_input == {"x": 1, "session": "s1"}
        assert spec.tool_name in client.options.system_prompt


def _stream(event: dict) -> StreamEvent:
    return StreamEvent(uuid="u", session_id="x", event=event)


def _result(**kw) -> ResultMessage:
    base = {"subtype": "success", "duration_ms": 1, "duration_api_ms": 1, "is_error": False,
            "num_turns": 1, "session_id": "x"}
    return ResultMessage(**{**base, **kw})


@pytest.mark.parametrize(
    "value, window", [(None, "60000"), ("40000", "40000"), ("off", None), ("junk", "60000")]
)
async def test_compaction_window_reaches_the_cli(fake, monkeypatch, value, window):
    """#151: the CLI auto-compacts at DTK_AGENT_COMPACT_AT (default 60k tokens)."""
    if value is not None:
        monkeypatch.setenv("DTK_AGENT_COMPACT_AT", value)
    _, _, _, client = await _started(fake, UiBridge("t"))
    assert client.options.env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == window


async def test_compaction_is_announced(fake):
    from claude_agent_sdk import SystemMessage

    bridge = UiBridge("t")
    listener = Listener(bridge)
    _, chat, adapter, client = await _started(fake, bridge)
    chat.turn = "t1"
    meta = {"trigger": "auto", "pre_tokens": 61234}
    client.script = [
        SystemMessage(subtype="compact_boundary", data={"compact_metadata": meta}),
        SystemMessage(subtype="init", data={}),
        _result(usage={"input_tokens": 1, "output_tokens": 1}, stop_reason="end_turn"),
    ]
    await adapter.send("hello")
    events = [e for e in listener.events() if e["type"] == "compacted"]
    assert events == [{"type": "compacted", "turn": "t1", "trigger": "auto", "pre_tokens": 61234}]


async def test_turn_translation(fake):
    bridge = UiBridge("t")
    listener = Listener(bridge)
    _, chat, adapter, client = await _started(fake, bridge)
    ack = {"identity": "i9", "data": {"id": "c3", "ok": None, "pending": "review"}}
    usage = {"input_tokens": 100, "cache_read_input_tokens": 50, "output_tokens": 7}
    client.script = [
        _stream({"type": "message_start", "message": {"id": "m1"}}),
        _stream({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
        AssistantMessage(
            content=[TextBlock("Hi"), ToolUseBlock("tu1", "mcp__dtk__propose_steps", {"ops": []})],
            model="m", message_id="m1", usage=usage,
        ),
        UserMessage(content=[ToolResultBlock("tu1", [{"type": "text", "text": json.dumps(ack)}])]),
        AssistantMessage(content=[TextBlock("Review it in Studio.")], model="m", message_id="m2"),
        _result(usage={"input_tokens": 300, "output_tokens": 20}, stop_reason="end_turn"),
    ]
    stop = await adapter.send("go")
    assert client.queries == ["go"] and stop == "end_turn"
    events = listener.events()
    assert [e["type"] for e in events] == [
        "assistant_delta", "tool_call", "tool_result", "assistant_delta",
    ]
    assert events[0]["text"] == "Hi"  # streamed once, the full TextBlock not repeated
    assert events[1] == {**events[1], "id": "tu1", "name": "propose_steps"}
    assert events[2] == {**events[2], "id": "tu1", "ok": True, "pending": "review",
                         "command": "c3", "identity": "i9"}
    assert events[3]["text"] == "Review it in Studio."  # not streamed: from the block
    assert chat.turn_usage == {"input_tokens": 300, "output_tokens": 20, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert chat.context_tokens == 150  # last call: 100 uncached + 50 cache reads


async def test_tool_error_and_result_error(fake):
    bridge = UiBridge("t")
    listener = Listener(bridge)
    _, _, adapter, client = await _started(fake, bridge)
    client.script = [
        UserMessage(content=[ToolResultBlock("tu2", "boom", is_error=True)]),
        _result(is_error=True, subtype="error_during_execution", errors=["CLI died"]),
    ]
    assert await adapter.send("x") == "error"
    events = listener.events()
    assert events[0] == {**events[0], "ok": False, "error": "boom"}
    assert events[1] == {**events[1], "type": "error", "message": "CLI died", "code": "pack_error"}


async def test_max_turns_stop_reason(fake):
    _, _, adapter, client = await _started(fake, UiBridge("t"))
    client.script = [_result(subtype="error_max_turns")]
    assert await adapter.send("x") == "max_turns"


async def test_cap_interrupts_mid_turn(fake):
    bridge = UiBridge("t")
    _, chat, adapter, client = await _started(fake, bridge, max_tokens=100)
    client.script = [
        AssistantMessage(content=[TextBlock("a")], model="m", message_id="m1",
                         usage={"input_tokens": 90, "output_tokens": 5}),
        AssistantMessage(content=[TextBlock("b")], model="m", message_id="m1",
                         usage={"input_tokens": 90, "output_tokens": 5}),  # same message
        AssistantMessage(content=[TextBlock("c")], model="m", message_id="m2",
                         usage={"input_tokens": 10, "output_tokens": 1}),
        _result(is_error=True, subtype="error_during_execution"),
    ]
    assert await adapter.send("x") == "max_tokens"
    assert client.interrupted == 1
    assert chat.over_cap()


async def test_cancel_and_close(fake):
    _, _, adapter, client = await _started(fake, UiBridge("t"))
    await adapter.cancel()
    assert client.interrupted == 1
    await adapter.close()
    assert not client.connected and adapter.client is None
