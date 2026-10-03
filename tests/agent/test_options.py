"""Agent options and per-session selection: hub configure, routes, discovery parsing."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from test_chat import AUTH, Studio

from dtk_engine.agent.chat import AgentBusyError, AgentHub, ConfigError, NoAgentError
from dtk_engine.agent.packs import agent_sdk
from dtk_engine.agent.packs.stub import StubAdapter
from dtk_engine.agent.packs.stub import chat_pack as stub_pack
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UI_TOKEN", "chat-token")
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    for name in ("DTK_AGENT_PACK", "DTK_AGENT_TERMINAL", "DTK_AGENT_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def _hub(bridge: UiBridge) -> AgentHub:
    return AgentHub(bridge, stub_pack())


def _entry(options: dict, pack_id: str) -> dict:
    return next(p for p in options["packs"] if p["id"] == pack_id)


async def test_options_list_stub_models_and_terminal_off():
    options = await _hub(UiBridge("t")).options()
    assert options["default"] == {"pack": "stub", "model": None}
    stub = _entry(options, "stub")
    assert [m["id"] for m in stub["models"]] == ["stub-small", "stub-large"]
    assert stub["mode"] == "test" and stub["panel"] == "chat" and stub["available"] is True
    assert stub["model_free_text"] is False and stub["default_model"] is None
    for pack_id in ("claude-code", "gemini", "opencode"):
        entry = _entry(options, pack_id)
        assert entry["panel"] == "terminal" and entry["mode"] == "cli"
        assert entry["available"] is False
        assert entry["reason"] == "terminal off: set DTK_AGENT_TERMINAL=1"


async def test_options_terminal_on_follows_path(monkeypatch):
    monkeypatch.setenv("DTK_AGENT_TERMINAL", "1")
    monkeypatch.setenv("PATH", "/nonexistent")
    entry = _entry(await _hub(UiBridge("t")).options(), "gemini")
    assert entry["available"] is False and "not found on PATH" in entry["reason"]
    assert entry["model_free_text"] is True and entry["models"] == []


async def test_discovery_is_cached_until_refresh():
    calls = []

    async def lister():
        calls.append(1)
        from dtk_engine.agent.models import ModelList

        return ModelList([{"id": "m1"}])

    hub = _hub(UiBridge("t"))
    await asyncio.gather(hub.models_of(lister), hub.models_of(lister))
    await hub.models_of(lister)
    assert len(calls) == 1
    await hub.options(refresh=True)
    await hub.models_of(lister)
    assert len(calls) == 2


async def test_configure_model_keeps_conversation_and_totals():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge)
    hub.send("s1", "hello")
    assert (await studio.until_done())[1]["text"] == "stub: hello"
    adapter = hub._chats["s1"].adapter
    status = await hub.configure("s1", "stub", "stub-large")
    assert status["model"] == "stub-large" and status["mode"] == "test"
    event = studio.queue.get_nowait()[1]
    assert event == {"type": "config", "turn": None, "pack": "stub", "model": "stub-large",
                     "reset": False}
    assert hub._chats["s1"].adapter is adapter
    hub.send("s1", "again")
    assert (await studio.until_done())[1]["text"] == "stub[stub-large]: again"
    assert hub.status("s1")["usage"] == {"input_tokens": 20, "output_tokens": 10}


async def test_configure_other_pack_resets_and_keeps_usage():
    bridge = UiBridge("t")
    studio = Studio(bridge)
    other = stub_pack()
    other = type(other)(**{**other.__dict__, "id": "stub2", "create": StubAdapter})
    hub = AgentHub(bridge, stub_pack(), packs={"stub": stub_pack(), "stub2": other})
    hub.send("s1", "one")
    await studio.until_done()
    await hub.configure("s1", "stub2", None)
    assert studio.queue.get_nowait()[1]["reset"] is True
    assert hub._chats["s1"].adapter is None
    assert hub.status("s1")["pack"] == "stub2"
    assert hub.status("s1")["usage"]["input_tokens"] == 10
    assert hub.status("other")["pack"] == "stub"  # the choice is per session


async def test_configure_errors():
    bridge = UiBridge("t")
    studio, hub = Studio(bridge), _hub(bridge)
    for pack, model, code in (
        ("nope", None, "UnknownPack"),
        ("stub", "gpt-x", "UnknownModel"),
        ("gemini", None, "WrongPanel"),
    ):
        with pytest.raises(ConfigError) as err:
            await hub.configure("s1", pack, model)
        assert err.value.code == code
    hub.send("s1", "hi")
    with pytest.raises(AgentBusyError):
        await hub.configure("s1", "stub", None)
    await studio.until_done()
    with pytest.raises(NoAgentError):
        await AgentHub(bridge, None).configure("s1", "stub", None)


def test_routes(monkeypatch):
    monkeypatch.setenv("DTK_AGENT_PACK", "stub")
    with TestClient(create_app(), base_url="http://localhost") as client:
        assert client.get("/api/ui/agent/options").status_code == 401
        options = client.get("/api/ui/agent/options?refresh=1", headers=AUTH).json()
        assert options["default"]["pack"] == "stub"
        ok = client.post("/api/ui/agent/config", headers=AUTH,
                         json={"session": "s", "pack": "stub", "model": "stub-small"})
        assert ok.status_code == 200 and ok.json()["model"] == "stub-small"
        assert client.get("/api/ui/agent?session=s", headers=AUTH).json()["model"] == "stub-small"
        for pack, model, kind in (("zz", None, "UnknownPack"), ("stub", "q", "UnknownModel"),
                                  ("claude-code", None, "WrongPanel")):
            bad = client.post("/api/ui/agent/config", headers=AUTH,
                              json={"session": "s", "pack": pack, "model": model})
            assert bad.status_code == 422 and bad.json()["type"] == kind


def test_routes_agent_off():
    with TestClient(create_app(), base_url="http://localhost") as client:
        options = client.get("/api/ui/agent/options", headers=AUTH).json()
        assert options["available"] is False and "DTK_AGENT_PACK" in options["reason"]
        bad = client.post("/api/ui/agent/config", headers=AUTH,
                          json={"session": "s", "pack": "stub"})
        assert bad.status_code == 503 and bad.json()["type"] == "NoAgent"


def test_claude_initialize_models_are_mapped():
    raw = [
        {"value": "default", "displayName": "Default", "description": "d",
         "resolvedModel": "claude-sonnet-5-5"},
        {"value": "opus"},
        {"displayName": "no value"},
    ]
    assert agent_sdk._model_entries(raw) == [
        {"id": "default", "label": "Default", "description": "d", "resolved": "claude-sonnet-5-5"},
        {"id": "opus", "label": "opus", "description": "", "resolved": None},
    ]


async def test_discovery_failure_is_free_text(monkeypatch):
    async def boom():
        raise RuntimeError("no cli")

    monkeypatch.setattr(agent_sdk, "_initialize_models", boom)
    listing = await agent_sdk.discover_models()
    assert listing.free_text and listing.models == [] and "no cli" in listing.error
    assert listing.to_json()["models_error"]
