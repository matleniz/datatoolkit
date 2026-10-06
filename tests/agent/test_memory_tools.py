"""Agent memory through the tools, the UI commands and the conversation intro (#179)."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from dtk_engine import contract
from dtk_engine.agent.chat import AgentHub
from dtk_engine.agent.digest import intro_note
from dtk_engine.agent.packs.stub import chat_pack
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio
MEMORY = [
    {"id": "m1", "text": "ledd is in mg/day, 0 means untreated", "kind": "fact"},
    {"id": "m2", "text": "the user wants median imputation", "kind": "preference"},
]
CONTEXT = {"session": "s1", "workspace": "demo", "role": "train", "version": 0,
           "latest": 0, "identity": "demo|train|v0|x"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    for name in ("DTK_UPLOAD_DIR", "DTK_AGENT_LOG", "DTK_AGENT_MAX_CHARS"):
        monkeypatch.delenv(name, raising=False)
    contract.save_workspace({
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "memory": MEMORY,
    })


@pytest.fixture
def bridge():
    b = UiBridge("t")
    b.put_context(CONTEXT)
    return b


@pytest.fixture
async def client(bridge):
    async with Client(build_server(LocalUiPort(bridge), AuditLog())) as c:
        yield c


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


async def test_get_memory(client):
    out = _payload(await client.call_tool("get_memory", {}))["data"]
    assert [e["id"] for e in out["entries"]] == ["m1", "m2"]
    assert out["chars"] == sum(len(e["text"]) for e in MEMORY) and out["max_chars"] == 8000
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert tools["get_memory"].annotations.read_only_hint is True


@pytest.mark.parametrize(("tool", "args"), [
    ("remember", {"text": "fare is in pounds", "kind": "fact"}),
    ("forget", {"memory_id": "m1"}),
])
async def test_memory_commands_are_relayed_with_the_workspace(client, bridge, tool, args):
    queue = bridge.add_listener("s1")
    call = asyncio.create_task(client.call_tool(tool, args))
    cmd = await asyncio.wait_for(queue.get(), 5)
    bridge.ack({"id": cmd["id"], "ok": True, "memory_id": "m3"})
    out = _payload(await call)["data"]
    assert out["ok"] is True
    assert cmd == {"id": cmd["id"], "type": tool, **args, "workspace": "demo"}


def test_intro_note_lists_memory_after_documents():
    note = intro_note([{"id": "d1", "name": "dict.md"}], MEMORY)
    assert note.index("[Workspace documents") < note.index("[Workspace memory")
    assert "- m1 (fact) ledd is in mg/day, 0 means untreated\n" in note
    assert note.endswith("- m2 (preference) the user wants median imputation]\n\n")
    assert intro_note([], []) == ""


async def test_memory_once_per_conversation(bridge):
    hub = AgentHub(bridge, chat_pack())
    chat = hub._chat("s1")
    first = await hub._workspace_note(chat)
    assert first.startswith("[Studio: ") and "[Workspace memory" in first and "m2 (preference)" in first
    assert "Workspace memory" not in await hub._workspace_note(chat)
    chat.introduced = set()  # a new adapter = a new conversation
    assert "Workspace memory" in await hub._workspace_note(chat)


async def test_new_conversation_gets_the_memory_again(bridge, monkeypatch):
    """Session 1 learns a fact; a fresh adapter (new chat) starts with it."""
    hub = AgentHub(bridge, chat_pack())
    chat = hub._chat("s1")
    await hub._start(chat)
    await hub._workspace_note(chat)
    ws = contract.get_workspace("demo")
    contract.save_workspace({**ws, "memory": [*ws["memory"], {"text": "fare in pounds"}]})
    assert "Workspace memory" not in await hub._workspace_note(chat)
    chat.adapter = None
    await hub._start(chat)
    assert "m3 (fact) fare in pounds" in await hub._workspace_note(chat)
