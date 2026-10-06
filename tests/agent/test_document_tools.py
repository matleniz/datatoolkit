"""Workspace documents through the agent tools and the turn note (datatoolkit-issues#178)."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from dtk_engine import contract
from dtk_engine.agent.digest import intro_note
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "up"))
    for name in ("DTK_AGENT_LOG", "DTK_AGENT_MAX_CHARS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "up" / "h1").mkdir(parents=True)
    contract.save_workspace({
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
    })
    return tmp_path


def _upload(home, name: str, text: str) -> str:
    path = home / "up" / "h1" / name
    path.write_text(text)
    return str(path)


@pytest.fixture
def bridge():
    b = UiBridge("t")
    b.put_context({"session": "s1", "workspace": "demo", "role": "train", "version": 0,
                   "latest": 0, "identity": "demo|train|v0|x"})
    return b


@pytest.fixture
async def client(bridge):
    async with Client(build_server(LocalUiPort(bridge), AuditLog())) as c:
        yield c


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


async def test_list_and_read_documents(client, home):
    contract.add_document("demo", _upload(home, "dictionary.md", "ledd: mg/day\n" * 3), note="dict")
    contract.add_document("demo", _upload(home, "visits.csv", "id,age\n1,2\n"))
    listed = _payload(await client.call_tool("list_documents", {}))["data"]
    assert [(d["id"], d["kind"], d["note"]) for d in listed] == [("d1", "text", "dict"), ("d2", "table", None)]
    assert "path" not in listed[0] and listed[1]["source"]["kind"] == "csv"
    read = _payload(await client.call_tool("read_document", {"id": "d1", "max_chars": 13}))
    assert read["data"]["text"] == "ledd: mg/day\n" and read["data"]["next_offset"] == 13
    assert read["data"]["total_chars"] == 39
    table = _payload(await client.call_tool("read_document", {"id": "d2"}))["data"]
    assert table["source"]["path"].endswith("visits.csv") and "usual tools" in table["hint"]
    missing = await client.call_tool("read_document", {"id": "d9"})
    assert missing.is_error and "no document" in missing.content[0].text


async def test_read_document_respects_the_size_cap(client, home, monkeypatch):
    monkeypatch.setenv("DTK_AGENT_MAX_CHARS", "500")
    contract.add_document("demo", _upload(home, "long.txt", "x" * 5000))
    out = _payload(await client.call_tool("read_document", {"id": "d1"}))
    assert len(json.dumps(out)) < 900 and out["data"]["next_offset"] < 5000


async def test_documents_tools_are_read_only(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert tools["list_documents"].annotations.read_only_hint is True
    assert tools["read_document"].annotations.read_only_hint is True
    assert tools["keep_attachment"].annotations is None  # a write (through Studio)


async def test_keep_attachment_is_relayed_with_the_workspace(client, bridge):
    queue = bridge.add_listener("s1")
    call = asyncio.create_task(client.call_tool("keep_attachment", {"attachment_id": "a2"}))
    cmd = await asyncio.wait_for(queue.get(), 5)
    bridge.ack({"id": cmd["id"], "ok": True, "document_id": "d1"})
    assert _payload(await call)["data"]["document_id"] == "d1"
    assert cmd == {"id": cmd["id"], "type": "keep_attachment", "attachment_id": "a2", "workspace": "demo"}


def test_intro_names_the_documents():
    docs = [{"id": f"d{i}", "name": f"f{i}.md"} for i in range(1, 23)]
    assert intro_note(docs[:2]) == "[Workspace documents (read_document): d1 f1.md, d2 f2.md]\n\n"
    assert intro_note([]) == ""
    assert intro_note(docs).endswith(", … 2 more (list_documents)]\n\n")


async def test_documents_once_per_conversation(home, monkeypatch):
    from dtk_engine.agent.chat import AgentHub
    from dtk_engine.agent.packs.stub import chat_pack

    contract.add_document("demo", _upload(home, "dictionary.md", "x"))
    bridge = UiBridge("t")
    bridge.put_context({"session": "s1", "workspace": "demo", "role": "train", "version": 0,
                        "latest": 0, "identity": "i"})
    hub = AgentHub(bridge, chat_pack())
    chat = hub._chat("s1")
    first = await hub._workspace_note(chat)
    assert "[Workspace documents (read_document): d1 dictionary.md]" in first
    assert "Workspace documents" not in await hub._workspace_note(chat)
    chat.introduced = set()  # what a new adapter (a new conversation) does
    assert "Workspace documents" in await hub._workspace_note(chat)
