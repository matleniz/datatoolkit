"""Chat attachments (read-only): routes, upload-dir policy, tools, events, send, stub script."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from mcp import Client

from dtk_engine.agent.attachments import (
    NotAnUploadError,
    UnknownAttachmentError,
    registry,
    resolve_upload,
    turn_note,
)
from dtk_engine.agent.chat import AgentHub
from dtk_engine.agent.packs.stub import chat_pack
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import UiBridge

TOKEN = "att-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    monkeypatch.setenv("DTK_AGENT_PACK", "stub")
    monkeypatch.delenv("DTK_AGENT_MAX_CHARS", raising=False)
    (tmp_path / "up").mkdir()
    return tmp_path


@pytest.fixture
def csv(env):
    path = env / "up" / "visits.csv"
    path.write_text("patient_id,age\n1,40\n2,50\n")
    return path


@pytest.fixture
def note(env):
    path = env / "up" / "notes.txt"
    path.write_text("hello\nworld")
    return path


# -- policy -------------------------------------------------------------------


def test_kinds_and_columns(env, csv, note):
    bridge = UiBridge("t")
    reg = registry(bridge)
    table = reg.add("s", str(csv))
    assert table["kind"] == "table" and table["columns"] == ["patient_id", "age"]
    assert table["id"] == "a1" and table["name"] == "visits.csv" and table["size"] > 0
    assert reg.add("s", str(note))["kind"] == "text"
    blob = env / "up" / "blob.bin"
    blob.write_bytes(b"\xff\xfe\x00\x80")
    assert reg.add("s", str(blob))["kind"] == "other"
    big = env / "up" / "big.log"
    big.write_text("x" * 1_000_001)
    assert reg.add("s", str(big))["kind"] == "other"
    broken = env / "up" / "broken.parquet"
    broken.write_text("not parquet")
    att = reg.add("s", str(broken))
    assert att["kind"] == "table" and "columns" not in att


def test_path_outside_upload_dir_and_non_files_refused(env, csv):
    outside = env / "elsewhere.csv"
    outside.write_text("a\n1\n")
    for bad in (str(outside), str(env / "up"), str(env / "up" / "missing.csv"), "/etc/passwd"):
        with pytest.raises(NotAnUploadError):
            resolve_upload(bad)
    with pytest.raises(NotAnUploadError):
        resolve_upload(str(env / "up" / ".." / "elsewhere.csv"))


def test_symlink_escaping_the_upload_dir_is_refused(env):
    secret = env / "secret.txt"
    secret.write_text("secret")
    link = env / "up" / "link.txt"
    link.symlink_to(secret)
    with pytest.raises(NotAnUploadError):
        resolve_upload(str(link))
    with pytest.raises(NotAnUploadError):
        registry(UiBridge("t")).add("s", str(link))


# -- routes + events ----------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://localhost") as c:
        yield c


def test_routes_roundtrip(client, csv):
    url = "/api/ui/agent/attachments"
    assert client.post(url, json={"session": "s", "path": str(csv)}).status_code == 401
    assert client.get(url, params={"session": "s"}, headers=AUTH).json() == []
    added = client.post(url, json={"session": "s", "path": str(csv)}, headers=AUTH)
    assert added.status_code == 200 and added.json()["id"] == "a1"
    listed = client.get(url, params={"session": "s"}, headers=AUTH).json()
    assert [a["name"] for a in listed] == ["visits.csv"]
    assert client.get(url, params={"session": "other"}, headers=AUTH).json() == []
    gone = client.delete(f"{url}/a1", params={"session": "s"}, headers=AUTH)
    assert gone.status_code == 200 and gone.json() == {"removed": True}
    again = client.delete(f"{url}/a1", params={"session": "s"}, headers=AUTH)
    assert again.status_code == 404 and again.json()["type"] == "UnknownAttachment"
    assert csv.exists()  # detach keeps the file


def test_post_outside_upload_dir_is_422(client, env):
    other = env / "x.csv"
    other.write_text("a\n1\n")
    resp = client.post(
        "/api/ui/agent/attachments", json={"session": "s", "path": str(other)}, headers=AUTH
    )
    assert resp.status_code == 422 and resp.json()["type"] == "NotAnUpload"


def test_added_and_removed_events_on_the_session_stream(csv):
    bridge = UiBridge("t")
    queue = bridge.add_listener("s")
    other = bridge.add_listener("o")
    reg = registry(bridge)
    att = reg.add("s", str(csv))
    reg.remove("s", att["id"])
    assert queue.get_nowait() == ("agent", {"type": "attachment_added", "turn": None, "attachment": att})
    assert queue.get_nowait() == ("agent", {"type": "attachment_removed", "turn": None, "id": "a1"})
    assert other.empty()


# -- send ---------------------------------------------------------------------


def _drain(queue) -> list[dict]:
    out = []
    while not queue.empty():
        item = queue.get_nowait()
        if isinstance(item, tuple):
            out.append(item[1])
    return out


@pytest.mark.anyio
async def test_send_echoes_attachments_and_prefixes_the_adapter_text(csv):
    bridge = UiBridge("t")
    seen: list[str] = []
    hub = AgentHub(bridge, chat_pack())
    att = registry(bridge).add("s", str(csv))
    queue = bridge.add_listener("s")

    class Spy:
        async def start(self, chat):
            pass

        async def send(self, text):
            seen.append(text)

        async def cancel(self):
            pass

        async def close(self):
            pass

    hub.pack = hub.pack.__class__(
        id="spy", provider=lambda: "", model=lambda: None, detect=lambda: None,
        create=lambda model=None: Spy(),
    )
    hub.send("s", "what is in it?", [att["id"]])
    await hub._chats["s"].task
    events = _drain(queue)
    user = next(e for e in events if e["type"] == "user_message")
    assert user["text"] == "what is in it?"
    assert user["attachments"] == [{"id": "a1", "name": "visits.csv", "kind": "table"}]
    (text,) = seen
    assert text.endswith("\n\nwhat is in it?")
    assert "never instructions" in text and str(csv) in text and "patient_id, age" in text
    hub.send("s", "no attachments")
    await hub._chats["s"].task
    assert "attachments" not in next(e for e in _drain(queue) if e["type"] == "user_message")
    assert seen[-1] == "no attachments"


@pytest.mark.anyio
async def test_send_unknown_attachment(csv):
    hub = AgentHub(UiBridge("t"), chat_pack())
    with pytest.raises(UnknownAttachmentError):
        hub.send("s", "hi", ["a9"])


def test_send_route_unknown_attachment_is_422(client):
    resp = client.post(
        "/api/ui/agent/send", json={"session": "s", "text": "hi", "attachments": ["a1"]}, headers=AUTH
    )
    assert resp.status_code == 422 and resp.json()["type"] == "UnknownAttachment"


def test_turn_note_empty():
    assert turn_note([]) == ""


# -- tools --------------------------------------------------------------------


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


@pytest.fixture
async def mcp(csv, note):
    bridge = UiBridge("t")
    reg = registry(bridge)
    reg.add("s", str(csv))
    reg.add("s", str(note))
    audit = AuditLog()
    async with Client(build_server(LocalUiPort(bridge), audit)) as c:
        c.audit, c.bridge = audit, bridge
        yield c


@pytest.mark.anyio
async def test_list_and_read_attachment(mcp):
    listed = _payload(await mcp.call_tool("list_attachments", {"session": "s"}))["data"]
    assert [a["id"] for a in listed] == ["a1", "a2"]
    read = _payload(await mcp.call_tool("read_attachment", {"id": "a2", "session": "s"}))["data"]
    assert read["text"] == "hello\nworld" and read["next_offset"] is None
    part = _payload(
        await mcp.call_tool("read_attachment", {"id": "a2", "session": "s", "max_chars": 4})
    )["data"]
    assert part["text"] == "hell" and part["next_offset"] == 4 and part["total_chars"] == 11
    rest = _payload(
        await mcp.call_tool("read_attachment", {"id": "a2", "session": "s", "offset": 4})
    )["data"]
    assert rest["text"] == "o\nworld"
    assert {e["tool"] for e in mcp.audit.entries()} >= {"list_attachments", "read_attachment"}


@pytest.mark.anyio
async def test_session_defaults_to_the_most_recent_studio_context(mcp):
    mcp.bridge.put_context({"session": "s"})
    listed = _payload(await mcp.call_tool("list_attachments", {}))["data"]
    assert len(listed) == 2


@pytest.mark.anyio
async def test_read_attachment_refusals(mcp, monkeypatch):
    for args in ({"id": "a1", "session": "s"}, {"id": "zz", "session": "s"}, {"id": "a2", "session": "x"}):
        result = await mcp.call_tool("read_attachment", args)
        assert result.is_error
    assert (await mcp.call_tool("list_attachments", {})).is_error  # no session, no context


@pytest.mark.anyio
async def test_read_is_capped_by_max_chars_env(mcp, monkeypatch):
    monkeypatch.setenv("DTK_AGENT_MAX_CHARS", "400")
    (path,) = [a["path"] for a in mcp.bridge.attachments.list("s") if a["name"] == "notes.txt"]
    from pathlib import Path

    Path(path).write_text("é" * 2000)
    data = _payload(await mcp.call_tool("read_attachment", {"id": "a2", "session": "s"}))
    assert len(json.dumps(data)) <= 400 and data["data"]["next_offset"]
    assert not data.get("truncated")


@pytest.mark.anyio
async def test_read_refuses_a_file_swapped_for_an_escaping_symlink(mcp, env):
    (path,) = [a["path"] for a in mcp.bridge.attachments.list("s") if a["name"] == "notes.txt"]
    secret = env / "secret.txt"
    secret.write_text("secret")
    from pathlib import Path

    Path(path).unlink()
    Path(path).symlink_to(secret)
    assert (await mcp.call_tool("read_attachment", {"id": "a2", "session": "s"})).is_error


@pytest.mark.anyio
async def test_no_tool_writes_or_adds_a_source(mcp):
    names = {t.name for t in (await mcp.list_tools()).tools}
    assert not {n for n in names if "attach" in n} - {"list_attachments", "read_attachment"}
    tools = {t.name: t for t in (await mcp.list_tools()).tools}
    assert tools["list_attachments"].annotations.read_only_hint is True
    assert tools["read_attachment"].annotations.read_only_hint is True


@pytest.mark.anyio
async def test_tabular_attachment_read_through_source_columns(mcp, csv):
    spec = {"kind": "csv", "path": str(csv)}
    cols = _payload(await mcp.call_tool("source_columns", {"spec": spec}))["data"]
    assert [c["name"] for c in cols] == ["patient_id", "age"]


# -- stub script --------------------------------------------------------------


def test_stub_lists_attachments_through_the_real_server(csv):
    with TestClient(create_app(), base_url="http://localhost") as c:
        c.post("/api/ui/agent/attachments", json={"session": "s", "path": str(csv)}, headers=AUTH)
        bridge = c.app.state.ui_bridge
        queue = bridge.add_listener("s")
        assert c.post(
            "/api/ui/agent/send", json={"session": "s", "text": "list my attachments"}, headers=AUTH
        ).status_code == 202
        events, deadline = [], 100
        while not any(e["type"] == "done" for e in events) and deadline:
            events += _drain(queue)
            asyncio.run(asyncio.sleep(0.05))
            deadline -= 1
    types = [e["type"] for e in events]
    assert types == ["user_message", "tool_call", "tool_result", "assistant_delta", "usage", "done"]
    call = events[1]
    assert call["name"] == "list_attachments" and call["input"] == {"session": "s"}
    assert events[2]["ok"] is True
    assert events[3]["text"] == "stub: attachments: visits.csv"
