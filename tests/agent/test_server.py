"""MCP server over the in-memory client: tool list, context fill, policy, UI relay."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from dtk_engine import contract
from dtk_engine.agent import policy
from dtk_engine.agent.commands import UI_COMMANDS
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio

STATIC_TOOLS = {
    "list_keys", "key_schema", "run_key", "list_transforms", "transform_schema",
    "list_workspaces", "get_workspace", "get_rows", "get_profiles", "preview_step",
    "align_report", "source_columns", "get_ui_context", "get_command_status",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DTK_HOME", str(h))
    for name in ("DTK_UPLOAD_DIR", "DTK_AGENT_LOG", "DTK_AGENT_MAX_CHARS"):
        monkeypatch.delenv(name, raising=False)
    return h


@pytest.fixture
def bridge():
    return UiBridge("t")


@pytest.fixture
def audit():
    return AuditLog()


@pytest.fixture
async def client(bridge, audit):
    async with Client(build_server(LocalUiPort(bridge), audit)) as c:
        yield c


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


def _save_demo(name: str = "demo") -> None:
    contract.save_workspace(
        {
            "name": name,
            "datasets": {
                "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
                "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
            },
        }
    )


def _context(session: str = "s1", **extra) -> dict:
    return {
        "session": session, "workspace": "demo", "role": "train", "version": 0,
        "latest": 0, "identity": "demo|train|0", **extra,
    }


async def _listener(bridge, session: str, acks: list[dict]):
    """A fake Studio: acks each relayed command with the next canned ack."""
    queue = bridge.add_listener(session)
    seen = []
    while acks:
        cmd = await queue.get()
        seen.append(cmd)
        bridge.ack({"id": cmd["id"], **acks.pop(0)})
    return seen


async def test_tool_list_is_static_plus_generated(client):
    names = {t.name for t in (await client.list_tools()).tools}
    assert names == STATIC_TOOLS | {c.tool_name for c in UI_COMMANDS.values()}
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert tools["get_rows"].annotations.read_only_hint is True
    assert not (tools["propose_steps"].annotations and tools["propose_steps"].annotations.read_only_hint)
    # Studio reuses an open window and keeps omitted params: the agent must know how to clear `by`.
    assert "one window per tool" in tools["open_window"].description
    assert 'params.by ""' in tools["open_window"].description


async def test_run_key_without_context_uses_demo_data(client):
    out = _payload(await client.call_tool("run_key", {"key": "dataset_overview"}))
    assert out["identity"] is None
    assert "no Studio context" in out["note"]
    assert all("plotly" not in f for f in out["data"]["figures"])
    full = _payload(
        await client.call_tool("run_key", {"key": "dataset_overview", "include_figures": True})
    )
    assert any("plotly" in f for f in full["data"]["figures"])


async def test_run_key_fills_dataset_source_from_context(client, bridge):
    _save_demo()
    bridge.put_context(_context())
    out = _payload(await client.call_tool("run_key", {"key": "dataset_overview"}))
    assert out["identity"] == "demo|train|0"
    assert "note" not in out


async def test_run_key_explicit_source_has_no_identity(client, bridge):
    _save_demo()
    bridge.put_context(_context())
    source = {"kind": "dataset", "workspace": "demo"}
    out = _payload(await client.call_tool("run_key", {"key": "dataset_overview", "params": {"source": source}}))
    assert out["identity"] is None


async def test_csv_outside_home_is_refused(client, tmp_path):
    outside = tmp_path / "x.csv"
    outside.write_text("a\n1\n")
    result = await client.call_tool(
        "run_key", {"key": "dataset_overview", "params": {"source": {"kind": "csv", "path": str(outside)}}}
    )
    assert result.is_error
    body = _payload(result)
    assert body["type"] == "PolicyError"
    assert "path not allowed" in body["message"]


async def test_sql_source_is_refused(client):
    result = await client.call_tool(
        "run_key",
        {"key": "dataset_overview", "params": {"source": {"kind": "sql", "query": "select 1"}}},
    )
    assert result.is_error
    assert "sql sources are not available" in _payload(result)["message"]


async def test_unknown_tool_and_key_are_tool_errors(client):
    assert (await client.call_tool("nope", {})).is_error
    result = await client.call_tool("key_schema", {"key": "nope"})
    assert result.is_error
    assert _payload(result)["type"] == "UnknownKeyError"


async def test_propose_steps_without_listener(client):
    result = await client.call_tool(
        "propose_steps", {"ops": [{"add": {"step": {"op": "drop_columns"}}}]}
    )
    assert not result.is_error
    assert _payload(result)["data"] == {"id": "c1", "ok": False, "error": "no_studio"}


async def test_propose_steps_relays_with_context_fill(client, bridge):
    bridge.put_context(_context())
    studio = asyncio.create_task(_listener(bridge, "s1", [{"ok": True, "identity": "demo|train|1"}]))
    await asyncio.sleep(0)
    ops = [{"add": {"step": {"op": "impute", "params": {}}}}]
    out = _payload(await client.call_tool("propose_steps", {"ops": ops}))
    (cmd,) = await studio
    assert cmd["type"] == "propose_steps"
    assert cmd["workspace"] == "demo"
    assert cmd["base_identity"] == "demo|train|0"
    assert cmd["ops"] == ops
    assert out["data"]["ok"] is True
    assert out["identity"] == "demo|train|1"


async def test_pending_review_then_status(client, bridge):
    bridge.put_context(_context())
    queue = bridge.add_listener("s1")

    async def studio():
        cmd = await queue.get()
        bridge.ack({"id": cmd["id"], "pending": "review"})
        await asyncio.sleep(0.05)
        bridge.ack({"id": cmd["id"], "ok": True, "identity": "demo|train|1"})

    task = asyncio.create_task(studio())
    pending = _payload(await client.call_tool("propose_steps", {"ops": [{"remove": {"index": 0}}]}))
    assert pending["data"]["pending"] == "review"
    cid = pending["data"]["id"]
    assert _payload(await client.call_tool("get_command_status", {"id": cid}))["data"]["pending"] == "review"
    await task
    final = _payload(await client.call_tool("get_command_status", {"id": cid}))
    assert final["data"]["ok"] is True
    assert final["identity"] == "demo|train|1"
    assert (await client.call_tool("get_command_status", {"id": "zzz"})).is_error


async def test_get_rows_limits(client, bridge, monkeypatch):
    monkeypatch.setattr(policy, "DEFAULT_ROWS", 10)
    _save_demo()
    bridge.put_context(_context())
    out = _payload(await client.call_tool("get_rows", {}))
    assert len(out["data"]["rows"]) == 10
    assert out["identity"] == "demo|train|0"
    other = _payload(await client.call_tool("get_rows", {"role": "test", "limit": 3}))
    assert len(other["data"]["rows"]) == 3
    assert other["identity"] is None
    refused = await client.call_tool("get_rows", {"limit": 501})
    assert refused.is_error
    assert "limit must be between 1 and 500" in _payload(refused)["message"]


async def test_get_rows_needs_workspace_without_context(client):
    result = await client.call_tool("get_rows", {})
    assert result.is_error
    assert "pass workspace" in _payload(result)["message"]


async def test_context_tool_and_resource(client, bridge):
    empty = _payload(await client.call_tool("get_ui_context", {}))
    assert empty["data"] is None
    bridge.put_context(_context())
    ctx = _payload(await client.call_tool("get_ui_context", {}))
    assert ctx["data"]["workspace"] == "demo"
    resource = await client.read_resource("studio://context")
    assert json.loads(resource.contents[0].text)["workspace"] == "demo"


async def test_audit_records_calls_without_values(client, bridge, audit):
    _save_demo()
    bridge.put_context(_context())
    await client.call_tool("get_rows", {"limit": 2})
    await client.call_tool("propose_steps", {"ops": [{"add": {"step": {"op": "secret_value_42"}}}]})
    await client.call_tool("get_rows", {"limit": 501})
    rows, propose, refused = audit.entries()
    assert (rows["tool"], rows["status"], rows["identity"]) == ("get_rows", "ok", "demo|train|0")
    assert propose["args"]["ops"] == "<list 1>"
    assert "secret_value_42" not in json.dumps(audit.entries())
    assert refused["status"] == "error"
