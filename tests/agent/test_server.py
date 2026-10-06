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
    "list_attachments", "read_attachment", "preview_steps", "evaluate",
    "get_notes", "export_workspace", "list_documents", "read_document",
    "get_memory",
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


async def test_run_key_explicit_source_gets_a_computed_identity(client, bridge):
    _save_demo()
    bridge.put_context(_context())
    source = {"kind": "dataset", "workspace": "demo"}
    out = _payload(await client.call_tool("run_key", {"key": "dataset_overview", "params": {"source": source}}))
    assert out["identity"].startswith("demo|train|v0|")  # #153: never null for a frame


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
    assert other["identity"].startswith("demo|test|v0|")  # not on screen: computed
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


def _save_longitudinal(home, name: str = "long") -> None:
    """7,000 patients x 3 visits; ``ledd`` missing on the middle visit."""
    import pandas as pd

    n = 7000
    df = pd.DataFrame({
        "patient_id": [i for i in range(n) for _ in range(3)],
        "age": [50 + v for _ in range(n) for v in range(3)],
        "ledd": [float(v * 10) if v != 1 else None for _ in range(n) for v in range(3)],
    })
    path = home / "long.csv"
    df.to_csv(path, index=False)
    contract.save_workspace(
        {"name": name, "datasets": {"train": {"x": {"kind": "csv", "path": str(path)}}}}
    )


@pytest.mark.parametrize(
    "step",
    [
        {"op": "group_agg", "params": {"group": "patient_id", "value": "age", "aggs": ["mean", "count"]}},
        {"op": "impute", "params": {
            "columns": ["ledd"], "strategy": "group_interp", "by": "patient_id", "order": "age",
        }},
    ],
)
async def test_preview_step_stays_small_on_high_cardinality(client, home, step):
    """#154: a 7k-group group_agg / a 7k-cell group_interp preview fits ~10k chars."""
    _save_longitudinal(home)
    result = await client.call_tool("preview_step", {"workspace": "long", "step": step})
    text = result.content[0].text
    assert not result.is_error, text
    assert len(text) < 10_000
    out = json.loads(text)
    data = out["data"]
    assert isinstance(data, dict) and "truncated" not in out
    assert data["shape"][0] == 21_000
    if step["op"] == "impute":
        assert data["changed_total"] == 7000 and data["elided"]["changed"] == 2000
    else:
        assert data["elided"]["state.groups"] == 7000
    detail = await client.call_tool(
        "preview_step", {"workspace": "long", "step": step, "detail": True}
    )
    more = json.loads(detail.content[0].text)["data"]
    sample = more["state"]["groups"] if step["op"] == "group_agg" else more["changed"]
    assert len(sample) == policy.PREVIEW_DETAIL_ITEMS


async def test_heavy_reads_are_compact_by_default(client, bridge):
    """#151: no histograms / matched columns / descriptions unless asked."""
    _save_demo()
    bridge.put_context(_context())
    prof = _payload(await client.call_tool("get_profiles", {}))["data"]
    assert all("histogram" not in c and len(c.get("top_values") or []) <= 3 for c in prof["columns"])
    assert "detail" in prof["note"]
    full = _payload(await client.call_tool("get_profiles", {"detail": True}))["data"]
    assert any("histogram" in c for c in full["columns"])
    one = _payload(await client.call_tool("get_profiles", {"columns": ["Age"]}))["data"]
    assert "histogram" in one["columns"][0]
    align = _payload(await client.call_tool("align_report", {}))["data"]
    assert all(c["status"] != "match" for c in align["columns"]) and align["matched"]
    keys = _payload(await client.call_tool("list_keys", {}))["data"]
    assert all("description" not in k for k in keys)
    ops = _payload(await client.call_tool("list_transforms", {"detail": True}))["data"]
    assert all("description" in o for o in ops)


async def test_study_in_memory_leaves_no_trace(client, bridge, home):
    """#155 acceptance: the ledd study (mask 1 row in 5, impute two ways, score)
    runs with preview_steps + evaluate only: no command reaches Studio, the
    workspace is unchanged."""
    import pandas as pd

    n = 500  # patients x 4 visits, ledd linear in the visit
    pd.DataFrame({
        "patient_id": [i for i in range(n) for _ in range(4)],
        "age": [50 + v for _ in range(n) for v in range(4)],
        "ledd": [10.0 * v + i for i in range(n) for v in range(4)],
    }).to_csv(home / "long.csv", index=False)
    contract.save_workspace(
        {"name": "long", "datasets": {"train": {"x": {"kind": "csv", "path": str(home / "long.csv")}}}}
    )
    picked = "(patient_id - 5 * floor(patient_id / 5) == 0) * (age == 51)"
    bridge.put_context({**_context(), "workspace": "long", "identity": "long|train|v0|x"})
    queue = bridge.add_listener("s1")
    stored = contract.get_workspace("long")
    mask = {"op": "formula", "params": {
        "name": "ledd_masked", "expr": f"where({picked}, -999, ledd)",
    }}
    unmask = {"op": "replace_sentinels", "params": {"sentinels": {"ledd_masked": [-999]}}}
    scores = {}
    for strategy in ("group_interp", "group_mean"):
        impute = {"op": "impute", "params": {
            "columns": ["ledd_masked"], "strategy": strategy, "by": "patient_id", "order": "age",
        }}
        steps = [mask, unmask, impute]
        preview = _payload(await client.call_tool("preview_steps", {"steps": steps}))
        assert [s["op"] for s in preview["data"]["steps"]] == ["formula", "replace_sentinels", "impute"]
        out = _payload(await client.call_tool("evaluate", {
            "steps": steps, "exprs": ["abs(ledd_masked - ledd)"],
            "where": picked,
        }))
        scores[strategy] = out["data"]["results"][0]
    assert scores["group_interp"]["count"] == 100
    assert scores["group_interp"]["mean"] == pytest.approx(0)  # linear visits: exact
    assert scores["group_mean"]["mean"] > 0
    assert queue.empty()
    assert contract.get_workspace("long") == stored


async def test_export_tool_writes_only_under_exports(client, bridge, home):
    """#156: the agent exports into $DTK_HOME/exports/<workspace>, nowhere else."""
    _save_demo()
    bridge.put_context(_context())
    out = _payload(await client.call_tool("export_workspace", {}))["data"]
    assert out["out_dir"] == str(home / "exports" / "demo")
    assert out["formats"] == ["ipynb"] and out["outputs"] == {"notebook": "code/pipeline.ipynb"}
    assert (home / "exports" / "demo" / "code" / "pipeline.ipynb").is_file()
    again = await client.call_tool("export_workspace", {"formats": ["py"]})
    assert again.is_error and "overwrite" in again.content[0].text
    done = _payload(await client.call_tool("export_workspace", {"formats": ["py"], "overwrite": True}))
    assert done["data"]["outputs"] == {"script": "code/pipeline.py"}
    assert not (home / "exports" / "demo" / "code" / "pipeline.ipynb").exists()
