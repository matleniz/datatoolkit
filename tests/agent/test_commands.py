"""The UI command table: generated tools, relay, policy, audit and the published schema."""

from __future__ import annotations

import asyncio
import json

import jsonschema
import pytest
from fastapi.testclient import TestClient
from mcp import Client

from dtk_engine.agent import policy
from dtk_engine.agent.commands import UI_COMMANDS, command_schemas
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.agent.tools import build_tools
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio

# One valid call per command, as Studio's parseCommand accepts it.
EXAMPLES: dict[str, dict] = {
    "propose_steps": {
        "ops": [{"add": {"step": {"op": "impute", "params": {}}}}],
        "workspace": "demo", "base_identity": "demo|train|0",
    },
    "open_window": {"tool": "dist", "params": {"column": "age", "by": "sex"}},
    "select_columns": {"columns": ["age", "fare"]},
    "set_view": {"role": "test", "version": None},
    "set_grid_view": {
        "filter": {
            "conditions": [
                {"column": "age", "op": "gt", "value": 30},
                {"column": "sex", "op": "isin", "value": ["male"]},
                {"column": "cabin", "op": "notna"},
            ],
            "combine": "and",
        },
        "sort": [{"column": "fare", "desc": True}],
    },
    "pick_row": {"rid": 3},
    "pick_cell": {"rid": 3, "column": "age"},
    "clear_selection": {},
    "set_target": {"column": "survived"},
    "set_dist_by": {"by": None},
    "set_tool_params": {"tool": "outliers", "params": {"method": "iqr"}, "column": "fare"},
    "add_variable": {"name": "mean_age", "stat": "mean", "column": "age"},
    "set_note": {"kind": "column", "column": "age", "text": "imputed once per patient"},
    "keep_attachment": {"attachment_id": "a1", "note": "data dictionary"},
    "draft_chart": {"params": {"chart": "scatter", "x": "age", "y": "fare", "log_y": True}},
    "add_chart": {"name": "Age vs fare", "params": {"chart": "scatter", "x": "age", "y": "fare"}},
    "edit_step": {"index": 0},
    "fill_editor": {"op": "impute", "params": {"strategy": "median"}, "target": "train"},
}
# Shapes Studio refuses (bad_command); the schema must refuse them too.
INVALID: list[tuple[str, dict]] = [
    ("pick_row", {"rid": -1}),
    ("pick_cell", {"rid": 1}),
    ("add_variable", {"name": "1x", "stat": "mean", "column": "age"}),
    ("add_variable", {"name": "x", "stat": "mode", "column": "age"}),
    ("draft_chart", {"params": {}}),
    ("draft_chart", {"params": {"bins": 1}}),
    ("draft_chart", {"params": {"colour": "age"}}),
    ("add_chart", {"name": "x" * 65, "params": {}}),
    ("set_grid_view", {"filter": {"conditions": [], "combine": "and"}}),
    ("set_grid_view", {"filter": {"conditions": [{"column": "a", "op": "like"}], "combine": "and"}}),
    ("set_grid_view", {"sort": [{"column": "a"}]}),
    ("set_target", {}),
    ("set_tool_params", {"tool": "outliers", "params": {}}),
    ("open_window", {"tool": "nope"}),
    ("fill_editor", {"target": "all"}),
    ("keep_attachment", {"note": "no id"}),
    ("keep_attachment", {"attachment_id": ""}),
]
# Added by datatoolkit-issues#100 (Studio commands beyond phase 1).
NEW_TOOLS = {
    "set_grid_view", "pick_row", "pick_cell", "clear_selection", "add_variable",
    "draft_chart", "add_chart", "edit_step", "fill_editor", "set_target",
    "set_dist_by", "set_tool_params",
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


# -- the table ---------------------------------------------------------------


def test_every_command_has_an_example():
    assert set(EXAMPLES) == set(UI_COMMANDS)


@pytest.mark.parametrize("cmd_type", sorted(UI_COMMANDS))
def test_schema_is_valid_and_accepts_the_example(cmd_type):
    schema = UI_COMMANDS[cmd_type].input_schema
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(EXAMPLES[cmd_type], schema)


@pytest.mark.parametrize(("cmd_type", "args"), INVALID)
def test_schema_refuses_what_studio_refuses(cmd_type, args):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(args, UI_COMMANDS[cmd_type].input_schema)


def test_tool_schemas_are_plain_objects_with_session():
    """Model APIs refuse top-level anyOf / oneOf / allOf in a tool's input_schema."""
    for spec in build_tools(LocalUiPort(UiBridge("t"))):
        schema = spec.input_schema
        assert schema["type"] == "object", spec.name
        assert not {"anyOf", "oneOf", "allOf"} & set(schema), spec.name
        if spec.name in {c.tool_name for c in UI_COMMANDS.values()}:
            assert "session" in schema["properties"], spec.name


def test_policy_allows_exactly_the_table():
    assert set(UI_COMMANDS) == policy.UI_COMMANDS
    for cmd_type in UI_COMMANDS:
        policy.check_command(cmd_type)
    with pytest.raises(policy.PolicyError):
        policy.check_command("eval_js")


# -- the MCP tools -----------------------------------------------------------


async def test_tool_list_has_the_new_commands(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) >= NEW_TOOLS
    for name in NEW_TOOLS:
        assert tools[name].description
        assert not (tools[name].annotations and tools[name].annotations.read_only_hint)


@pytest.mark.parametrize("cmd_type", sorted(UI_COMMANDS))
async def test_each_command_without_studio(cmd_type, client, audit):
    """No session, no listener: the bridge answers no_studio (a framed ack, not a tool error)."""
    spec = UI_COMMANDS[cmd_type]
    result = await client.call_tool(spec.tool_name, EXAMPLES[cmd_type])
    assert not result.is_error
    out = _payload(result)
    assert out["data"] == {"id": "c1", "ok": False, "error": "no_studio"}
    (entry,) = audit.entries()
    assert entry["tool"] == spec.tool_name and entry["status"] == "ok"


@pytest.mark.parametrize("cmd_type", sorted(UI_COMMANDS))
async def test_each_command_relays_its_body(cmd_type, client, bridge, audit):
    """A fake Studio tab gets exactly ``{id, type, **args}`` (session routes, is not sent)."""
    queue = bridge.add_listener("tab")
    args = {**EXAMPLES[cmd_type], "session": "tab"}
    call = asyncio.create_task(client.call_tool(UI_COMMANDS[cmd_type].tool_name, args))
    cmd = await asyncio.wait_for(queue.get(), 5)
    bridge.ack({"id": cmd["id"], "ok": True, "identity": "demo|train|1"})
    out = _payload(await call)
    assert cmd == {"id": cmd["id"], "type": cmd_type, **EXAMPLES[cmd_type]}
    assert out["data"]["ok"] is True
    assert out["identity"] == "demo|train|1"
    assert audit.entries()[-1]["identity"] == "demo|train|1"


async def test_studio_refusal_is_relayed(client, bridge):
    queue = bridge.add_listener("tab")
    call = asyncio.create_task(
        client.call_tool("pick_cell", {"rid": 0, "column": "ghost", "session": "tab"})
    )
    cmd = await asyncio.wait_for(queue.get(), 5)
    bridge.ack({"id": cmd["id"], "ok": False, "error": 'bad_command: unknown column "ghost"'})
    out = _payload(await call)
    assert out["data"]["error"].startswith("bad_command")


# -- the published schema ----------------------------------------------------


def test_schema_route_matches_the_table(monkeypatch):
    monkeypatch.setenv("DTK_UI_TOKEN", "tok")
    with TestClient(create_app(), base_url="http://localhost") as http:
        assert http.get("/api/ui/commands/schema").status_code == 401
        r = http.get("/api/ui/commands/schema", headers={"Authorization": "Bearer tok"})
    assert r.status_code == 200
    body = r.json()
    assert body == command_schemas()
    assert set(body) == set(UI_COMMANDS)
    assert body["propose_steps"]["destructive"] is True
    assert body["pick_row"] == {
        "input_schema": UI_COMMANDS["pick_row"].input_schema, "destructive": False,
    }
