"""Every MCP tool declares a title and explicit behaviour hints (agent/tool_meta.py)."""

from __future__ import annotations

import pytest
from mcp import Client

from dtk_engine.agent.commands import UI_COMMANDS
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.agent.tool_meta import TOOL_META
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio

READ_ONLY = {
    "list_keys", "key_schema", "run_key", "list_transforms", "transform_schema",
    "list_workspaces", "get_workspace", "get_rows", "get_profiles", "preview_step",
    "preview_steps", "evaluate", "get_notes", "get_memory", "align_report",
    "source_columns", "get_ui_context", "list_attachments", "read_attachment",
    "list_documents", "read_document", "get_command_status",
}
# name: (destructive, idempotent) for the tools that are not read-only
WRITING = {
    "export_workspace": (True, True),
    "propose_steps": (True, False),
    "set_note": (True, True),
    "keep_attachment": (False, False),
    "remember": (True, False),
    "forget": (True, True),
    "add_variable": (False, True),
    "add_chart": (False, True),
    "open_window": (False, True),
    "select_columns": (False, True),
    "set_view": (False, True),
    "set_grid_view": (False, True),
    "pick_row": (False, True),
    "pick_cell": (False, True),
    "clear_selection": (False, True),
    "set_target": (False, True),
    "set_dist_by": (False, True),
    "set_tool_params": (False, True),
    "draft_chart": (False, True),
    "edit_step": (False, True),
    "fill_editor": (False, True),
}


@pytest.fixture
async def tools():
    server = build_server(LocalUiPort(UiBridge()), AuditLog())
    async with Client(server) as client:
        return {t.name: t for t in (await client.list_tools()).tools}


async def test_every_tool_has_title_and_explicit_hints(tools):
    for name, tool in tools.items():
        assert tool.title and tool.title.strip(), f"{name}: no title"
        a = tool.annotations
        assert a is not None, f"{name}: no annotations"
        assert a.title == tool.title, name
        assert a.read_only_hint is not None, f"{name}: read_only_hint not set"
        assert a.open_world_hint is False, f"{name}: engine-local tools are closed-world"


async def test_hint_table_covers_every_tool_and_pins_the_hints(tools):
    pinned = READ_ONLY | set(WRITING)
    assert not READ_ONLY & set(WRITING)
    missing = set(tools) - pinned
    assert not missing, f"add {sorted(missing)} to the hint table in test_tool_meta.py"
    assert set(tools) == pinned
    assert set(tools) == set(TOOL_META)
    for name in READ_ONLY:
        a = tools[name].annotations
        assert (a.read_only_hint, a.destructive_hint, a.idempotent_hint) == (True, False, True), name
    for name, (destructive, idempotent) in WRITING.items():
        a = tools[name].annotations
        assert a.read_only_hint is False, f"{name} writes: must not be read-only"
        assert (a.destructive_hint, a.idempotent_hint) == (destructive, idempotent), name


async def test_export_description_warns_it_overwrites(tools):
    assert "destructive" in tools["export_workspace"].description


def test_ui_commands_all_have_meta():
    assert {c.tool_name for c in UI_COMMANDS.values()} <= set(TOOL_META)


async def test_closed_world_holds_because_no_tool_can_open_a_network_source():
    """open_world_hint=False everywhere is true only while sql never reaches a reader:
    policy.check_args refuses it in every tool's args and workspaces hold file sources only."""
    sql = {"kind": "sql", "url_env": "NO_SUCH_URL", "query": "select 1"}
    calls = {
        "source_columns": {"spec": sql},
        "run_key": {"key": "dataset_overview", "params": {"source": sql}},
        "preview_step": {"step": {"op": "x", "params": {"source": sql}}},
    }
    server = build_server(LocalUiPort(UiBridge()), AuditLog())
    async with Client(server) as client:
        for name, args in calls.items():
            result = await client.call_tool(name, args)
            assert result.is_error, name
            assert "sql sources are not available" in result.content[0].text, name
