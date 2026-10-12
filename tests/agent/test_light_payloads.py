"""Lighter get_workspace / get_notes payloads (datatoolkit-issues#192)."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from dtk_engine import contract
from dtk_engine.agent.policy import NOTE_CAP, AuditLog, cap_notes, compact_workspace
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio
LONG = "x" * 1000


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DTK_HOME", str(h))
    monkeypatch.delenv("DTK_AGENT_MAX_CHARS", raising=False)


@pytest.fixture
async def client():
    async with Client(build_server(LocalUiPort(UiBridge("t")), AuditLog())) as c:
        yield c


def _save() -> dict:
    steps = [
        {"op": "scale", "target": "both", "params": {"columns": ["Age"]}},
        {"op": "log1p", "target": "both", "params": {"columns": ["Fare"]}, "note": LONG},
    ]
    return contract.save_workspace({
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": steps,
        "notes": {"workspace": LONG, "columns": {"Age": LONG, "Fare": "short"}},
    })


async def _call(client, tool: str, args: dict) -> dict:
    return json.loads((await client.call_tool(tool, args)).content[0].text)


def test_compact_workspace_lines_and_heads():
    out = compact_workspace(_save())
    step = out["steps"][1]
    assert set(step) == {"id", "op", "target", "params", "has_note"}
    assert "has_note" not in out["steps"][0]
    assert step["params"] == '{"columns":["Fare"]}'
    wide = {"steps": [{"id": "s1", "op": "x", "target": "both", "params": {"c": "y" * 500}}]}
    assert len(compact_workspace(wide)["steps"][0]["params"]) <= 81
    assert len(out["notes"]["workspace"]) < 100
    assert out["notes"]["columns"]["Fare"] == "short"


def test_cap_notes_names_what_was_cut():
    out = cap_notes({"workspace": LONG, "steps": {"s1": "ok"}, "columns": {"A": LONG}})
    assert len(out["workspace"]) == NOTE_CAP + 1
    assert out["steps"]["s1"] == "ok"
    assert out["capped"] == ["workspace", "column:A"]


async def test_get_workspace_compact_then_detail(client):
    saved = _save()
    compact = await _call(client, "get_workspace", {"name": "demo"})
    full = await _call(client, "get_workspace", {"name": "demo", "detail": True})
    assert full["data"]["steps"] == saved["steps"]
    assert len(json.dumps(compact)) < len(json.dumps(full))
    assert compact["data"]["steps"][1]["id"] == saved["steps"][1]["id"]


async def test_get_notes_capped_full_and_filtered(client):
    saved = _save()
    r = await _call(client, "get_notes", {"workspace": "demo"})
    capped = r["data"]
    assert len(capped["workspace"]) == NOTE_CAP + 1
    assert "workspace" in capped["capped"]
    full = (await _call(client, "get_notes", {"workspace": "demo", "full": True}))["data"]
    assert full["workspace"] == LONG and "capped" not in full
    sid = saved["steps"][1]["id"]
    one = (await _call(client, "get_notes", {"workspace": "demo", "step_id": sid, "full": True}))["data"]
    assert one["steps"] == {sid: LONG} and one["columns"] == {}
    cols = (await _call(client, "get_notes", {"workspace": "demo", "columns": ["Fare"]}))["data"]
    assert cols["columns"] == {"Fare": "short"} and cols["steps"] == {}
