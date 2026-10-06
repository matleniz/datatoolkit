"""Stable step ids, change digest, base_steps and identities (datatoolkit-issues#153)."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from dtk_engine import contract
from dtk_engine.agent.digest import fnv1a, frame_identity, js_json
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.ui_bridge import UiBridge
from dtk_engine.workspace.dataset import workspace_key
from dtk_engine.workspace.models import Workspace

SCALE = {"op": "scale", "target": "both", "params": {"columns": ["Age"]}}
pytestmark = pytest.mark.anyio

LOG = {"op": "log1p", "target": "both", "params": {"columns": ["Fare"]}}


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
async def client(bridge):
    async with Client(build_server(LocalUiPort(bridge), AuditLog())) as c:
        yield c


def _ws(steps: list[dict]) -> dict:
    return {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": steps,
    }


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


def _context(**extra) -> dict:
    return {
        "session": "s1", "workspace": "demo", "role": "train", "version": 0,
        "latest": 0, "identity": "demo|train|v0|x", **extra,
    }


# -- model / migration -------------------------------------------------------


def test_missing_ids_are_filled_by_position_and_kept():
    ws = Workspace.model_validate(_ws([SCALE, {**LOG, "id": "s1"}, SCALE]))
    assert [s.id for s in ws.steps] == ["s1-2", "s1", "s3"]
    again = Workspace.model_validate(ws.model_dump(mode="json"))
    assert [s.id for s in again.steps] == ["s1-2", "s1", "s3"]


def test_duplicate_or_malformed_ids_are_refused():
    with pytest.raises(ValueError, match="duplicate step id"):
        Workspace.model_validate(_ws([{**SCALE, "id": "sa"}, {**LOG, "id": "sa"}]))
    with pytest.raises(ValueError):
        Workspace.model_validate(_ws([{**SCALE, "id": "Bad id"}]))
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws([{**SCALE, "id": "sa"}, {**LOG, "id": "sa"}]))


def test_legacy_file_reads_the_same_ids_until_saved(home):
    contract.save_workspace(_ws([]))
    path = home / "workspaces" / "demo.json"
    raw = json.loads(path.read_text())
    raw["steps"] = [SCALE, LOG]  # a file written before step ids
    path.write_text(json.dumps(raw))
    assert [s["id"] for s in contract.get_workspace("demo")["steps"]] == ["s1", "s2"]
    assert [s["id"] for s in contract.workspace_steps("demo")] == ["s1", "s2"]
    contract.save_workspace(contract.get_workspace("demo"))
    assert [s["id"] for s in json.loads(path.read_text())["steps"]] == ["s1", "s2"]


def test_ids_stay_out_of_the_replay_cache_key():
    a = Workspace.model_validate(_ws([{**SCALE, "id": "sa"}]))
    b = Workspace.model_validate(_ws([{**SCALE, "id": "sb"}]))
    assert workspace_key(a, "x", "train", a.steps) == workspace_key(b, "x", "train", b.steps)


# -- engine-side identity ----------------------------------------------------


def test_fnv1a_and_js_json_match_studio():
    assert fnv1a("") == "811c9dc5"
    assert fnv1a("a") == "e40c292c"
    assert js_json({"a": 1.0, "b": [0.5, 1e-7, 1e-5, "é", None, True]}) == (
        '{"a":1,"b":[0.5,1e-7,0.00001,"é",null,true]}'
    )


def test_frame_identity_ignores_ids_and_later_steps():
    ws = contract.save_workspace(_ws([SCALE, LOG]))
    v1 = frame_identity(ws, "train", 1)
    assert v1.startswith("demo|train|v1|")
    renamed = {**ws, "steps": [{**s, "id": f"s{9 - i}"} for i, s in enumerate(ws["steps"])]}
    assert frame_identity(renamed, "train", 1) == v1
    assert frame_identity({**ws, "steps": ws["steps"][:1]}, "train", 1) == v1
    assert frame_identity(ws, "train", None).startswith("demo|train|v2|")


# -- change digest and base_steps (bridge side) ------------------------------


async def _studio(bridge, act, ack: dict):
    """A fake Studio: on the next command, ``act(cmd)`` then ``ack``."""
    queue = bridge.add_listener("s1")
    cmd = await queue.get()
    act(cmd)
    bridge.ack({"id": cmd["id"], **ack})
    return cmd


async def test_user_removes_a_step_mid_task(client, bridge):
    contract.save_workspace(_ws([SCALE, LOG]))
    bridge.put_context(_context())
    seen = _payload(await client.call_tool("get_workspace", {"name": "demo"}))
    assert [s["id"] for s in seen["data"]["steps"]] == ["s1", "s2"]
    assert "workspace_changes" not in seen

    contract.save_workspace(_ws([{**LOG, "id": "s2"}]))  # the user removes s1 in Studio

    stale = {"ok": False, "error": "stale: step s1 (scale) removed",
             "stale": [{"id": "s1", "reason": "removed"}]}
    studio = asyncio.create_task(_studio(bridge, lambda cmd: None, stale))
    await asyncio.sleep(0)
    out = _payload(await client.call_tool("propose_steps", {"ops": [{"remove": {"id": "s1"}}]}))
    cmd = await studio
    assert cmd["ops"] == [{"remove": {"id": "s1"}}]
    assert cmd["base_steps"] == {"s1": {k: SCALE[k] for k in ("op", "target", "params")}}
    assert out["workspace_changes"]["removed"] == [{"id": "s1", "op": "scale"}]
    assert out["data"]["stale"] == [{"id": "s1", "reason": "removed"}]
    assert out["data"]["error"].startswith("stale")


async def test_user_change_is_reported_once_and_base_is_what_was_seen(client, bridge):
    contract.save_workspace(_ws([SCALE]))
    bridge.put_context(_context())
    await client.call_tool("get_workspace", {"name": "demo"})
    changed = {**SCALE, "id": "s1", "params": {"columns": ["Fare"]}}
    contract.save_workspace(_ws([changed, LOG]))
    first = _payload(await client.call_tool("list_keys", {}))
    assert first["workspace_changes"]["changed"] == [{"id": "s1", "op": "scale"}]
    assert first["workspace_changes"]["added"] == [{"id": "s2", "op": "log1p"}]
    again = _payload(await client.call_tool("list_keys", {}))
    assert "workspace_changes" not in again
    assert again["identity"] == "demo|train|v0|x"  # no frame: the one on screen

    studio = asyncio.create_task(_studio(bridge, lambda cmd: None, {"ok": True}))
    await asyncio.sleep(0)
    await client.call_tool("propose_steps", {"ops": [{"replace": {"id": "s1", "step": LOG}}]})
    cmd = await studio
    assert cmd["base_steps"]["s1"]["params"] == {"columns": ["Fare"]}


async def test_own_applied_proposal_is_not_a_user_change(client, bridge):
    contract.save_workspace(_ws([SCALE]))
    bridge.put_context(_context())
    await client.call_tool("get_workspace", {"name": "demo"})

    def apply(cmd):  # Studio applies the add and autosaves before acking
        contract.save_workspace(_ws([{**SCALE, "id": "s1"}, {**LOG, "id": "s7f3a09c1"}]))

    studio = asyncio.create_task(_studio(bridge, apply, {"ok": True, "added_ids": ["s7f3a09c1"]}))
    await asyncio.sleep(0)
    out = _payload(await client.call_tool("propose_steps", {"ops": [{"add": {"step": LOG}}]}))
    await studio
    assert out["data"]["added_ids"] == ["s7f3a09c1"]
    after = _payload(await client.call_tool("list_keys", {}))
    assert "workspace_changes" not in after


async def test_reviewed_proposal_outcome_is_reported(client, bridge):
    contract.save_workspace(_ws([SCALE]))
    bridge.put_context(_context())
    await client.call_tool("get_workspace", {"name": "demo"})
    studio = asyncio.create_task(_studio(bridge, lambda cmd: None, {"pending": "review"}))
    await asyncio.sleep(0)
    out = _payload(await client.call_tool("propose_steps", {"ops": [{"remove": {"id": "s1"}}]}))
    await studio
    cid = out["data"]["id"]
    contract.save_workspace(_ws([]))  # the user applies the removal
    bridge.ack({"id": cid, "ok": True, "identity": "demo|train|v0|y"})
    note = _payload(await client.call_tool("list_keys", {}))["workspace_changes"]
    assert note["commands"] == [{"id": cid, "status": "applied"}]
    assert note["removed"] == [{"id": "s1", "op": "scale"}]
    assert "workspace_changes" not in _payload(await client.call_tool("list_keys", {}))


# -- per-turn note (#151) ----------------------------------------------------


def test_turn_note_lists_steps_then_only_changes():
    from dtk_engine.agent.digest import step_line, turn_note

    ctx = _context(version=2, latest=2)
    steps = [{**SCALE, "id": "s1"}, {**LOG, "id": "s2", "target": "train"}]
    first = turn_note(ctx, steps, None, first=True)
    assert 'workspace "demo"' in first and "viewing version 2 of 2" in first
    assert "s1 scale columns=Age; s2 log1p [train] columns=Fare" in first
    same = turn_note(ctx, steps, None, first=False)
    assert "unchanged" in same and "s1 scale" not in same
    moved = turn_note(ctx, steps[1:], {
        "removed": [{"id": "s1", "op": "scale"}], "changed": [{"id": "s2", "op": "log1p"}],
    }, first=False)
    assert "(1 now): changed s2 log1p [train] columns=Fare; removed s1 (scale)." in moved
    long = step_line({"id": "s9", "op": "onehot", "params": {"columns": [f"c{i}" for i in range(50)]}})
    assert long == "s9 onehot columns=c0,c1,c2+47"


# -- notes (#152) ------------------------------------------------------------


async def test_get_notes_resolves_columns_and_steps(client, bridge):
    rename = {"op": "rename", "target": "both", "params": {"mapping": {"Age": "age"}}}
    ws = _ws([{**SCALE, "note": "scaled for knn"}, rename])
    ws["notes"] = {"workspace": "study", "columns": {"Age": "imputed once per patient"}}
    contract.save_workspace(ws)
    bridge.put_context(_context(version=2, latest=2))
    out = _payload(await client.call_tool("get_notes", {}))["data"]
    assert out == {
        "workspace": "study",
        "steps": {"s1": "scaled for knn"},
        "columns": {"age": "imputed once per patient"},
    }


def test_step_line_marks_a_note():
    from dtk_engine.agent.digest import step_line

    assert step_line({**SCALE, "id": "s1", "note": "x"}) == "s1 scale (note) columns=Age"
