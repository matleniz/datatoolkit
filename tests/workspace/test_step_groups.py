"""Step groups: model invariants, lazy normalisation, HTTP round trip (datatoolkit-issues#249)."""

from __future__ import annotations

import json

import pytest

from dtk_engine import contract
from dtk_engine.agent.commands import UI_COMMANDS
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.workspace.models import Workspace

jsonschema = pytest.importorskip("jsonschema")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    return tmp_path


def _ws(steps=3, **extra) -> dict:
    return {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": [
            {"id": f"s{i}", "op": "drop_duplicates", "target": "train", "params": {"keep": "none"}}
            for i in range(1, steps + 1)
        ],
        **extra,
    }


def _g(gid, *step_ids, **extra) -> dict:
    return {"id": gid, "name": f"Group {gid}", "step_ids": list(step_ids), **extra}


def _ids(ws: Workspace) -> list[list[str]]:
    return [g.step_ids for g in ws.groups]


def test_old_workspace_becomes_one_group_per_step():
    ws = Workspace.model_validate(_ws())
    assert _ids(ws) == [["s1"], ["s2"], ["s3"]]
    first = ws.groups[0]
    assert (first.id, first.name, first.intent, first.origin) == ("g1", "drop_duplicates train", "", "auto")


def test_no_steps_no_groups():
    assert Workspace.model_validate(_ws(steps=0)).groups == []


def test_mixed_covered_and_uncovered_keep_their_position():
    ws = Workspace.model_validate(_ws(steps=5, groups=[_g("g-a", "s2", "s3")]))
    assert _ids(ws) == [["s1"], ["s2", "s3"], ["s4"], ["s5"]]
    assert ws.groups[1].id == "g-a"


def test_empty_group_is_dropped():
    ws = Workspace.model_validate(_ws(groups=[_g("g-a"), _g("g-b", "s1", "s2", "s3")]))
    assert [g.id for g in ws.groups] == ["g-b"]


def test_singleton_id_never_collides():
    ws = Workspace.model_validate(_ws(groups=[_g("g1", "s2", "s3")]))
    ids = [g.id for g in ws.groups]
    assert len(set(ids)) == 2 and "g1" in ids
    assert ws.groups[0].step_ids == ["s1"] and ws.groups[0].id == "g1-2"


def test_normalisation_is_idempotent():
    ws = Workspace.model_validate(_ws(groups=[_g("g-a", "s1", "s2")]))
    again = Workspace.model_validate(ws.model_dump(mode="json"))
    assert again.groups == ws.groups


@pytest.mark.parametrize(
    ("groups", "needle"),
    [
        ([_g("g1", "s1"), _g("g1", "s2")], "duplicate group id 'g1'"),
        ([_g("g1", "s1", "s2"), _g("g2", "s2", "s3")], "step 's2' is in groups 'g1' and 'g2'"),
        ([_g("g1", "s1", "s1")], "step 's1' is in groups 'g1' and 'g1'"),
        ([_g("g1", "s1", "nope")], "group 'g1' names unknown step id 'nope'"),
        ([_g("g1", "s1", "s3")], "group 'g1' is not a contiguous slice"),
        ([_g("g1", "s2", "s1")], "group 'g1' is not a contiguous slice"),
        ([_g("g1", "s2", "s3"), _g("g2", "s1")], "is out of step order"),
    ],
)
def test_invariant_violations_are_named(groups, needle):
    with pytest.raises(ValueError, match=needle):
        Workspace.model_validate(_ws(groups=groups))


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "G1"}, {"id": "x1"}, {"name": ""}, {"name": "x" * 81},
        {"intent": "x" * 401}, {"step_ids": []}, {"origin": "robot"}, {"extra": 1},
    ],
)
def test_group_field_limits(bad):
    group = {**_g("g1", "s1"), **bad}
    if bad == {"step_ids": []}:  # empty groups are dropped, not refused
        assert Workspace.model_validate(_ws(groups=[group])).groups[0].step_ids == ["s1"]
        return
    with pytest.raises(ValueError):
        Workspace.model_validate(_ws(groups=[group]))


def test_groups_do_not_change_version_replay_or_step_count():
    plain = contract.save_workspace(_ws())
    grouped = contract.save_workspace(
        _ws(groups=[_g("g-a", "s1", "s2", "s3", origin="manual", produced_by="ui:x")])
    )
    assert grouped["steps"] == plain["steps"]
    summary = contract.list_workspace_summaries()[0]
    assert summary["step_count"] == 3
    for version in range(4):
        a = contract.preview_workspace({**plain, "steps": plain["steps"][:version], "groups": []}, "train")
        b = contract.preview_workspace({**grouped, "steps": grouped["steps"][:version], "groups": []}, "train")
        assert a == b


def test_old_file_without_groups_loads_then_saves_with_groups(home):
    contract.save_workspace(_ws())
    path = home / "workspaces" / "demo.json"
    if not path.exists():
        path = next(home.rglob("demo.json"))
    raw = json.loads(path.read_text())
    raw.pop("groups")
    path.write_text(json.dumps(raw))
    loaded = contract.get_workspace("demo")
    assert [g["step_ids"] for g in loaded["groups"]] == [["s1"], ["s2"], ["s3"]]
    contract.save_workspace(loaded)
    assert len(json.loads(path.read_text())["groups"]) == 3


def test_propose_steps_accepts_a_group():
    schema = UI_COMMANDS["propose_steps"].input_schema
    ops = [{"add": {"step": {"op": "impute", "params": {}}}}]
    jsonschema.validate({"ops": ops, "group": {"name": "Impute ledd", "intent": "why"}}, schema)
    jsonschema.validate({"ops": ops, "group": {"name": "Impute ledd"}}, schema)
    for bad in ({"intent": "x"}, {"name": ""}, {"name": "x" * 81}, {"name": "a", "intent": "x" * 401}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"ops": ops, "group": bad}, schema)


def test_http_round_trip_and_422():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    client = TestClient(create_app())
    body = _ws(steps=4)
    r = client.put("/api/workspaces/demo", json=body)
    assert r.status_code == 200
    assert [g["step_ids"] for g in r.json()["groups"]] == [["s1"], ["s2"], ["s3"], ["s4"]]

    grouped = {**body, "groups": [_g("g-a", "s1"), _g("g-b", "s2", "s3", "s4", intent="why")]}
    assert client.put("/api/workspaces/demo", json=grouped).status_code == 200
    got = client.get("/api/workspaces/demo").json()
    assert [g["step_ids"] for g in got["groups"]] == [["s1"], ["s2", "s3", "s4"]]
    assert got["groups"][1]["intent"] == "why"

    bad = {**body, "groups": [_g("g-a", "s1", "s3"), _g("g-b", "s2", "s4")]}
    r = client.put("/api/workspaces/demo", json=bad)
    assert r.status_code == 422
    assert "group 'g-a' is not a contiguous slice" in r.json()["message"]
    assert client.get("/api/workspaces/demo").json()["groups"] == got["groups"]

    schema = client.get("/openapi.json").json()
    assert "groups" in json.dumps(schema)


@pytest.mark.parametrize(
    ("groups", "rule", "group_id", "step_id", "needle"),
    [
        ([_g("g1", "s1"), _g("g1", "s2")], "duplicate_group_id", "g1", None, "duplicate group id 'g1'"),
        ([_g("g1", "s1", "s2"), _g("g2", "s2", "s3")], "step_shared", "g2", "s2",
         "step 's2' is in groups 'g1' and 'g2'"),
        ([_g("g1", "s1", "nope")], "unknown_step", "g1", "nope", "group 'g1' names unknown step id 'nope'"),
        ([_g("g9", "nope")], "unknown_step", "g9", "nope", "group 'g9' names unknown step id 'nope'"),
        ([_g("g1", "s1", "s3")], "not_contiguous", "g1", "s3", "group 'g1' is not a contiguous slice"),
        ([_g("g1", "s2", "s1")], "not_contiguous", "g1", "s1", "group 'g1' is not a contiguous slice"),
        ([_g("g1", "s2", "s3"), _g("g2", "s1")], "out_of_order", "g1", "s1", "is out of step order"),
        ([{**_g("g1", "s1"), "name": "a\nb"}], "invalid_name", "g1", None, "group name must be one line"),
    ],
)
def test_http_422_names_the_violated_rule(groups, rule, group_id, step_id, needle):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    client = TestClient(create_app())
    r = client.put("/api/workspaces/demo", json=_ws(groups=groups))
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == {"rule": rule, "group_id": group_id, "step_id": step_id}
    assert needle in body["message"]
    assert body["type"] == "KeyParamsError"
    assert body["details"]


def test_http_unrelated_422_has_no_detail():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    client = TestClient(create_app())
    r = client.put("/api/workspaces/demo", json={**_ws(), "bogus": 1})
    assert r.status_code == 422
    assert "detail" not in r.json()


def test_library_path_unchanged():
    from dtk_engine.errors import KeyParamsError

    with pytest.raises(KeyParamsError, match="duplicate group id 'g1'") as err:
        contract.save_workspace(_ws(groups=[_g("g1", "s1"), _g("g1", "s2")]))
    assert isinstance(err.value, ValueError)
    assert err.value.detail["rule"] == "duplicate_group_id"
