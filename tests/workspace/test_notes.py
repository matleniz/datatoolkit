"""Notes on steps, columns and the workspace (datatoolkit-issues#152)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dtk_engine import contract
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.http import create_app
from dtk_engine.workspace.dataset import workspace_key
from dtk_engine.workspace.models import NOTE_MAX, Workspace

RENAME = {"op": "rename", "target": "both", "params": {"mapping": {"Age": "age"}}}
LOG = {"op": "log1p", "target": "both", "params": {"columns": ["Fare"]}, "note": "skewed"}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))


def _ws(steps=(), notes=None) -> dict:
    out = {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": list(steps),
    }
    if notes is not None:
        out["notes"] = notes
    return out


def test_notes_round_trip_and_defaults():
    assert contract.save_workspace(_ws())["notes"] == {"workspace": None, "columns": {}}
    saved = contract.save_workspace(_ws([LOG], {"workspace": "parkinson study",
                                                "columns": {"Age": "imputed", "Fare": ""}}))
    again = contract.get_workspace("demo")
    assert again == saved
    assert again["steps"][0]["note"] == "skewed"
    assert again["notes"] == {"workspace": "parkinson study", "columns": {"Age": "imputed"}}


def test_note_limits():
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws([{**LOG, "note": "x" * (NOTE_MAX + 1)}]))
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws(notes={"columns": {"Age": "x" * (NOTE_MAX + 1)}}))


def test_notes_are_not_data():
    a = Workspace.model_validate(_ws([{**LOG, "id": "s1"}]))
    b = Workspace.model_validate(_ws([{**LOG, "id": "s1", "note": "other"}]))
    assert workspace_key(a, "x", "train", a.steps) == workspace_key(b, "x", "train", b.steps)


def test_column_notes_follow_renames():
    ws = contract.save_workspace(_ws(
        [RENAME, {"op": "rename", "target": "both", "params": {"mapping": {"age": "age_years"}}}],
        {"columns": {"Age": "imputed", "Fare": "in pounds"}},
    ))
    at0 = contract.column_notes(ws, "train", 0)
    assert at0 == {"notes": {"Age": "imputed", "Fare": "in pounds"}, "keys": {}}
    at1 = contract.column_notes(ws, "train", 1)
    assert at1["notes"]["age"] == "imputed" and at1["keys"] == {"age": "Age"}
    latest = contract.column_notes(ws, "test")
    assert latest["notes"] == {"age_years": "imputed", "Fare": "in pounds"}
    assert latest["keys"] == {"age_years": "Age"}


def test_column_notes_ignore_renames_on_the_other_side():
    one_side = {**RENAME, "target": "test"}
    ws = contract.save_workspace(_ws([one_side], {"columns": {"Age": "imputed"}}))
    assert contract.column_notes(ws, "train")["notes"] == {"Age": "imputed"}
    assert contract.column_notes(ws, "test")["notes"] == {"age": "imputed"}


def test_column_notes_route():
    ws = contract.save_workspace(_ws([RENAME], {"columns": {"Age": "imputed"}}))
    client = TestClient(create_app())
    r = client.post("/api/workspace/column-notes", json={"workspace": ws})
    assert r.status_code == 200
    assert r.json() == {"notes": {"age": "imputed"}, "keys": {"age": "Age"}}
