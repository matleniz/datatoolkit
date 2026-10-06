"""Memory / document ids are never reused in a workspace (datatoolkit-issues#180)."""

from __future__ import annotations

import pytest

from dtk_engine import contract
from dtk_engine.agent.commands import UI_COMMANDS
from dtk_engine.agent.digest import intro_note, keys_note
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))


def _ws(**extra) -> dict:
    return {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        **extra,
    }


def test_forgotten_memory_id_is_not_reused():
    saved = contract.save_workspace(_ws(memory=[{"text": "a"}]))
    assert saved["memory"][0]["id"] == "m1"
    saved = contract.save_workspace({**saved, "memory": []})
    saved = contract.save_workspace({**saved, "memory": [{"text": "b"}]})
    assert saved["memory"][0]["id"] == "m2"


def test_counter_survives_a_client_dropping_it():
    saved = contract.save_workspace(_ws(memory=[{"text": "a"}, {"text": "b"}]))
    saved.pop("id_counters")
    saved = contract.save_workspace({**saved, "memory": [saved["memory"][0], {"text": "c"}]})
    assert [e["id"] for e in saved["memory"]] == ["m1", "m3"]
    # m2 was forgotten, m3 too: neither comes back
    saved.pop("id_counters")
    saved = contract.save_workspace({**saved, "memory": [{"text": "d"}]})
    assert saved["memory"][0]["id"] == "m4"


def test_stored_workspace_without_counters_reads_from_its_ids(tmp_path):
    path = tmp_path / "workspaces" / "demo.json"
    contract.save_workspace(_ws(memory=[{"id": "m5", "text": "a"}]))
    import json

    raw = json.loads(path.read_text())
    raw.pop("id_counters")
    path.write_text(json.dumps(raw))
    saved = contract.save_workspace({**contract.get_workspace("demo"), "memory": [{"text": "n"}]})
    assert saved["memory"][0]["id"] == "m6"


def test_document_ids_are_not_reused(tmp_path):
    f = tmp_path / "d.md"
    f.write_text("hi")
    saved = contract.save_workspace(_ws())
    assert saved["id_counters"] == {"d": 0, "m": 0}
    from dtk_engine.workspace.models import Workspace

    ws = Workspace.model_validate(
        _ws(
            documents=[{"name": "a", "path": "p"}],
            id_counters={"d": 3},
        )
    )
    assert ws.documents[0].id == "d4"
    assert ws.id_counters["d"] == 4


def test_ack_tool_descriptions_state_the_save_semantics():
    for name in ("propose_steps", "set_note", "remember", "forget", "keep_attachment"):
        assert "Studio has saved" in UI_COMMANDS[name].description, name


def test_keys_note_comes_from_the_registry():
    note = keys_note(contract.list_keys())
    for key in contract.list_keys():
        assert key["id"] in note
    assert "impute_benchmark" in note and len(note) < 500
    assert keys_note([]) == ""
    assert intro_note([], [], contract.list_keys()) == note
    assert intro_note([], []) == ""
