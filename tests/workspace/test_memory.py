"""Agent memory in the workspace JSON (datatoolkit-issues#179)."""

from __future__ import annotations

import pytest

from dtk_engine import contract
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace.dataset import workspace_key
from dtk_engine.workspace.models import (
    MEMORY_CHARS_MAX,
    MEMORY_ENTRIES_MAX,
    MEMORY_ENTRY_MAX,
    Workspace,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))


def _ws(memory=None) -> dict:
    out = {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
    }
    if memory is not None:
        out["memory"] = memory
    return out


def test_round_trip_ids_and_defaults():
    assert contract.save_workspace(_ws())["memory"] == []
    saved = contract.save_workspace(_ws([
        {"text": "ledd is in mg/day, 0 means untreated"},
        {"id": "m7", "text": "impute per patient", "kind": "decision",
         "updated_at": "2026-10-06T10:00:00Z"},
        {"text": "check outliers in fare", "kind": "todo"},
    ]))
    assert [(e["id"], e["kind"]) for e in saved["memory"]] == [
        ("m8", "fact"), ("m7", "decision"), ("m9", "todo"),
    ]
    assert contract.get_workspace("demo") == saved
    assert contract.workspace_memory("demo") == saved["memory"]
    summary = contract.memory_summary("demo")
    assert summary["chars"] == sum(len(e["text"]) for e in saved["memory"])
    assert summary["max_chars"] == MEMORY_CHARS_MAX


def test_caps():
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws([{"text": "x" * (MEMORY_ENTRY_MAX + 1)}]))
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws([{"text": ""}]))
    with pytest.raises(KeyParamsError, match="memory entries"):
        contract.save_workspace(_ws([{"text": "x"}] * (MEMORY_ENTRIES_MAX + 1)))
    over = MEMORY_CHARS_MAX // MEMORY_ENTRY_MAX + 1
    with pytest.raises(KeyParamsError, match="characters"):
        contract.save_workspace(_ws([{"text": "x" * MEMORY_ENTRY_MAX}] * over))
    with pytest.raises(KeyParamsError, match="duplicate memory id"):
        contract.save_workspace(_ws([{"id": "m1", "text": "a"}, {"id": "m1", "text": "b"}]))
    with pytest.raises(KeyParamsError):
        contract.save_workspace(_ws([{"text": "a", "kind": "note"}]))


def test_memory_travels_with_duplicate_and_is_not_data():
    contract.save_workspace(_ws([{"text": "ledd in mg/day"}]))
    assert contract.duplicate_workspace("demo", "copy")["memory"][0]["text"] == "ledd in mg/day"
    a = Workspace.model_validate(_ws())
    b = Workspace.model_validate(_ws([{"text": "x"}]))
    assert workspace_key(a, "x", "train", a.steps) == workspace_key(b, "x", "train", b.steps)


def test_memory_of_a_legacy_workspace_is_empty():
    contract.save_workspace(_ws())
    assert contract.workspace_memory("demo") == []
