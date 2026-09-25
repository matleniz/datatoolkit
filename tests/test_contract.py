import json

import pytest
from dtk_engine import (
    delete_workspace,
    get_workspace,
    key_schema,
    list_keys,
    list_workspaces,
    run_key,
    save_workspace,
)
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, UnknownKeyError
from dtk_engine.workspace import WorkspaceNotFoundError

KEY_IDS = [k["id"] for k in list_keys()]


def test_keys_registered():
    assert KEY_IDS


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_schema_is_dict(key_id):
    assert isinstance(key_schema(key_id), dict)


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_run_defaults_is_json(key_id):
    assert json.dumps(run_key(key_id, {}))


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_bad_params_raise(key_id):
    with pytest.raises(KeyParamsError):
        run_key(key_id, {"__not_a_param__": object()})


def test_unknown_key_raises():
    with pytest.raises(UnknownKeyError):
        run_key("does-not-exist", {})
    with pytest.raises(UnknownKeyError):
        key_schema("does-not-exist")


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_unknown_param_rejected(key_id):
    with pytest.raises(KeyParamsError):
        run_key(key_id, {"not_a_param": 1})


def _workspace(name="w"):
    return {
        "name": name,
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": TRAIN_CSV},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": TEST_CSV}},
        },
    }


def test_workspace_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    assert list_workspaces() == []
    saved = save_workspace(_workspace())
    assert saved["label"] == {"mode": "order", "key": None} and saved["steps"] == []
    assert json.dumps(saved)
    assert get_workspace("w") == saved
    assert [w["name"] for w in list_workspaces()] == ["w"]
    assert (tmp_path / "workspaces" / "w.json").exists()
    res = run_key("dataset_overview", {"source": {"kind": "dataset", "workspace": "w"}})
    assert res["metrics"]["rows"] == 41
    delete_workspace("w")
    assert list_workspaces() == []
    with pytest.raises(WorkspaceNotFoundError):
        get_workspace("w")


def test_save_invalid_workspace_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    with pytest.raises(KeyParamsError):
        save_workspace({"name": "w"})
    with pytest.raises(KeyParamsError):
        save_workspace(_workspace("../escape"))
