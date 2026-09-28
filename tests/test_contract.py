import json

import pytest

from dtk_engine import (
    delete_workspace,
    get_workspace,
    key_schema,
    list_keys,
    list_transforms,
    list_workspaces,
    preview_workspace,
    run_key,
    save_workspace,
    transform_schema,
)
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, UnknownKeyError, UnknownTransformError
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


def test_list_keys_needs_target():
    keys = {k["id"]: k for k in list_keys()}
    assert json.dumps(keys)
    assert keys["feature_selection"]["needs_target"] is True
    assert keys["target_analysis"]["needs_target"] is True
    # An optional target does not make the key supervised.
    assert keys["preprocessing_advisor"]["needs_target"] is False
    assert keys["dataset_overview"]["needs_target"] is False


TARGET_KEYS = [k for k in KEY_IDS if "target" in key_schema(k)["properties"]]
# Params that make a key actually read its target.
READS_TARGET = {"column_distribution": {"by_label": True}}


@pytest.mark.parametrize("key_id", TARGET_KEYS)
def test_unknown_target_raises_params_error(key_id):
    with pytest.raises(KeyParamsError, match="nope"):
        run_key(key_id, {"target": "nope", **READS_TARGET.get(key_id, {})})


STEP_TABLES = {
    "preprocessing_advisor": {"recommendations"},
    "feature_selection": {"suggested_steps"},
    "correlations": {"suggested_steps"},
    "missing_values": {"suggested_steps"},
}


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_steps_tables_are_tagged(key_id):
    tables = run_key(key_id, {})["tables"]
    steps = {t["title"] for t in tables if t["kind"] == "steps"}
    assert steps == STEP_TABLES.get(key_id, set())
    for t in tables:
        if t["kind"] == "steps":
            assert all({"op", "target", "params"} <= set(r) for r in t["records"])


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


def test_list_transforms_is_json():
    transforms = list_transforms()
    assert json.dumps(transforms)
    assert {"op", "title", "description", "needs_target"} == set(transforms[0])
    by_op = {t["op"]: t for t in transforms}
    assert by_op["select_k_best"]["needs_target"] is True
    assert by_op["drop_columns"]["needs_target"] is False


def test_transform_schema():
    schema = transform_schema("drop_columns")
    assert json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert "columns" in schema["required"]
    with pytest.raises(UnknownTransformError):
        transform_schema("does-not-exist")


def _snapshot(root):
    return sorted((str(p), p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_preview_workspace_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    save_workspace(
        _workspace("w.preview")
    )  # a real workspace with the old scratch name
    before = _snapshot(tmp_path)
    ws = _workspace("w")
    ws["steps"] = [
        {"op": "drop_columns", "target": "both", "params": {"columns": ["Name"]}}
    ]
    out = preview_workspace(ws, "train", head_rows=3)
    assert json.dumps(out)
    assert out["shape"] == [41, len(out["columns"])]
    assert "Name" not in out["columns"] and len(out["head"]) == 3
    assert _snapshot(tmp_path) == before
    assert [w["name"] for w in list_workspaces()] == ["w.preview"]


def test_preview_workspace_impute_both_uses_train_median(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DTK_HOME", str(home))
    (tmp_path / "train.csv").write_text("Age,id\n10,1\n20,2\n30,3\n")  # median 20
    (tmp_path / "test.csv").write_text("Age,id\n100,1\n,2\n300,3\n")  # own median 200
    ws = {
        "name": "w",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(tmp_path / "train.csv")}},
            "test": {"x": {"kind": "csv", "path": str(tmp_path / "test.csv")}},
        },
        "steps": [
            {
                "op": "impute",
                "target": "both",
                "params": {"columns": ["Age"], "strategy": "median"},
            }
        ],
    }
    out = preview_workspace(ws, "test")
    assert out["shape"] == [3, 2]
    assert [r["Age"] for r in out["head"]] == [100, 20, 300]
    assert not home.exists()


def test_preview_workspace_errors():
    with pytest.raises(KeyParamsError):
        preview_workspace({"name": "w"}, "train")
    with pytest.raises(KeyParamsError):
        preview_workspace(_workspace(), "val")


@pytest.mark.parametrize(
    ("step", "error"),
    [
        ({"op": "nope", "target": "both"}, UnknownTransformError),
        ({"op": "drop_columns", "target": "both", "params": {}}, KeyParamsError),
        (
            {"op": "drop_columns", "target": "train", "params": {"colums": ["a"]}},
            KeyParamsError,
        ),
    ],
)
def test_save_and_preview_reject_invalid_steps(tmp_path, monkeypatch, step, error):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    ws = _workspace("w") | {"steps": [step]}
    with pytest.raises(error):
        save_workspace(ws)
    assert list_workspaces() == []
    with pytest.raises(error):
        preview_workspace(ws, "train")
