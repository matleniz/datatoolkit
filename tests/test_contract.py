import json
from pathlib import Path

import pytest

from dtk_engine import (
    contract,
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


def test_rename_duplicate_and_summaries(tmp_path, monkeypatch):
    from dtk_engine import (
        duplicate_workspace,
        list_workspace_summaries,
        rename_workspace,
    )

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    contract._SHAPES.clear()
    save_workspace(
        _workspace("alpha")
        | {
            "steps": [
                {
                    "op": "drop_columns",
                    "target": "both",
                    "params": {"columns": ["Name"]},
                }
            ]
        }
    )
    save_workspace(_workspace("beta"))

    summaries = list_workspace_summaries()
    assert json.dumps(summaries)
    assert [s["name"] for s in summaries] == ["alpha", "beta"]
    alpha = summaries[0]
    assert alpha["step_count"] == 1
    assert alpha["target"] == "Survived"
    assert "+00:00" in alpha["mtime"] or alpha["mtime"].endswith("Z")
    assert alpha["train"]["file"] == Path(TRAIN_CSV).name
    assert alpha["train"]["shape"] == [41, 11]  # Name dropped
    assert alpha["test"]["file"] == Path(TEST_CSV).name
    assert alpha["test"]["shape"] == [20, 10]

    # Cached shapes: no frame load on a second summaries call (MAT-200 / MAT-204).
    def _boom(*_a, **_k):
        raise AssertionError("workspace_frame must not run when shape is cached")

    monkeypatch.setattr("dtk_engine.contract.workspace_frame", _boom)
    monkeypatch.setattr("dtk_engine.workspace.dataset.workspace_frame", _boom)
    again = list_workspace_summaries()
    assert again[0]["train"]["shape"] == [41, 11]
    assert again[0]["test"]["shape"] == [20, 10]

    dup = duplicate_workspace("alpha", "alpha-copy")
    assert dup["name"] == "alpha-copy"
    assert dup["steps"] == get_workspace("alpha")["steps"]
    assert (
        dup["datasets"]["train"]["x"]["path"]
        == get_workspace("alpha")["datasets"]["train"]["x"]["path"]
    )

    renamed = rename_workspace("alpha-copy", "alpha-renamed")
    assert renamed["name"] == "alpha-renamed"
    with pytest.raises(WorkspaceNotFoundError):
        get_workspace("alpha-copy")
    # Rename/dupe share content keys — shapes still cached, no frame load.
    assert [s["name"] for s in list_workspace_summaries()] == [
        "alpha",
        "alpha-renamed",
        "beta",
    ]

    delete_workspace("alpha")
    delete_workspace("beta")
    assert [s["name"] for s in list_workspace_summaries()] == ["alpha-renamed"]


def test_summaries_shape_cache_many_workspaces(tmp_path, monkeypatch):
    """Warm once, then many workspaces stay fast; step changes refresh shape."""
    from dtk_engine import list_workspace_summaries
    from dtk_engine.workspace.dataset import workspace_frame as real_frame

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    contract._SHAPES.clear()
    for i in range(12):
        save_workspace(_workspace(f"ws{i:02d}"))

    first = list_workspace_summaries()
    assert len(first) == 12
    assert all(s["train"]["shape"] == [41, 12] for s in first)
    assert all(s["test"]["shape"] == [20, 11] for s in first)

    calls = {"n": 0}

    def _counting(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("workspace_frame must not run when shapes are cached")

    monkeypatch.setattr("dtk_engine.contract.workspace_frame", _counting)
    monkeypatch.setattr("dtk_engine.workspace.dataset.workspace_frame", _counting)
    second = list_workspace_summaries()
    assert calls["n"] == 0
    assert [s["train"]["shape"] for s in second] == [[41, 12]] * 12

    # Restore frame loader so a step change can recompute.
    monkeypatch.setattr("dtk_engine.contract.workspace_frame", real_frame)
    monkeypatch.setattr("dtk_engine.workspace.dataset.workspace_frame", real_frame)
    save_workspace(
        _workspace("ws00")
        | {
            "steps": [
                {
                    "op": "drop_columns",
                    "target": "both",
                    "params": {"columns": ["Name"]},
                }
            ]
        }
    )
    after = {s["name"]: s for s in list_workspace_summaries()}
    assert after["ws00"]["train"]["shape"] == [41, 11]
    assert after["ws00"]["test"]["shape"] == [20, 10]
    assert after["ws01"]["train"]["shape"] == [41, 12]

    monkeypatch.setattr("dtk_engine.contract.workspace_frame", _counting)
    monkeypatch.setattr("dtk_engine.workspace.dataset.workspace_frame", _counting)
    calls["n"] = 0
    assert {s["name"]: s["train"]["shape"] for s in list_workspace_summaries()}[
        "ws00"
    ] == [41, 11]
    assert calls["n"] == 0


def test_rename_duplicate_errors(tmp_path, monkeypatch):
    from dtk_engine import duplicate_workspace, rename_workspace

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    save_workspace(_workspace("a"))
    save_workspace(_workspace("b"))
    with pytest.raises(KeyParamsError, match="already exists"):
        rename_workspace("a", "b")
    with pytest.raises(KeyParamsError, match="already exists"):
        duplicate_workspace("a", "b")
    with pytest.raises(WorkspaceNotFoundError):
        rename_workspace("missing", "x")
    with pytest.raises(WorkspaceNotFoundError):
        duplicate_workspace("missing", "x")
    with pytest.raises(KeyParamsError, match="invalid"):
        rename_workspace("a", "../x")


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
