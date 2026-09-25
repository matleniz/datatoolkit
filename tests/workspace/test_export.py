import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dtk_engine import api, export_workspace, save_workspace
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.sources import load
from dtk_engine.sources.spec import DatasetSource
from dtk_engine.workspace import WorkspaceNotFoundError
from dtk_engine.workspace.export import export_workspace as export
from dtk_engine.workspace.export import load_state

STEPS = [
    {"op": "drop_duplicates", "target": "train", "params": {"keep": "none"}},
    {
        "op": "replace_sentinels",
        "target": "test",
        "params": {"sentinels": {"Age": ["unknown"]}},
    },
    {"op": "cast", "target": "test", "params": {"dtypes": {"Age": "float64"}}},
    {
        "op": "drop_columns",
        "target": "both",
        "params": {"columns": ["Name", "Ticket", "Cabin"]},
    },
    {"op": "impute_knn", "target": "both", "params": {"columns": ["Age", "Fare"]}},
    {"op": "onehot", "target": "both", "params": {"columns": ["Sex", "Embarked"]}},
    {"op": "scale", "target": "both", "params": {"columns": ["Fare"]}},
]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    save_workspace(
        {
            "name": "demo",
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": TRAIN_CSV},
                    "target_column": "Survived",
                },
                "test": {"x": {"kind": "csv", "path": TEST_CSV}},
            },
            "steps": STEPS,
        }
    )
    return "demo"


def test_round_trip_equals_replay(workspace, tmp_path):
    raw = {p: _sha(p) for p in (TRAIN_CSV, TEST_CSV)}
    out = tmp_path / "export"
    returned = api.export_workspace(workspace, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == returned

    for role in ("train", "test"):
        replayed = load(DatasetSource(workspace=workspace, role=role))
        written = pd.read_parquet(out / "processed" / f"{role}.parquet")
        pd.testing.assert_frame_equal(written, replayed.reset_index(drop=True))
        assert manifest["outputs"][role]["rows"] == len(replayed)
        assert manifest["outputs"][role]["sha256"] == _sha(
            out / manifest["outputs"][role]["path"]
        )
    assert {p: _sha(p) for p in raw} == raw  # raw inputs untouched

    src = {s["role"]: s for s in manifest["sources"]}
    assert src["train"]["sha256"] == raw[TRAIN_CSV]
    assert src["train"]["size"] == Path(TRAIN_CSV).stat().st_size
    assert src["train"]["mtime"] and src["test"]["path"] == str(
        Path(TEST_CSV).resolve()
    )
    assert set(manifest["versions"]) >= {"dtk_engine", "pandas", "sklearn"}
    assert manifest["exported_at"]

    steps = manifest["steps"]
    assert [s["op"] for s in steps] == [s["op"] for s in STEPS]
    assert steps[2]["fitted_on"] == "test" and steps[4]["fitted_on"] == "train"
    assert set(steps[5]["state"]) == {"Sex", "Embarked"}  # small state: inline


def test_big_state_goes_to_side_file(workspace, tmp_path):
    out = tmp_path / "export"
    manifest = export_workspace(workspace, str(out))
    knn = manifest["steps"][4]
    assert "state" in knn  # 40 rows: small enough to inline

    # Tiny inline limit: the impute_knn state (train matrix) goes to a side file.
    manifest = export(workspace, out, overwrite=True, inline_state_bytes=256)
    knn = manifest["steps"][4]
    assert "state" not in knn and knn["state_file"] == "states/step_4_impute_knn.json"
    state = load_state(out, knn)
    assert state["columns"] == ["Age", "Fare"] and len(state["train"]) == 39
    assert knn["state_sha256"] == _sha(out / knn["state_file"])

    # Overwrite removes the files the previous export listed.
    export(workspace, out, overwrite=True)
    assert not (out / knn["state_file"]).exists()


def test_existing_export_needs_overwrite(workspace, tmp_path):
    export_workspace(workspace, str(tmp_path))
    with pytest.raises(KeyParamsError, match="overwrite"):
        export_workspace(workspace, str(tmp_path))
    assert (
        export_workspace(workspace, str(tmp_path), overwrite=True)["workspace"]
        == "demo"
    )


def _tamper(out, mutate):
    manifest = json.loads((out / "manifest.json").read_text())
    mutate(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("bad", ["../victim.csv", "processed/../../victim.csv", "abs"])
def test_overwrite_refuses_unsafe_manifest_paths(workspace, tmp_path, bad):
    out = tmp_path / "export"
    export_workspace(workspace, str(out))
    victim = tmp_path / "victim.csv"
    victim.write_text("raw")
    bad = str(victim) if bad == "abs" else bad
    _tamper(out, lambda m: m["outputs"]["train"].update(path=bad))
    with pytest.raises(KeyParamsError, match="outside|non-relative"):
        export(workspace, out, overwrite=True)
    assert victim.exists() and (out / "processed" / "train.parquet").exists()


def test_overwrite_refuses_foreign_manifest(workspace, tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    keep = out / "processed" / "train.parquet"
    keep.parent.mkdir()
    keep.write_text("mine")
    (out / "manifest.json").write_text(
        json.dumps({"outputs": {"train": {"path": "processed/train.parquet"}}})
    )
    with pytest.raises(KeyParamsError, match="not written by"):
        export(workspace, out, overwrite=True)
    assert keep.exists()


def test_refuses_to_overwrite_a_raw_input(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    raw = tmp_path / "processed" / "train.parquet"
    raw.parent.mkdir()
    pd.DataFrame({"a": [1, 2]}).to_parquet(raw)
    save_workspace(
        {
            "name": "w",
            "datasets": {"train": {"x": {"kind": "parquet", "path": str(raw)}}},
        }
    )
    with pytest.raises(KeyParamsError, match="raw input"):
        export_workspace("w", str(tmp_path))
    manifest = export_workspace("w", str(tmp_path / "elsewhere"))
    assert "test" not in manifest["outputs"] and manifest["steps"] == []


def test_unknown_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    with pytest.raises(WorkspaceNotFoundError):
        export_workspace("nope", str(tmp_path))
