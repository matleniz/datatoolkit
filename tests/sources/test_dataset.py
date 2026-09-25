import pandas as pd
import pytest

from dtk_engine import run_key, save_workspace
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.sources import DatasetSource, load
from dtk_engine.workspace import replay as replay_mod
from dtk_engine.workspace.replay import transform


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    x_train = tmp_path / "x_train.csv"
    x_train.write_text("Index,id,f\n0,a,1\n1,b,2\n2,c,3\n")
    y_train = tmp_path / "y_train.csv"
    y_train.write_text("Index,target\n0,0.5\n1,1.5\n2,2.5\n")
    x_test = tmp_path / "x_test.csv"
    x_test.write_text("Index,id,f\n0,d,10\n1,e,20\n")
    return {"x_train": str(x_train), "y_train": str(y_train), "x_test": str(x_test)}


def _save(files, name="w", **kw):
    ws = {
        "name": name,
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": files["x_train"]},
                "y": {"kind": "csv", "path": files["y_train"]},
            },
            "test": {"x": {"kind": "csv", "path": files["x_test"]}},
        },
    }
    return save_workspace(ws | kw)


def test_train_labeled_and_unlabeled(files):
    _save(files)
    df = load(DatasetSource(workspace="w"))
    assert list(df.columns) == ["Index", "id", "f", "target"]
    assert df["target"].tolist() == [0.5, 1.5, 2.5]
    assert "target" not in load(DatasetSource(workspace="w", labeled=False))


def test_test_role(files):
    _save(files)
    assert load(DatasetSource(workspace="w", role="test")).shape == (2, 3)


def test_key_join(files):
    y = pd.DataFrame({"id": ["c", "a", "b"], "target": [3, 1, 2]})
    y.to_csv(files["y_train"], index=False)
    _save(files, label={"mode": "key", "key": "id"})
    assert load(DatasetSource(workspace="w"))["target"].tolist() == [1, 2, 3]


def test_join_error_is_source_error(files):
    pd.DataFrame({"target": [1, 2]}).to_csv(files["y_train"], index=False)
    _save(files)
    with pytest.raises(SourceError, match="X has 3 rows, y has 2 rows"):
        load(DatasetSource(workspace="w"))


def test_target_column_in_x(files):
    ws = _save(files)
    ws["datasets"]["train"] = {
        "x": {"kind": "csv", "path": files["x_train"]},
        "target_column": "f",
    }
    save_workspace(ws)
    assert "f" in load(DatasetSource(workspace="w"))
    assert "f" not in load(DatasetSource(workspace="w", labeled=False))
    ws["datasets"]["train"]["target_column"] = "nope"
    save_workspace(ws)
    with pytest.raises(SourceError, match="target column 'nope' not in X"):
        load(DatasetSource(workspace="w"))


def test_missing_workspace_or_role(files):
    with pytest.raises(SourceError, match="workspace not found"):
        load(DatasetSource(workspace="ghost"))
    ws = _save(files)
    ws["datasets"]["test"] = None
    save_workspace(ws)
    with pytest.raises(SourceError, match="has no test dataset"):
        load(DatasetSource(workspace="w", role="test"))


def test_steps_replayed_per_role(files, monkeypatch):
    monkeypatch.setattr(replay_mod, "_TRANSFORMS", {})

    @transform("center_f")
    def center_f(df, params, fit):
        ref = df if fit is None else fit
        return df.assign(f=df["f"] - ref["f"].mean())

    _save(files, steps=[{"op": "center_f", "target": "both"}])
    assert load(DatasetSource(workspace="w"))["f"].tolist() == [-1.0, 0.0, 1.0]
    # test is centered with the train mean (2), not its own (15)
    test = load(DatasetSource(workspace="w", role="test"))
    assert test["f"].tolist() == [8.0, 18.0]


def test_unknown_step_op(files):
    _save(files, steps=[{"op": "nope", "target": "test"}])
    with pytest.raises(SourceError, match="unknown transform op 'nope'"):
        load(DatasetSource(workspace="w", role="test"))


def test_keys_run_on_dataset_source(files):
    _save(files)
    res = run_key(
        "train_test_check",
        {
            "train": {"kind": "dataset", "workspace": "w", "role": "train"},
            "test": {"kind": "dataset", "workspace": "w", "role": "test"},
        },
    )
    assert res["metrics"]["n_only_train"] == 1  # target
    overview = run_key(
        "dataset_overview", {"source": {"kind": "dataset", "workspace": "w"}}
    )
    assert overview["metrics"]["rows"] == 3


def test_dataset_spec_strict():
    with pytest.raises(KeyParamsError):
        run_key(
            "dataset_overview",
            {"source": {"kind": "dataset", "workspace": "w", "x": 1}},
        )
    with pytest.raises(KeyParamsError):
        run_key("dataset_overview", {"source": {"kind": "dataset", "role": "valid"}})
