import pandas as pd
import pytest

from dtk_engine import run_key, save_workspace
from dtk_engine import transform_registry as registry
from dtk_engine.errors import KeyParamsError, SourceError, UnknownTransformError
from dtk_engine.sources import DatasetSource, load
from dtk_engine.transform_registry import TransformParams, transform


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
    monkeypatch.setattr(registry, "_TRANSFORMS", {})

    def fit(df, params):
        return {"mean": float(df["f"].mean())}

    @transform("center_f", params_model=TransformParams, fit=fit)
    def center_f(df, params, state):
        return df.assign(f=df["f"] - state["mean"])

    _save(files, steps=[{"op": "center_f", "target": "both"}])
    assert load(DatasetSource(workspace="w"))["f"].tolist() == [-1.0, 0.0, 1.0]
    # test is centered with the train mean (2), not its own (15)
    test = load(DatasetSource(workspace="w", role="test"))
    assert test["f"].tolist() == [8.0, 18.0]


def test_unknown_step_op(files):
    # rejected at save, not deferred to the first load
    with pytest.raises(UnknownTransformError, match="unknown transform op 'nope'"):
        _save(files, steps=[{"op": "nope", "target": "test"}])


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


def test_version_none_replays_every_step(files, monkeypatch):
    monkeypatch.setattr(registry, "_TRANSFORMS", {})

    def fit(df, params):
        return {"mean": float(df["f"].mean())}

    @transform("center_f", params_model=TransformParams, fit=fit)
    def center_f(df, params, state):
        return df.assign(f=df["f"] - state["mean"])

    _save(files, steps=[{"op": "center_f", "target": "both"}])
    assert load(DatasetSource(workspace="w"))["f"].tolist() == [-1.0, 0.0, 1.0]
    assert load(DatasetSource(workspace="w", version=None))["f"].tolist() == [
        -1.0,
        0.0,
        1.0,
    ]


def test_version_replays_only_first_n_steps(files, monkeypatch):
    monkeypatch.setattr(registry, "_TRANSFORMS", {})

    def fit(df, params):
        return {"mean": float(df["f"].mean())}

    @transform("center_f", params_model=TransformParams, fit=fit)
    def center_f(df, params, state):
        return df.assign(f=df["f"] - state["mean"])

    @transform("double_f", params_model=TransformParams)
    def double_f(df, params, state):
        return df.assign(f=df["f"] * 2)

    _save(
        files,
        steps=[
            {"op": "center_f", "target": "both"},
            {"op": "double_f", "target": "both"},
        ],
    )
    assert load(DatasetSource(workspace="w", version=0))["f"].tolist() == [1, 2, 3]
    assert load(DatasetSource(workspace="w", version=1))["f"].tolist() == [
        -1.0,
        0.0,
        1.0,
    ]
    assert load(DatasetSource(workspace="w", version=2))["f"].tolist() == [
        -2.0,
        0.0,
        2.0,
    ]


def test_version_beyond_step_count_raises(files):
    _save(
        files,
        steps=[{"op": "drop_columns", "target": "both", "params": {"columns": ["id"]}}],
    )
    with pytest.raises(KeyParamsError, match="version 2 exceeds workspace step count"):
        load(DatasetSource(workspace="w", version=2))


def test_version_negative_raises(files):
    _save(files)
    with pytest.raises(KeyParamsError, match="version must be a non-negative int"):
        load(DatasetSource(workspace="w", version=-1))
