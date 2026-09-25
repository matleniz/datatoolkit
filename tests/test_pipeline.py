import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from dtk_engine import DtkTransformer, save_workspace, workspace_pipeline
from dtk_engine import transform_registry as registry
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError, UnknownTransformError
from dtk_engine.transform_registry import TransformParams, transform


@pytest.fixture
def center(monkeypatch):
    """A stat op: subtract the fitted mean of every column."""
    monkeypatch.setattr(registry, "_TRANSFORMS", dict(registry._TRANSFORMS))

    @transform(
        "center_all",
        params_model=TransformParams,
        fit=lambda df, p: {c: float(df[c].mean()) for c in df.columns},
    )
    def center_all(df, params, state):
        return df - pd.Series(state)


def _frame(n=40):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "name": ["a"] * n})
    return df, (x > 0).astype(int)


def test_fit_on_train_transform_other(center):
    t = DtkTransformer("center_all").fit(pd.DataFrame({"v": [0.0, 2.0]}))
    assert t.state_ == {"v": 1.0}
    assert t.transform(pd.DataFrame({"v": [5.0]}))["v"].tolist() == [4.0]
    assert t.get_feature_names_out().tolist() == ["v"]


def test_pipeline_cross_val_score(center):
    df, y = _frame()
    pipe = Pipeline(
        [
            ("drop", DtkTransformer("drop_columns", columns=["name"])),
            ("center", DtkTransformer("center_all")),
            ("model", LogisticRegression()),
        ]
    )
    scores = cross_val_score(pipe, df, y, cv=4)
    assert len(scores) == 4 and scores.mean() > 0.8


def test_params_are_sklearn_params():
    t = DtkTransformer("drop_columns", columns=["a"])
    assert t.get_params() == {"op": "drop_columns", "columns": ["a"]}
    c = clone(t).set_params(columns=["b"], missing_ok=True)
    assert c.params == {"columns": ["b"], "missing_ok": True}
    assert t.params == {"columns": ["a"]}
    assert "drop_columns" in repr(t)


def test_set_output_pandas_and_numpy_input():
    t = DtkTransformer("drop_columns", columns=["x1"]).set_output(transform="pandas")
    out = t.fit_transform(np.zeros((2, 3)))
    assert isinstance(out, pd.DataFrame) and out.columns.tolist() == ["x0", "x2"]


def test_invalid_params_at_fit():
    with pytest.raises(KeyParamsError):
        DtkTransformer("drop_columns", colums=["a"]).fit(pd.DataFrame({"a": [1]}))
    with pytest.raises(UnknownTransformError):
        DtkTransformer("nope").fit(pd.DataFrame({"a": [1]}))


def test_workspace_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    ws = {
        "name": "w",
        "datasets": {"train": {"x": {"kind": "csv", "path": TRAIN_CSV}}},
        "steps": [
            {"op": "drop_columns", "target": "train", "params": {"columns": ["Age"]}},
            {"op": "drop_columns", "target": "both", "params": {"columns": ["Name"]}},
        ],
    }
    save_workspace(ws)
    pipe = workspace_pipeline("w")
    assert [name for name, _ in pipe.steps] == ["1_drop_columns"]
    out = pipe.fit_transform(pd.DataFrame({"Name": ["x"], "Age": [1]}))
    assert out.columns.tolist() == ["Age"]

    ws["steps"] = []
    save_workspace(ws)
    df = pd.DataFrame({"a": [1]})
    assert workspace_pipeline("w").fit_transform(df).equals(df)

    ws["steps"] = [{"op": "drop_columns", "target": "both", "params": {}}]
    save_workspace(ws)
    with pytest.raises(KeyParamsError):
        workspace_pipeline("w")
