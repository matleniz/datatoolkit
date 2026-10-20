import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from dtk_engine import DtkTransformer, save_workspace, workspace_pipeline
from dtk_engine import transform_registry as registry
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError, UnknownTransformError
from dtk_engine.transform_registry import TransformParams, transform
from dtk_engine.workspace import JsonWorkspaceStore, Workspace


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


def _cat_frame():
    return pd.DataFrame({"c": ["a", "b", "a", "b"], "n": [1.0, 2.0, 3.0, 4.0]})


def test_set_output_pandas_pipeline_names_from_onehot():
    pipe = Pipeline([("enc", DtkTransformer("onehot", columns=["c"]))])
    pipe.set_output(transform="pandas")
    out = pipe.fit_transform(_cat_frame())
    assert isinstance(out, pd.DataFrame)
    assert out.columns.tolist() == pipe[-1].get_feature_names_out().tolist()
    assert out.columns.tolist() != ["c", "n"]  # the op renamed / added columns
    assert pipe.transform(_cat_frame()).columns.tolist() == out.columns.tolist()
    assert pipe.get_feature_names_out().tolist() == out.columns.tolist()


def test_set_output_pandas_column_transformer_names():
    df = _cat_frame().assign(n2=[5.0, 6.0, 7.0, 8.0])
    ct = ColumnTransformer(
        [
            ("cat", DtkTransformer("onehot", columns=["c"]), ["c"]),
            ("keep", DtkTransformer("drop_columns", columns=["n2"]), ["n", "n2"]),
        ],
        remainder="drop",
    ).set_output(transform="pandas")
    out = ct.fit_transform(df)
    assert isinstance(out, pd.DataFrame)
    cat_names = ct.named_transformers_["cat"].get_feature_names_out().tolist()
    assert out.columns.tolist() == [f"cat__{n}" for n in cat_names] + ["keep__n"]
    assert ct.get_feature_names_out().tolist() == out.columns.tolist()
    assert ct.transform(df).columns.tolist() == out.columns.tolist()


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
    with pytest.raises(KeyParamsError):  # rejected at save
        save_workspace(ws)
    # a workspace file written behind the contract's back still fails early
    JsonWorkspaceStore().save(Workspace.model_validate(ws))
    with pytest.raises(KeyParamsError):
        workspace_pipeline("w")


def test_advised_workspace_pipeline_in_cross_val_score(tmp_path, monkeypatch):
    """End to end on the demo data: advisor plan -> workspace -> sklearn CV."""
    from dtk_engine import api
    from dtk_engine.ops.advisor import advise, as_steps

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    train = api.load(TRAIN_CSV)
    recs, _ = advise(train, model_family="linear", target="Survived")
    steps = as_steps(recs)
    assert {s["target"] for s in steps} == {"train", "both"}  # drop_duplicates: train
    save_workspace(
        {
            "name": "advised",
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": TRAIN_CSV},
                    "target_column": "Survived",
                }
            },
            "steps": steps,
        }
    )
    pipe = Pipeline(
        [
            ("prep", workspace_pipeline("advised")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    X, y = train.drop(columns="Survived"), train["Survived"]
    scores = cross_val_score(pipe, X, y, cv=3)
    assert len(scores) == 3 and np.isfinite(scores).all()


def test_set_params_op_change_drops_old_op_params():
    t = DtkTransformer("drop_columns", columns=["a"], missing_ok=True)
    t.set_params(op="scale")
    assert t.get_params() == {"op": "scale", "columns": ["a"]}
    t.set_params(op="scale", method="minmax")  # same op: params merge
    assert t.params == {"columns": ["a"], "method": "minmax"}
    t.set_params(op="nope")
    assert t.get_params() == {"op": "nope"}
    # a fixed op still clones / round-trips as before
    c = clone(DtkTransformer("drop_columns", columns=["a"], missing_ok=True))
    assert c.get_params() == {
        "op": "drop_columns",
        "columns": ["a"],
        "missing_ok": True,
    }


def test_param_named_op_and_name_round_trip():
    """derive's params ``op`` and ``name`` clash with nothing (#253)."""
    t = DtkTransformer("derive", a="x", b="y", op="product", name="xy")
    assert t.op == "derive" and t.params["op"] == "product" and t.params["name"] == "xy"
    assert t.get_params() == {
        "op": "derive", "a": "x", "b": "y", "param_op": "product", "name": "xy",
    }
    c = clone(t)
    assert c.op == "derive" and c.params == t.params and c.get_params() == t.get_params()
    c.set_params(param_op="difference", name="d")
    assert c.params["op"] == "difference" and c.params["name"] == "d" and c.op == "derive"
    assert t.params["op"] == "product"  # the original is untouched
    c.set_params(op="scale")  # a plain ``op`` is still the transform identity
    assert c.op == "scale"


def test_pipeline_with_derive_fits_and_clones():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    pipe = Pipeline([("d", DtkTransformer("derive", a="x", b="y", op="product", name="xy"))])
    out = clone(pipe).fit_transform(df)
    assert out["xy"].tolist() == [4.0, 10.0, 18.0]
