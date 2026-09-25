import json

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from dtk_engine import DtkTransformer
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay, replay_fitted


def _fit(op, df, **params):
    t = get_transform(op)
    p = t.parse(params)
    return t, p, t.fit(df, p)


def planted(n=300, seed=0, regression=False):
    """informative a, b; noise n1, n2; dup ~ 2a; const; a text column; target y."""
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "n1": rng.normal(size=n),
            "b": b,
            "dup": 2 * a + rng.normal(scale=0.01, size=n),
            "n2": rng.normal(size=n),
            "const": np.ones(n),
            "name": ["x"] * n,
        }
    )
    signal = 2 * a + b
    df["y"] = signal + rng.normal(scale=0.1, size=n) if regression else (signal > 0)
    df["y"] = df["y"].astype(float if regression else int)
    return df


def test_drop_low_variance_keeps_non_candidates():
    df = planted()
    t, p, state = _fit("drop_low_variance", df, target="y")
    assert state["dropped"] == ["const"]
    out = t.apply(df, p, state)
    assert "const" not in out and {"name", "y", "a"} <= set(out.columns)
    json.dumps(state)


def test_drop_correlated_order_and_target():
    df = planted()
    _, _, state = _fit("drop_correlated", df.drop(columns="const"))
    assert state["dropped"] == ["dup"]  # first in column order kept
    # With a target, the more target-correlated of the pair is kept.
    df2 = df[["dup", "a", "y"]].copy()
    df2["dup"] = df2["a"] + np.random.default_rng(1).normal(scale=0.3, size=len(df))
    _, _, state = _fit("drop_correlated", df2, target="y", threshold=0.9)
    assert state["selected"] == ["a"] and state["dropped"] == ["dup"]


@pytest.mark.parametrize("score", ["mutual_info", "f_test"])
@pytest.mark.parametrize("regression", [False, True])
def test_select_k_best_finds_informative(score, regression):
    df = planted(regression=regression).drop(columns=["dup", "const"])
    t, p, state = _fit("select_k_best", df, target="y", score=score, k=2)
    assert sorted(state["selected"]) == ["a", "b"]
    assert state["task"] == ("regression" if regression else "classification")
    assert t.apply(df, p, state).columns.tolist() == ["a", "b", "name", "y"]
    json.dumps(state)


def test_select_k_best_percentile_and_params():
    df = planted().drop(columns=["dup", "const"])
    _, _, state = _fit("select_k_best", df, target="y", percentile=50)
    assert len(state["selected"]) == 2
    _, _, state = _fit("select_k_best", df, target="y", k=99)
    assert state["dropped"] == []
    for bad in ({"target": "y"}, {"target": "y", "k": 1, "percentile": 10}):
        with pytest.raises(KeyParamsError):
            get_transform("select_k_best").parse(bad)
    with pytest.raises(ValueError, match="cannot be a candidate"):
        _fit("select_k_best", df, target="y", k=1, columns=["a", "y"])


@pytest.mark.parametrize("model", ["l1", "tree"])
@pytest.mark.parametrize("regression", [False, True])
def test_select_from_model_finds_informative(model, regression):
    df = planted(regression=regression).drop(columns=["dup", "const"])
    _, _, state = _fit(
        "select_from_model",
        df,
        target="y",
        model=model,
        max_features=2,
        threshold="median",
    )
    assert sorted(state["selected"]) == ["a", "b"]
    assert len(state["importance"]) == 4
    json.dumps(state)


def test_select_from_model_multiclass_l1():
    df = planted().drop(columns=["dup", "const"])
    df["y"] = np.digitize(2 * df["a"] + df["b"], [-1, 1])
    _, _, state = _fit("select_from_model", df, target="y", model="l1", max_features=2)
    assert sorted(state["selected"]) == ["a", "b"]


def test_missing_values_raise_clearly():
    df = planted().drop(columns=["dup", "const"])
    df.loc[3, "n1"] = np.nan
    for op, params in [
        ("select_k_best", {"target": "y", "k": 1}),
        ("select_from_model", {"target": "y"}),
        ("pca", {"target": "y"}),
    ]:
        with pytest.raises(ValueError, match="impute"):
            _fit(op, df, **params)


def test_pca_matches_sklearn():
    df = planted().drop(columns=["name", "y"])
    t, p, state = _fit("pca", df, n_components=3, standardize=False)
    out = t.apply(df, p, state)
    assert out.columns.tolist() == ["pc1", "pc2", "pc3"]
    ref = PCA(n_components=3, svd_solver="full").fit(df.to_numpy())
    assert np.allclose(out.to_numpy(), ref.transform(df.to_numpy()))
    json.dumps(state)

    t, p, state = _fit("pca", df, n_components=2, standardize=False, whiten=True)
    ref = PCA(n_components=2, whiten=True, svd_solver="full").fit(df.to_numpy())
    assert np.allclose(t.apply(df, p, state).to_numpy(), ref.transform(df.to_numpy()))


def test_pca_variance_fraction_and_placement():
    df = planted()
    t, p, state = _fit("pca", df, n_components=0.95, target="y")
    ratio = state["explained_variance_ratio"]
    assert sum(ratio) >= 0.95 and sum(ratio[:-1]) < 0.95
    out = t.apply(df, p, state)
    names = [f"pc{i + 1}" for i in range(len(ratio))]
    assert out.columns.tolist() == [*names, "name", "y"]  # target never selected
    # dup ~ 2a and const add no dimension: 5 numeric inputs, fewer components.
    assert len(ratio) < 6


def test_pca_standardize_uses_train_stats():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"u": rng.normal(0, 1, 100), "v": rng.normal(0, 1000, 100)})
    test = train.iloc[:10] + 5.0
    tr = DtkTransformer("pca", n_components=2).fit(train)
    Z = (train - train.mean()) / train.std(ddof=0)
    ref = PCA(n_components=2, svd_solver="full").fit(Z.to_numpy())
    Zt = (test - train.mean()) / train.std(ddof=0)
    assert np.allclose(tr.transform(test).to_numpy(), ref.transform(Zt.to_numpy()))
    with pytest.raises(KeyParamsError):
        get_transform("pca").parse({"n_components": 1.5})


def test_selection_fitted_on_train_only():
    train = planted(seed=0).drop(columns=["dup", "const"])
    test = planted(n=50, seed=7).drop(columns=["dup", "const", "y"])
    # Make n1 look informative on test only: must not change train's selection.
    test["n1"] = test["a"] * 10
    tr = DtkTransformer("select_k_best", target="y", k=2).fit(train)
    out = tr.transform(test)
    assert out.columns.tolist() == ["a", "b", "name"]

    steps = [Step(op="select_k_best", target="both", params={"target": "y", "k": 2})]
    assert replay(steps, "test", test, train).columns.tolist() == ["a", "b", "name"]
    tr_out, te_out, fitted = replay_fitted(steps, train, test)
    assert tr_out.columns.tolist() == ["a", "b", "name", "y"]
    assert te_out.columns.tolist() == ["a", "b", "name"]
    assert fitted[0]["fitted_on"] == "train"


def test_replay_both_chain_with_pca():
    train = planted(seed=0).drop(columns=["name"])
    test = planted(n=40, seed=3).drop(columns=["name", "y"])
    steps = [
        Step(op="drop_low_variance", target="both", params={"target": "y"}),
        Step(op="drop_correlated", target="both", params={"target": "y"}),
        Step(op="pca", target="both", params={"n_components": 2, "target": "y"}),
    ]
    out = replay(steps, "test", test, train)
    assert out.columns.tolist() == ["pc1", "pc2"]
    assert replay(steps, "train", train).columns.tolist() == ["pc1", "pc2", "y"]

    bad = [Step(op="select_k_best", target="both", params={"target": "zz", "k": 1})]
    with pytest.raises(SourceError, match="step 0"):
        replay(bad, "test", test, train)


def test_supervised_selector_in_pipeline_cross_val_score():
    df = planted().drop(columns=["name", "const", "dup"])
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    pipe = Pipeline(
        [
            ("select", DtkTransformer("select_k_best", target="y", k=2)),
            ("model", LogisticRegression()),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=5)
    assert scores.mean() > 0.9
    fitted = pipe.fit(X, y).named_steps["select"]
    assert sorted(fitted.get_feature_names_out().tolist()) == ["a", "b"]
    assert "y" not in fitted.feature_names_in_  # y joined for fit only

    with pytest.raises(KeyError, match="pass y"):
        DtkTransformer("select_k_best", target="y", k=2).fit(X)


def test_needs_target_flag():
    assert get_transform("select_k_best").needs_target
    assert get_transform("drop_correlated").needs_target
    assert not get_transform("pca").needs_target
    assert not get_transform("scale").needs_target
