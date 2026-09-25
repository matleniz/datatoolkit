import json

import numpy as np
import pandas as pd
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from dtk_engine import DtkTransformer
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def _fit(op, df, **params):
    t = get_transform(op)
    p = t.parse(params)
    return t, p, t.fit(df, p)


def test_impute_median_uses_train_stats_on_test():
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 10.0]})
    test = pd.DataFrame({"x": [np.nan, 100.0, np.nan]})
    t, p, state = _fit("impute", train, columns=["x"])
    assert state == {"fill": {"x": 2.0}}
    assert t.apply(test, p, state)["x"].tolist() == [2.0, 100.0, 2.0]  # no leak
    assert test["x"].isna().sum() == 2  # input untouched


@pytest.mark.parametrize("strategy", ["median", "mean", "most_frequent"])
def test_impute_matches_simple_imputer(strategy):
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.integers(0, 5, 50).astype(float)})
    train.loc[::7, "a"] = np.nan
    test = pd.DataFrame({"a": [np.nan, 3.0, np.nan]})
    t, p, state = _fit("impute", train, columns=["a"], strategy=strategy)
    ref = SimpleImputer(strategy=strategy).fit(train[["a"]])
    assert np.allclose(t.apply(test, p, state)[["a"]], ref.transform(test[["a"]]))


def test_impute_categorical_and_constant():
    train = pd.DataFrame({"c": ["a", "b", "b", None], "n": [1, None, 3, 4]})
    t, p, state = _fit("impute", train, columns=["c", "n"], strategy="constant")
    assert state == {"fill": {"c": "MISSING", "n": 0}}
    out = t.apply(train, p, state)
    assert out["c"].tolist() == ["a", "b", "b", "MISSING"]
    assert out["n"].tolist() == [1, 0, 3, 4]

    t, p, state = _fit("impute", train, columns=["c"], strategy="most_frequent")
    assert state == {"fill": {"c": "b"}}

    t, p, state = _fit(
        "impute", train, columns=["c"], strategy="constant", fill_value="?"
    )
    assert t.apply(train, p, state)["c"].iloc[3] == "?"


def test_impute_errors():
    df = pd.DataFrame({"c": ["a", None], "n": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="numeric"):
        _fit("impute", df, columns=["c"], strategy="median")
    with pytest.raises(ValueError, match="entirely missing"):
        _fit("impute", df, columns=["n"])
    with pytest.raises(ValueError, match="string fill"):
        _fit("impute", df, columns=["n"], strategy="constant", fill_value="x")
    with pytest.raises(KeyParamsError):
        get_transform("impute").parse({"columns": ["n"], "strategy": "max"})


def test_impute_indicator():
    train = pd.DataFrame({"x": [1.0, np.nan], "y": [0, 0]})
    test = pd.DataFrame({"x": [5.0, 6.0], "y": [0, 0]})
    t, p, state = _fit("impute", train, columns=["x"], add_indicator=True)
    out = t.apply(train, p, state)
    assert out.columns.tolist() == ["x", "x_was_missing", "y"]
    assert out["x_was_missing"].tolist() == [0, 1]
    # Same columns on test even with nothing missing there.
    assert t.apply(test, p, state).columns.tolist() == out.columns.tolist()
    with pytest.raises(ValueError, match="already exists"):
        t.apply(out, p, state)


def _knn_frames():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(30, 3)), columns=["a", "b", "c"])
    train.iloc[::4, 1] = np.nan
    test = pd.DataFrame(rng.normal(size=(5, 3)), columns=["a", "b", "c"])
    test.iloc[[0, 2], 0] = np.nan
    return train, test


def test_impute_knn_matches_sklearn_and_state_is_json():
    train, test = _knn_frames()
    t, p, state = _fit("impute_knn", train, columns=["a", "b", "c"], n_neighbors=3)
    json.dumps(state, allow_nan=False)  # strict JSON: missing stored as null
    out = t.apply(test, p, state)
    ref = KNNImputer(n_neighbors=3).fit(train).transform(test)
    assert np.allclose(out.to_numpy(), ref)
    assert not out.isna().any().any()


def test_impute_iterative_matches_sklearn_and_is_deterministic():
    train, test = _knn_frames()
    train["c"] = train["a"] * 2 + 1  # c is predictable from a
    t, p, state = _fit("impute_iterative", train, columns=["a", "b", "c"], max_iter=5)
    json.dumps(state, allow_nan=False)
    out = t.apply(test, p, state)
    ref = IterativeImputer(max_iter=5, random_state=0).fit(train).transform(test)
    assert np.allclose(out.to_numpy(), ref)
    assert out.equals(t.apply(test, p, state))


def test_model_imputers_refuse_text():
    df = pd.DataFrame({"a": [1.0, np.nan], "s": ["x", "y"]})
    for op in ("impute_knn", "impute_iterative"):
        with pytest.raises(ValueError, match="numeric"):
            _fit(op, df, columns=["a", "s"])


def test_ffill_in_sort_order_never_backward():
    df = pd.DataFrame(
        {"t": [3, 1, 2, 4, 0], "v": [np.nan, 1.0, np.nan, np.nan, np.nan]},
        index=[7, 7, 8, 9, 9],  # duplicated index labels
    )
    t, p, state = _fit("ffill", df, sort_by="t")
    out = t.apply(df, p, state)
    # t=0 has nothing before it: stays missing (no backward fill).
    assert out["v"].tolist()[:4] == [1.0, 1.0, 1.0, 1.0]
    assert np.isnan(out["v"].iloc[4])
    assert out.index.tolist() == df.index.tolist()

    t, p, state = _fit("ffill", df, sort_by="t", limit=1)
    assert t.apply(df, p, state)["v"].isna().tolist() == [
        True,
        False,
        False,
        True,
        True,
    ]


def test_ffill_params():
    with pytest.raises(KeyParamsError):
        get_transform("ffill").parse({"columns": ["v"]})  # sort_by required
    with pytest.raises(KeyParamsError):
        get_transform("ffill").parse({"sort_by": "t", "columns": ["t"]})
    df = pd.DataFrame({"t": [1, None], "v": [1, None]})
    t = get_transform("ffill")
    with pytest.raises(ValueError, match="sort column"):
        t.fit_apply(df, t.parse({"sort_by": "t"}))


def test_impute_replay_both_and_transformer():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan]})
    steps = [Step(op="impute", target="both", params={"columns": ["x"]})]
    assert replay(steps, "test", test, train)["x"].tolist() == [2.0]
    tr = DtkTransformer("impute", columns=["x"], strategy="mean").fit(train)
    assert tr.transform(test)["x"].tolist() == [2.0]

    bad = [Step(op="impute", target="both", params={"columns": ["zz"]})]
    with pytest.raises(SourceError, match="step 0"):
        replay(bad, "test", test, train)
