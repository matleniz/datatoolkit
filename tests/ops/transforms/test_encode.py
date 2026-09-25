import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from dtk_engine.errors import KeyParamsError
from dtk_engine.transform_registry import get_transform


def _fit(op, df, **params):
    t = get_transform(op)
    p = t.parse(params)
    return t, p, t.fit(df, p)


def test_onehot_vocabulary_from_train_unknown_ignored():
    train = pd.DataFrame({"id": [1, 2, 3], "color": ["red", "blue", "red"]})
    test = pd.DataFrame({"id": [4, 5, 6], "color": ["green", "blue", None]})
    t, p, state = _fit("onehot", train, columns=["color"])
    json.dumps(state)
    assert state == {"color": {"categories": ["blue", "red"], "infrequent": []}}
    out = t.apply(test, p, state)
    assert out.columns.tolist() == ["id", "color_blue", "color_red"]
    # green (unknown) and None -> all zeros; no test-only column (no leak).
    assert out[["color_blue", "color_red"]].values.tolist() == [[0, 0], [1, 0], [0, 0]]


def test_onehot_stable_names_whatever_row_order():
    df = pd.DataFrame({"c": ["b", "a", "c", "a"]})
    _, _, s1 = _fit("onehot", df, columns=["c"])
    _, _, s2 = _fit("onehot", df.iloc[::-1], columns=["c"])
    assert s1 == s2


def test_onehot_matches_sklearn():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {"c": rng.choice(list("abcde"), 60, p=[0.4, 0.3, 0.2, 0.07, 0.03])}
    )
    test = pd.DataFrame({"c": list("abcdez")})
    t, p, state = _fit("onehot", train, columns=["c"], min_frequency=5)
    ref = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False
    ).fit(train)
    out = t.apply(test, p, state)
    expected = ref.transform(test[:-1])  # sklearn maps unknown z to infrequent
    assert np.array_equal(out.to_numpy()[:-1], expected)
    assert out.iloc[-1].sum() == 0  # we ignore unknown values
    assert out.columns[-1] == "c_infrequent"


def test_onehot_min_frequency_fraction_and_drop_first():
    train = pd.DataFrame({"c": ["a"] * 8 + ["b"] + ["c"]})
    t, p, state = _fit("onehot", train, columns=["c"], min_frequency=0.15)
    assert state["c"] == {"categories": ["a"], "infrequent": ["b", "c"]}
    t, p, state = _fit("onehot", train, columns=["c"], drop_first=True)
    assert t.apply(train, p, state).columns.tolist() == ["c_b", "c_c"]
    with pytest.raises(KeyParamsError):
        get_transform("onehot").parse({"columns": ["c"], "min_frequency": 1.5})
    with pytest.raises(KeyParamsError):
        get_transform("onehot").parse({"columns": ["c"], "min_frequency": 0})


def test_onehot_unknown_error_and_collisions():
    train = pd.DataFrame({"c": ["a", "b"], "c_a": [0, 0]})
    t, p, state = _fit("onehot", train, columns=["c"], handle_unknown="error")
    with pytest.raises(ValueError, match="unknown categories"):
        t.apply(pd.DataFrame({"c": ["z"], "c_a": [0]}), p, state)
    with pytest.raises(ValueError, match="collide"):
        t.apply(train, p, state)


def test_ordinal_explicit_order_unknown_minus_one():
    df = pd.DataFrame({"size": ["M", "S", "XL", "L", None]})
    t, p, state = _fit("ordinal", df, categories={"size": ["S", "M", "L"]})
    assert state == {}
    out = t.apply(df, p, state)
    assert out["size"].tolist()[:4] == [1, 0, -1, 2]
    assert np.isnan(out["size"].iloc[4])  # missing stays missing
    full = t.apply(df.iloc[:4], p, state)
    assert full["size"].dtype.kind == "i"


def test_ordinal_params_required_and_unique():
    t = get_transform("ordinal")
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["size"]})  # no alphabetical default
    with pytest.raises(KeyParamsError):
        t.parse({"categories": {"size": []}})
    with pytest.raises(KeyParamsError):
        t.parse({"categories": {"size": ["S", "S"]}})
