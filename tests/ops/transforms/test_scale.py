import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from dtk_engine import DtkTransformer
from dtk_engine.transform_registry import get_transform

SKLEARN = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
}


def _fit(op, df, **params):
    t = get_transform(op)
    p = t.parse(params)
    return t, p, t.fit(df, p)


@pytest.mark.parametrize("method", sorted(SKLEARN))
def test_scale_matches_sklearn_with_train_stats(method):
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(5, 3, 40), "b": rng.exponential(2, 40)})
    test = pd.DataFrame({"a": [-100.0, 5.0, 100.0], "b": [0.0, 1.0, 50.0]})
    t, p, state = _fit("scale", train, columns=["a", "b"], method=method)
    json.dumps(state)
    out = t.apply(test, p, state)
    ref = SKLEARN[method]().fit(train).transform(test)
    assert np.allclose(out.to_numpy(), ref)


def test_minmax_not_clipped_and_constant_column():
    train = pd.DataFrame({"x": [0.0, 10.0], "k": [3, 3]})
    t, p, state = _fit("scale", train, columns=["x", "k"], method="minmax")
    out = t.apply(pd.DataFrame({"x": [-5.0, 20.0], "k": [3, 4]}), p, state)
    assert out["x"].tolist() == [-0.5, 2.0]  # beyond the train range, kept
    assert out["k"].tolist() == [0.0, 1.0]  # zero range -> scale 1


def test_scale_ignores_missing_and_refuses_text():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "s": ["a", "b", "c"]})
    t, p, state = _fit("scale", train, columns=["x"])
    assert state["x"]["center"] == 2.0
    assert np.isnan(t.apply(train, p, state)["x"].iloc[1])
    with pytest.raises(ValueError, match="numeric"):
        _fit("scale", train, columns=["s"])


def test_log1p():
    df = pd.DataFrame({"x": [0, 1, np.e - 1]})
    t, p, state = _fit("log1p", df, columns=["x"])
    assert np.allclose(t.apply(df, p, state)["x"], [0, np.log(2), 1])
    with pytest.raises(ValueError, match="negative"):
        t.apply(pd.DataFrame({"x": [1, -2]}), p, state)


def test_chain_in_pipeline_cross_val_score():
    rng = np.random.default_rng(0)
    n = 120
    income = rng.exponential(30_000, n)
    size = rng.choice(["S", "M", "L"], n)
    color = rng.choice(["red", "blue", "green"], n)
    y = ((income > 20_000) | (size == "L")).astype(int)
    X = pd.DataFrame({"income": income, "size": size, "color": color})
    X.loc[::9, "income"] = np.nan
    X.loc[::11, "color"] = None
    pipe = Pipeline(
        [
            (
                "impute",
                DtkTransformer("impute", columns=["income"], add_indicator=True),
            ),
            ("log", DtkTransformer("log1p", columns=["income"])),
            ("scale", DtkTransformer("scale", columns=["income"], method="robust")),
            (
                "impute_c",
                DtkTransformer("impute", columns=["color"], strategy="constant"),
            ),
            ("onehot", DtkTransformer("onehot", columns=["color"], min_frequency=5)),
            (
                "ordinal",
                DtkTransformer("ordinal", categories={"size": ["S", "M", "L"]}),
            ),
            ("model", LogisticRegression()),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=4)
    assert len(scores) == 4 and scores.mean() > 0.8
    pipe.fit(X, y)
    names = pipe[:-1].get_feature_names_out().tolist()
    assert names[:3] == ["income", "income_was_missing", "size"]
    assert "color_MISSING" in names
