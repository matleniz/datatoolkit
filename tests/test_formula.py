"""Tests for the formula transform op (safe AST whitelist, train-fitted vars)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from dtk_engine import transform_schema
from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform


def _parse(**params):
    return get_transform("formula").parse(params)


def _run(df, **params):
    t = get_transform("formula")
    return t.fit_apply(df, t.parse(params))


def _fit_apply(train, test, **params):
    t = get_transform("formula")
    p = t.parse(params)
    return t.apply(test, p, t.fit(train, p))


def test_arithmetic_and_precedence():
    df = pd.DataFrame({"a": [2.0, 10.0], "b": [3.0, 4.0], "c": [1.0, 2.0]})
    out = _run(df, name="y", expr="a + b * c")
    assert out["y"].tolist() == [5.0, 18.0]
    out = _run(df, name="y", expr="(a + b) * c")
    assert out["y"].tolist() == [5.0, 28.0]
    out = _run(df, name="y", expr="a ** 2 - b")
    assert out["y"].tolist() == [1.0, 96.0]
    out = _run(df, name="y", expr="-a + +b")
    assert out["y"].tolist() == [1.0, -6.0]
    assert df.columns.tolist() == ["a", "b", "c"]  # input untouched


def test_functions():
    df = pd.DataFrame({"x": [1.0, 4.0, np.e - 1]})
    out = _run(df, name="y", expr="sqrt(x) + abs(-x) + round(log1p(x))")
    assert out["y"].iloc[0] == pytest.approx(1.0 + 1.0 + round(np.log1p(1.0)))
    assert out["y"].iloc[1] == pytest.approx(2.0 + 4.0 + round(np.log1p(4.0)))
    out = _run(df, name="y", expr="min(x, 2) + max(x, 0)")
    x2 = np.e - 1
    assert out["y"].tolist() == pytest.approx(
        [1.0 + 1.0, 2.0 + 4.0, min(x2, 2) + max(x2, 0)]
    )
    out = _run(pd.DataFrame({"x": [1.0]}), name="y", expr="exp(log(x))")
    assert out["y"].iloc[0] == pytest.approx(1.0)


def test_variables_fitted_on_train_reused_on_test():
    train = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
    test = pd.DataFrame({"x": [100.0, 200.0]})  # different stats must not change value
    params = {
        "name": "z",
        "expr": "x - @mu",
        "variables": [{"name": "mu", "stat": "mean", "column": "x"}],
    }
    t = get_transform("formula")
    p = t.parse(params)
    state = t.fit(train, p)
    assert state == {"variables": {"mu": 3.0}}
    out = t.apply(test, p, state)
    assert out["z"].tolist() == [97.0, 197.0]
    # Refitting on test would give mu=150; frozen state must win.
    refit = t.fit(test, p)
    assert refit["variables"]["mu"] == 150.0
    assert t.apply(test, p, state)["z"].tolist() == [97.0, 197.0]


def test_nan_propagation_and_divide_by_zero():
    df = pd.DataFrame({"a": [1.0, np.nan, 4.0], "b": [2.0, 1.0, 0.0]})
    out = _run(df, name="y", expr="a / b")
    assert out["y"].iloc[0] == 0.5
    assert np.isnan(out["y"].iloc[1])
    assert np.isnan(out["y"].iloc[2])
    out = _run(df, name="y", expr="a / 1e-13")  # |denom| < 1e-12 -> NaN
    assert np.isnan(out["y"].iloc[0])
    out = _run(pd.DataFrame({"a": [-1.0]}), name="y", expr="sqrt(a)")
    assert np.isnan(out["y"].iloc[0])
    assert out["y"].dtype == float


@pytest.mark.parametrize(
    "expr",
    [
        "a.__class__",
        "a[0]",
        "(lambda x: x)(a)",
        "round(a, ndigits=1)",
        '"hello"',
        "a > 1",
        "__import__('os')",
        "eval(a)",
        "[a]",
        "a if a else b",
        "a and b",
        "a ^ b",
    ],
)
def test_refuses_disallowed_constructs(expr):
    with pytest.raises(KeyParamsError, match="formula"):
        _parse(name="y", expr=expr)


def test_unknown_function_and_unknown_variable_at_params():
    with pytest.raises(KeyParamsError, match="unknown function"):
        _parse(name="y", expr="sin(a)")
    with pytest.raises(KeyParamsError, match="@variable"):
        _parse(name="y", expr="a + @mu")


def test_unknown_column_at_apply():
    t = get_transform("formula")
    p = t.parse({"name": "y", "expr": "missing_col + 1"})
    with pytest.raises(KeyParamsError, match="unknown column 'missing_col'"):
        t.fit_apply(pd.DataFrame({"a": [1.0]}), p)


def test_dtk_transformer_and_sklearn_pipeline_parity():
    train = pd.DataFrame({"a": [2.0, 4.0], "b": [1.0, 1.0]})
    test = pd.DataFrame({"a": [10.0], "b": [2.0]})
    params = {
        "name": "y",
        "expr": "(a - @mu) / b",
        "variables": [{"name": "mu", "stat": "mean", "column": "a"}],
    }
    direct = _fit_apply(train, test, **params)
    tr = DtkTransformer("formula", **params).fit(train)
    assert tr.state_ == {"variables": {"mu": 3.0}}
    via_tr = tr.transform(test)
    assert via_tr["y"].tolist() == direct["y"].tolist() == [3.5]

    pipe = Pipeline([("f", DtkTransformer("formula", **params))])
    pipe.fit(train)
    assert pipe.transform(test)["y"].tolist() == [3.5]


def test_json_schema_has_fields():
    schema = transform_schema("formula")
    props = schema["properties"]
    assert set(props) >= {"name", "expr", "variables"}
    assert "name" in schema["required"] and "expr" in schema["required"]
    var = props["variables"]["items"]
    # Pydantic may inline or $ref the nested model.
    if "$ref" in var:
        var = schema["$defs"][var["$ref"].rsplit("/", 1)[-1]]
    assert set(var["properties"]) >= {"name", "stat", "column"}
    assert set(var["properties"]["stat"]["enum"]) == {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "q25",
        "q75",
        "count",
    }
    # Fitted state must be JSON-serializable.
    t = get_transform("formula")
    p = t.parse(
        {
            "name": "y",
            "expr": "@n",
            "variables": [{"name": "n", "stat": "count", "column": "a"}],
        }
    )
    state = t.fit(pd.DataFrame({"a": [1.0, np.nan, 2.0]}), p)
    json.dumps(state)
    assert state["variables"]["n"] == 2.0


def test_overwrite_existing_column():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = _run(df, name="a", expr="a * 2")
    assert out["a"].tolist() == [2.0, 4.0]
