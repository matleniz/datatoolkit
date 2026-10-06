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


def test_sin_cos_and_pi():
    df = pd.DataFrame({"theta": [0.0, np.pi / 2, np.pi]})
    out = _run(df, name="y", expr="sin(theta)")
    assert out["y"].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-10)
    out = _run(df, name="y", expr="cos(theta)")
    assert out["y"].tolist() == pytest.approx([1.0, 0.0, -1.0], abs=1e-10)
    out = _run(df, name="y", expr="sin(pi / 2)")
    assert out["y"].tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-10)
    # pi is a constant, not a column lookup.
    out = _run(pd.DataFrame({"a": [1.0]}), name="y", expr="cos(0) + pi - pi")
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
    ("expr", "message"),
    [
        ("a.__class__", "dunder attribute access \\(__class__\\)"),
        ("a[0]", 'subscript is only allowed as df\\["column"\\]'),
        ("(lambda x: x)(a)", "calling the result of an expression"),
        ("round(a, ndigits=1)", "keyword arguments is not allowed"),
        ('"hello"', "text constant 'hello'"),
        ("__import__('os')", "unknown function __import__\\(\\)"),
        ("eval(a)", "unknown function eval\\(\\)"),
        ("[a]", "a list literal is not allowed"),
        ("a ^ b", "\\^ \\(use \\*\\* for powers\\) is not allowed"),
        ("a is b", "'is' is not allowed"),
        ("a in b", "'in' is not allowed"),
    ],
)
def test_refuses_disallowed_constructs(expr, message):
    with pytest.raises(KeyParamsError, match=f"formula: {message}"):
        _parse(name="y", expr=expr)


def test_comparisons_and_where():
    df = pd.DataFrame({"Age": [10.0, 18.0, 25.0, np.nan]})
    out = _run(df, name="is_child", expr="where(Age < 18, 1, 0)")
    assert out["is_child"].tolist()[:3] == pytest.approx([1.0, 0.0, 0.0])
    assert np.isnan(out["is_child"].iloc[3])
    # Bare comparison -> 0/1.
    out = _run(df, name="adult", expr="Age >= 18")
    assert out["adult"].tolist()[:3] == pytest.approx([0.0, 1.0, 1.0])
    assert np.isnan(out["adult"].iloc[3])
    out = _run(df, name="mid", expr="10 < Age < 20")
    assert out["mid"].tolist()[:3] == pytest.approx([0.0, 1.0, 0.0])


def test_new_math_functions():
    df = pd.DataFrame({"x": [-1.5, 0.0, 4.0, np.nan]})
    out = _run(df, name="y", expr="floor(x)")
    assert out["y"].iloc[0] == -2.0 and out["y"].iloc[2] == 4.0
    assert np.isnan(out["y"].iloc[3])
    out = _run(df, name="y", expr="ceil(x)")
    assert out["y"].iloc[0] == -1.0 and out["y"].iloc[1] == 0.0
    out = _run(df, name="y", expr="sign(x)")
    assert out["y"].tolist()[:3] == pytest.approx([-1.0, 0.0, 1.0])
    out = _run(df, name="y", expr="square(x)")
    assert out["y"].tolist()[:3] == pytest.approx([2.25, 0.0, 16.0])
    out = _run(df, name="y", expr="clip(x, 0, 2)")
    assert out["y"].tolist()[:3] == pytest.approx([0.0, 0.0, 2.0])
    out = _run(df, name="y", expr="isnull(x)")
    assert out["y"].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
    out = _run(pd.DataFrame({"x": [1.0, 4.0]}), name="y", expr="log2(x) + log10(x)")
    assert out["y"].iloc[0] == pytest.approx(np.log2(1.0) + np.log10(1.0))
    assert out["y"].iloc[1] == pytest.approx(np.log2(4.0) + np.log10(4.0))
    out = _run(pd.DataFrame({"x": [0.0]}), name="y", expr="tanh(x)")
    assert out["y"].iloc[0] == pytest.approx(0.0)


def test_new_functions_arity():
    with pytest.raises(KeyParamsError, match="where\\(\\) takes 3"):
        _parse(name="y", expr="where(a)")
    with pytest.raises(KeyParamsError, match="clip\\(\\) takes 3"):
        _parse(name="y", expr="clip(a, 0)")
    with pytest.raises(KeyParamsError, match="square\\(\\) takes 1"):
        _parse(name="y", expr="square(a, b)")


def test_unknown_function_and_unknown_variable_at_params():
    with pytest.raises(KeyParamsError, match="unknown function"):
        _parse(name="y", expr="tan(a)")
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


def test_round_with_integer_decimals_and_nan():
    df = pd.DataFrame({"a": [1.23, np.nan, 4.56]})
    out = _run(df, name="y", expr="round(a, 1)")
    assert out["y"].iloc[0] == pytest.approx(1.2)
    assert np.isnan(out["y"].iloc[1])
    assert out["y"].iloc[2] == pytest.approx(4.6)


def test_round_refuses_non_constant_or_non_integer_decimals():
    with pytest.raises(
        KeyParamsError, match="round\\(\\) second argument must be an integer constant"
    ):
        _parse(name="y", expr="round(a, b)")
    with pytest.raises(
        KeyParamsError, match="round\\(\\) second argument must be an integer constant"
    ):
        _parse(name="y", expr="round(a, 1.5)")


# --- MAT-241: Python-flavoured syntax, mapped onto the same whitelist ---------

_PY = pd.DataFrame(
    {
        "a": [2.0, -3.5, 7.0, 0.0],
        "b": [3.0, 2.0, -2.0, 5.0],
        "Age": [10.0, 18.0, 40.0, 70.0],
        "Nom col": [1.0, 2.0, 3.0, 4.0],
    }
)
_A = _PY["a"].to_numpy()
_B = _PY["b"].to_numpy()
_AGE = _PY["Age"].to_numpy()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        # numpy namespace -> whitelist
        ("np.log1p(np.abs(a))", np.log1p(np.abs(_A))),
        ("numpy.sqrt(b ** 2)", np.sqrt(_B**2)),
        ("np.log2(Age) + np.log10(Age) + np.log(Age)",
         np.log2(_AGE) + np.log10(_AGE) + np.log(_AGE)),
        ("np.exp(a / 10)", np.exp(_A / 10)),
        ("np.round(a / 3, 2)", np.round(_A / 3, 2)),
        ("np.floor(a) + np.ceil(b)", np.floor(_A) + np.ceil(_B)),
        ("np.sign(a) * np.square(b)", np.sign(_A) * np.square(_B)),
        ("np.sin(a) + np.cos(b) + np.tanh(a)",
         np.sin(_A) + np.cos(_B) + np.tanh(_A)),
        ("np.clip(a, -1, 3)", np.clip(_A, -1, 3)),
        ("np.where(a > 0, a, b)", np.where(_A > 0, _A, _B)),
        ("np.minimum(a, b) + np.maximum(a, b)",
         np.minimum(_A, _B) + np.maximum(_A, _B)),
        ("np.isnan(a)", np.zeros(4)),
        ("np.pi * a", np.pi * _A),
        ("numpy.pi", np.full(4, np.pi)),
        # conditional expression -> where (nested)
        ("a if a > b else b", np.where(_A > _B, _A, _B)),
        ("0 if Age < 18 else (1 if Age < 65 else 2)",
         np.where(_AGE < 18, 0.0, np.where(_AGE < 65, 1.0, 2.0))),
        # boolean logic, element-wise -> 0/1
        ("Age >= 18 and a > 0", ((_AGE >= 18) & (_A > 0)).astype(float)),
        ("Age < 18 or Age > 65", ((_AGE < 18) | (_AGE > 65)).astype(float)),
        ("not a", (_A == 0).astype(float)),
        ("(a > 0) & (b > 0)", ((_A > 0) & (_B > 0)).astype(float)),
        ("(a > 0) | (b < 0)", ((_A > 0) | (_B < 0)).astype(float)),
        ("~(a > 0)", (~(_A > 0)).astype(float)),
        ("18 <= Age < 65", ((_AGE >= 18) & (_AGE < 65)).astype(float)),
        ("a > 0 and b > 0 and Age > 30",
         ((_A > 0) & (_B > 0) & (_AGE > 30)).astype(float)),
        # arithmetic operators
        ("a ** 2", _A**2),
        ("Age % 7", np.mod(_AGE, 7)),
        ("a % b", np.mod(_A, _B)),
        ("a // b", np.floor_divide(_A, _B)),
        # builtins, element-wise
        ("abs(a)", np.abs(_A)),
        ("round(a / 3, 1)", np.round(_A / 3, 1)),
        ("min(a, b)", np.minimum(_A, _B)),
        ("max(a, b, 0)", np.maximum(np.maximum(_A, _B), 0)),
        # column access
        ('df["Nom col"] * 2', _PY["Nom col"].to_numpy() * 2),
        ("df.Age - df['a']", _AGE - _A),
        ('np.log(df["Age"]) if df.a > 0 else -1',
         np.where(_A > 0, np.log(_AGE), -1.0)),
    ],
)
def test_python_style_accepted(expr, expected):
    out = _run(_PY, name="y", expr=expr)
    assert out["y"].to_numpy() == pytest.approx(np.asarray(expected, dtype=float))


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("(lambda x: x + 1)", "lambda is not allowed in a formula"),
        ("import os", "import is not allowed in a formula"),
        ("from os import path", "import is not allowed in a formula"),
        ("a.__class__", r"dunder attribute access \(__class__\)"),
        ("df.__class__", r"dunder attribute access \(__class__\)"),
        ("np.__dict__", r"dunder attribute access \(__dict__\)"),
        ("a.apply(f)", r"method calls like \.apply\(\.\.\.\) are not allowed"),
        ('df["a"].fillna(0)', r"method calls like \.fillna\(\.\.\.\) are not allowed"),
        ("[x for x in a]", "a comprehension is not allowed in a formula"),
        ("min(x for x in a)", "a comprehension is not allowed in a formula"),
        ("df.eval('a + b')", r"method calls like df\.eval\(\.\.\.\) are not allowed"),
        ("getattr(a, 'b')", r"unknown function getattr\(\) is not allowed"),
        ("np.random.rand(3)", r"np\.random\.rand is not allowed in a formula"),
        ("np.random", r"np\.random is not allowed in a formula"),
        ("np.e", r"np\.e is not allowed in a formula"),
        ("np.log", r"np\.log must be called, e\.g\. np\.log\(x\)"),
        ("np.mean(a)", r"np\.mean is not allowed in a formula"),
        ("torch.sin(a)", r"method calls like \.sin\(\.\.\.\) are not allowed"),
        ("torch.x", r"only allowed as np\.<function> or df\.<column> \(got torch\.x\)"),
        ("open('f')", r"unknown function open\(\) is not allowed"),
        ("a.b", r"attribute access is only allowed as np\.<function> or df\.<column>"),
        ("df.a.b", r"attribute access is only allowed as np\.<function>"),
        ("a[0]", r'subscript is only allowed as df\["column"\]'),
        ("df[0]", r'df\[\.\.\.\] needs a column name in quotes'),
        ("df[a]", r'df\[\.\.\.\] needs a column name in quotes'),
        ("df['a':'b']", r'df\[\.\.\.\] needs a column name in quotes'),
        ("log(a)(b)", "calling the result of an expression is not allowed"),
        ("np.log(a).real", "attribute access is only allowed"),
        ("__builtins__", "dunder name __builtins__ is not allowed"),
        ("(x := a)", r"assignment \(:=\) is not allowed"),
        ("f'{a}'", "an f-string is not allowed"),
        ("True", "constant True is not allowed"),
        ("np.minimum(a)", r"np\.minimum\(\) takes 2 arguments, got 1"),
        ("np.round(a, decimals=2)", "keyword arguments is not allowed"),
        ("a @ b", "@ .*is not allowed"),
    ],
)
def test_python_style_refused_with_named_message(expr, message):
    with pytest.raises(KeyParamsError, match=message):
        _parse(name="y", expr=expr)


@pytest.mark.parametrize(
    "expr",
    [
        # classic sandbox escapes
        "().__class__.__bases__[0].__subclasses__()",
        "a.__class__.__mro__",
        "df.__init__.__globals__",
        "np.__builtins__",
        "np.log.__self__",
        "log.__globals__",
        "(1).__class__",
        '"".__class__',
        # subscript on anything but df
        "np['log']",
        "a['x']",
        "(a + b)['x']",
        '{"a": 1}["a"]',
        # calls on the result of a call / attribute chains
        "log(a)(b)",
        "where(a, b, a)()",
        "np.log(a).__class__",
        "df.a.apply(log)",
        "getattr(np, 'log')(a)",
        "vars()",
        "globals()",
        "exec('1')",
        "compile('1', '', 'eval')",
    ],
)
def test_security_escapes_refused(expr):
    with pytest.raises(KeyParamsError, match="formula: "):
        _parse(name="y", expr=expr)


def test_df_column_access_edge_cases():
    df = pd.DataFrame({"pi": [2.0], "log": [3.0], "a@b": [5.0], "x": [7.0]})
    # df["pi"] / df.log are columns, not the constant / function.
    out = _run(df, name="y", expr='df["pi"] + df.log + pi * 0')
    assert out["y"].iloc[0] == pytest.approx(5.0)
    # "@" inside a column name is not a variable.
    out = _run(df, name="y", expr='df["a@b"] + 1')
    assert out["y"].iloc[0] == 6.0
    # Bare name and df.x read the same column.
    out = _run(df, name="y", expr='x - df.x + df["x"]')
    assert out["y"].iloc[0] == 7.0
    with pytest.raises(KeyParamsError, match="unknown column 'Nope col'"):
        _run(df, name="y", expr='df["Nope col"] + 1')
    with pytest.raises(KeyParamsError, match="unknown column 'nope'"):
        _run(df, name="y", expr="df.nope")


def test_python_style_nan_semantics():
    df = pd.DataFrame({"a": [1.0, np.nan, 0.0], "b": [0.0, 1.0, 1.0]})
    out = _run(df, name="y", expr="a and b")
    assert out["y"].iloc[0] == 0.0 and np.isnan(out["y"].iloc[1])
    out = _run(df, name="y", expr="not a")
    assert out["y"].iloc[2] == 1.0 and np.isnan(out["y"].iloc[1])
    out = _run(df, name="y", expr="1 if a > 0 else 2")
    assert out["y"].iloc[0] == 1.0 and np.isnan(out["y"].iloc[1])
    # % and // by ~0 -> NaN, like /.
    out = _run(df, name="y", expr="a % b")
    assert np.isnan(out["y"].iloc[0])
    out = _run(df, name="y", expr="a // b")
    assert np.isnan(out["y"].iloc[0])


def test_python_style_equals_legacy_form():
    """Python and mini-language spellings give identical results."""
    df = pd.DataFrame({"Age": [5.0, 30.0, np.nan, 80.0], "Fare": [1.0, 0.0, 3.0, 9.0]})
    pairs = [
        ("where(Age < 18, 1, 0)", "1 if Age < 18 else 0"),
        ("log1p(Fare)", "np.log1p(df.Fare)"),
        ("isnull(Age)", "np.isnan(Age)"),
        ("min(Age, Fare)", "np.minimum(Age, Fare)"),
        ("sin(pi / 2) * Age", "np.sin(np.pi / 2) * df['Age']"),
    ]
    for legacy, python in pairs:
        a = _run(df, name="y", expr=legacy)["y"].to_numpy()
        b = _run(df, name="y", expr=python)["y"].to_numpy()
        np.testing.assert_array_equal(a, b)


def test_python_style_variables_and_pipeline_replay():
    train = pd.DataFrame({"Age": [10.0, 20.0, 30.0]})
    test = pd.DataFrame({"Age": [5.0, 25.0]})
    params = {
        "name": "y",
        "expr": "1 if df['Age'] > @mu and not np.isnan(Age) else 0",
        "variables": [{"name": "mu", "stat": "mean", "column": "Age"}],
    }
    assert _fit_apply(train, test, **params)["y"].tolist() == [0.0, 1.0]
    pipe = Pipeline([("f", DtkTransformer("formula", **params))]).fit(train)
    assert pipe.transform(test)["y"].tolist() == [0.0, 1.0]


# --- group functions (datatoolkit-issues#8) -----------------------------------


def _visits():
    return pd.DataFrame(
        {
            "patient id": ["a", "a", "a", "b", "b"],
            "age": [50.0, 52.0, 51.0, 30.0, 31.0],
            "ledd": [100.0, 300.0, np.nan, 7.0, np.nan],
        },
        index=[0, 0, 1, 1, 2],  # duplicated index labels
    )


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ('group_mean(ledd, by=df["patient id"])', [200.0, 200.0, 200.0, 7.0, 7.0]),
        ('group_prev(ledd, by="patient id", order=age)', [100, 300, 100, 7, 7]),
        ('group_interp(ledd, by="patient id", order=age)', [100, 300, 200, 7, None]),
        # order is an expression: reversed time carries values backward.
        ('group_prev(ledd, by="patient id", order=-age)', [100, 300, 300, 7, None]),
        # Composable: entity mean minus the row's value.
        ('ledd - group_mean(ledd, by="patient id")', [-100, 100, None, 0, None]),
    ],
)
def test_group_functions(expr, expected):
    out = _run(_visits(), name="y", expr=expr)
    expected = [np.nan if v is None else float(v) for v in expected]
    np.testing.assert_allclose(out["y"].to_numpy(), expected)
    assert out.index.tolist() == [0, 0, 1, 1, 2]


def test_group_functions_are_computed_per_frame():
    train = _visits()
    test = pd.DataFrame(
        {"patient id": ["a", "a"], "age": [60.0, 61.0], "ledd": [1.0, 3.0]}
    )
    out = _fit_apply(train, test, name="m", expr='group_mean(ledd, by="patient id")')
    assert out["m"].tolist() == [2.0, 2.0]  # test's own rows, not train's


@pytest.mark.parametrize(
    ("expr", "match"),
    [
        ("group_mean(ledd)", "keyword arguments by="),
        ("group_prev(ledd, by=age)", "by=, order="),
        ("group_mean(ledd, by=age, order=age)", "keyword arguments by="),
        ("group_mean(ledd, by=age + 1)", "must name a column"),
        ("group_mean(ledd, by=pi)", "must name a column"),
        ("group_mean(ledd, by=@v)", "must name a column"),
        ("group_mean(ledd, **kw)", r"\*\*unpacking"),
        ("np.group_mean(ledd, by=age)", "np.group_mean is not allowed"),
        ("log(ledd, base=2)", "keyword arguments is not allowed"),
        ("group_mean(ledd, age, by=age)", "takes 1 argument"),
    ],
)
def test_group_function_errors(expr, match):
    with pytest.raises(KeyParamsError, match=match):
        _parse(name="y", expr=expr)


def test_group_by_unknown_column():
    with pytest.raises(KeyParamsError, match="unknown column 'nope'"):
        _run(_visits(), name="y", expr="group_mean(ledd, by=nope)")


# --- NaN-aware functions (datatoolkit-issues#158) ---

_NAN = float("nan")


def _ev(values, expr, **cols):
    from dtk_engine.ops.transforms.formula import evaluate

    return evaluate(pd.DataFrame({"x": values, **cols}), expr)


def test_where_nan_condition_is_nan():
    out = _ev([1.0, _NAN], "where(x > 0, 1, 2)")
    assert out[0] == 1 and np.isnan(out[1])


@pytest.mark.parametrize("fn,all_nan,no_nan", [("isna", 1.0, 0.0), ("notna", 0.0, 1.0)])
def test_isna_notna(fn, all_nan, no_nan):
    mixed = _ev([1.0, _NAN], f"{fn}(x)").tolist()
    assert mixed == [no_nan, all_nan]
    assert set(_ev([_NAN, _NAN], f"{fn}(x)")) == {all_nan}
    assert set(_ev([1.0, 2.0], f"{fn}(x)")) == {no_nan}


def test_fillna():
    assert _ev([1.0, _NAN], "fillna(x, 9)").tolist() == [1.0, 9.0]
    assert _ev([_NAN, _NAN], "fillna(x, 0)").tolist() == [0.0, 0.0]
    assert _ev([1.0, 2.0], "fillna(x, 0)").tolist() == [1.0, 2.0]
    assert _ev([_NAN, 2.0], "fillna(x, y)", y=[5.0, 6.0]).tolist() == [5.0, 2.0]


def test_fill_only_missing_rows():
    out = _ev([40.0, _NAN, _NAN], "where(isna(x), age - 5.6, x)", age=[50.0, 60.0, 70.0])
    assert out.tolist() == pytest.approx([40.0, 54.4, 64.4])


def test_nan_functions_validation():
    from dtk_engine.ops.transforms.formula import check_expr

    with pytest.raises(ValueError, match="takes 2 arguments"):
        check_expr("fillna(x)")
    with pytest.raises(ValueError, match="not allowed"):
        check_expr("np.isna(x)")


def test_description_mentions_nan_functions():
    desc = get_transform("formula").description
    assert "isna" in desc and "fillna" in desc
