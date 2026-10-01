import json

import numpy as np
import pandas as pd
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from dtk_engine import DtkTransformer
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.ops.transforms.impute import GROUP_STRATEGIES
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


# --- impute strategy "formula" (datatoolkit-issues#7) -------------------------


def _diag_frames():
    """age_at_diagnosis = age - years_since_diagnosis, with gaps."""
    train = pd.DataFrame(
        {
            "age": [60.0, 70.0, 55.0, 80.0],
            "years": [5.0, np.nan, 2.0, 10.0],
            "age_at_diagnosis": [np.nan, 61.0, np.nan, 70.0],
        }
    )
    test = pd.DataFrame(
        {
            "age": [50.0, 65.0, 72.0],
            "years": [3.0, 1.0, np.nan],
            "age_at_diagnosis": [np.nan, 40.0, np.nan],
        },
        index=[4, 4, 5],  # duplicated index labels
    )
    return train, test


FORMULA = {
    "columns": ["age_at_diagnosis"],
    "strategy": "formula",
    "expr": "age - years",
}


def test_impute_formula_fills_only_missing_rows():
    train, test = _diag_frames()
    t, p, state = _fit("impute", train, **FORMULA)
    assert state == {"fill": {}}  # stateless: the expression is the state
    out = t.apply(train, p, state)
    assert out["age_at_diagnosis"].tolist() == [55.0, 61.0, 53.0, 70.0]
    out = t.apply(test, p, state)
    # Observed 40.0 kept although age - years = 64; undefined expr -> stays NaN.
    assert out["age_at_diagnosis"].tolist()[:2] == [47.0, 40.0]
    assert np.isnan(out["age_at_diagnosis"].iloc[2])
    assert out.index.tolist() == [4, 4, 5]
    assert test["age_at_diagnosis"].isna().sum() == 2  # input untouched


def test_impute_formula_train_test_replay_identical():
    train, test = _diag_frames()
    steps = [Step(op="impute", target="both", params=FORMULA)]
    t, p, state = _fit("impute", train, **FORMULA)
    replayed = replay(steps, "test", test, train)
    assert replayed.equals(t.apply(test, p, state))
    # Test is computed from its own rows only: fitting on test changes nothing.
    assert replayed.equals(t.fit_apply(test, p))


def test_impute_formula_indicator_and_python_syntax():
    train, _ = _diag_frames()
    params = {**FORMULA, "expr": 'df["age"] - np.abs(years)', "add_indicator": True}
    t, p, state = _fit("impute", train, **params)
    out = t.apply(train, p, state)
    assert out.columns.tolist() == [
        "age",
        "years",
        "age_at_diagnosis",
        "age_at_diagnosis_was_missing",
    ]
    assert out["age_at_diagnosis_was_missing"].tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"expr": "age -"}, "invalid expression"),
        ({"expr": "__import__('os')"}, "not allowed"),
        ({"expr": "age - @offset"}, "@variables"),
        ({"expr": "  "}, "needs an expr"),
        ({"expr": None}, "needs an expr"),
        ({"columns": ["age", "years"]}, "exactly one column"),
    ],
)
def test_impute_formula_invalid_params(params, match):
    with pytest.raises(KeyParamsError, match=match):
        get_transform("impute").parse({**FORMULA, **params})


def test_impute_formula_runtime_errors():
    train, _ = _diag_frames()
    t = get_transform("impute")
    with pytest.raises(KeyParamsError, match="unknown column 'nope'"):
        t.fit_apply(train, t.parse({**FORMULA, "expr": "nope + 1"}))
    text = pd.DataFrame({"c": ["a", None], "age": [1.0, 2.0]})
    with pytest.raises(ValueError, match="numeric"):
        _fit("impute", text, columns=["c"], strategy="formula", expr="age")


def test_impute_formula_expr_ignored_by_other_strategies():
    # A leftover expr (strategy switched in a form) is not validated nor used.
    p = get_transform("impute").parse({"columns": ["x"], "expr": "age -"})
    assert p.strategy == "median"


def test_impute_formula_transformer_and_api():
    from dtk_engine import api

    train, test = _diag_frames()
    tr = DtkTransformer("impute", **FORMULA).fit(train)
    t, p, state = _fit("impute", train, **FORMULA)
    assert tr.transform(test).equals(t.apply(test, p, state))
    out = api.transform(train, "impute", **FORMULA)
    assert out["age_at_diagnosis"].tolist() == [55.0, 61.0, 53.0, 70.0]


def test_impute_schema_shows_expr_only_for_formula():
    from dtk_engine import transform_schema

    props = transform_schema("impute")["properties"]
    assert "formula" in props["strategy"]["enum"]
    assert props["expr"]["x-dtk-when"] == {"strategy": "formula"}
    assert props["fill_value"]["x-dtk-when"] == {"strategy": "constant"}


# --- impute group strategies + ffill by (datatoolkit-issues#8) ----------------


def _visit_frames():
    """Train: a has a gap at age 57.5; b has a single observed visit."""
    train = pd.DataFrame(
        {
            "pid": ["a", "a", "a", "b", "b"],
            "age": [56.9, 58.9, 57.5, 30.0, 31.0],
            "ledd": [885.0, 835.0, np.nan, np.nan, 10.0],
            "arm": ["on", None, None, "off", None],
        }
    )
    # Test: patient c only exists here; d has nothing observed.
    test = pd.DataFrame(
        {
            "pid": ["c", "c", "c", "d"],
            "age": [40.0, 42.0, 41.0, 70.0],
            "ledd": [100.0, 300.0, np.nan, np.nan],
            "arm": ["off", None, None, None],
        },
        index=[9, 9, 8, 8],  # duplicated index labels
    )
    return train, test


GROUP = {"columns": ["ledd"], "by": "pid", "order": "age"}


@pytest.mark.parametrize(
    ("strategy", "train_ledd", "test_ledd"),
    [
        ("group_mean", [885, 835, 860, 10, 10], [100, 300, 200, None]),
        ("group_prev", [885, 835, 885, None, 10], [100, 300, 100, None]),
        ("group_interp", [885, 835, 870, None, 10], [100, 300, 200, None]),
    ],
)
def test_impute_group_strategies_fill_within_entity(strategy, train_ledd, test_ledd):
    train, test = _visit_frames()
    t, p, state = _fit("impute", train, **GROUP, strategy=strategy)
    assert state == {"fill": {}}  # nothing learned without a fallback

    def as_float(values):
        return [np.nan if v is None else float(v) for v in values]

    np.testing.assert_allclose(t.apply(train, p, state)["ledd"], as_float(train_ledd))
    out = t.apply(test, p, state)  # test's own rows: patient c is new
    np.testing.assert_allclose(out["ledd"], as_float(test_ledd))
    assert out.index.tolist() == [9, 9, 8, 8]
    assert test["ledd"].isna().sum() == 2  # input untouched


def test_impute_group_fallback_learned_on_train():
    train, test = _visit_frames()
    params = {**GROUP, "strategy": "group_interp", "fallback": "median"}
    t, p, state = _fit("impute", train, **params)
    assert state == {"fill": {"ledd": 835.0}}  # train median
    assert t.apply(train, p, state)["ledd"].tolist() == [885, 835, 870, 835, 10]
    assert t.apply(test, p, state)["ledd"].tolist() == [100, 300, 200, 835]
    steps = [Step(op="impute", target="both", params=params)]
    assert replay(steps, "test", test, train).equals(t.apply(test, p, state))
    tr = DtkTransformer("impute", **params).fit(train)
    assert tr.transform(test).equals(t.apply(test, p, state))


def test_impute_group_prev_any_dtype_and_dates():
    train, _ = _visit_frames()
    train["visit"] = pd.to_datetime(
        ["2020-01-01", "2020-03-01", "2020-02-01", None, "2021-01-01"]
    )
    t, p, state = _fit(
        "impute", train, columns=["arm"], strategy="group_prev", by="pid", order="visit"
    )
    out = t.apply(train, p, state)["arm"].tolist()
    # Row 3 has no visit date: not ordered, so neither filled nor a source.
    assert out[:3] == ["on", "on", "on"] and out[3] == "off" and pd.isna(out[4])


def test_impute_group_strategies_through_formula_and_api():
    from dtk_engine import api

    train, _ = _visit_frames()
    via_strategy = api.transform(train, "impute", **GROUP, strategy="group_interp")
    via_formula = api.transform(
        train,
        "impute",
        columns=["ledd"],
        strategy="formula",
        expr="group_interp(ledd, by=pid, order=age)",
    )
    assert via_strategy.equals(via_formula)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"strategy": "group_mean", "by": None}, "needs by"),
        ({"strategy": "group_prev", "order": None}, "needs order"),
        ({"strategy": "group_interp", "columns": ["ledd", "age"]}, "cannot be filled"),
        ({"strategy": "group_mean", "columns": ["pid"]}, "cannot be filled"),
        ({"strategy": "group_mean", "fallback": "max"}, "fallback"),
    ],
)
def test_impute_group_invalid_params(params, match):
    with pytest.raises(KeyParamsError, match=match):
        get_transform("impute").parse({**GROUP, **params})


def test_impute_group_numeric_only_for_mean_and_interp():
    train, _ = _visit_frames()
    for strategy in ("group_mean", "group_interp"):
        with pytest.raises(ValueError, match="numeric"):
            _fit(
                "impute",
                train,
                columns=["arm"],
                strategy=strategy,
                by="pid",
                order="age",
            )


def test_ffill_by_stays_within_entity():
    df = pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5, 6],
            "pid": ["a", "b", "a", "b", None, None],
            "v": [1.0, np.nan, np.nan, 7.0, 9.0, np.nan],
        },
        index=[0, 0, 1, 1, 2, 2],
    )
    t, p, state = _fit("ffill", df, sort_by="t", by="pid")
    out = t.apply(df, p, state)
    # b's first row is not filled from a's 1.0; rows without pid form one group.
    np.testing.assert_allclose(out["v"], [1.0, np.nan, 1.0, 7.0, 9.0, 9.0])
    assert out["pid"].tolist() == df["pid"].tolist()  # by is never filled
    assert out.index.tolist() == df.index.tolist()
    tr = DtkTransformer("ffill", sort_by="t", by="pid").fit(df)
    assert tr.transform(df).equals(out)


def test_ffill_by_params():
    for params in (
        {"sort_by": "t", "by": "t"},
        {"sort_by": "t", "by": "pid", "columns": ["pid"]},
    ):
        with pytest.raises(KeyParamsError):
            get_transform("ffill").parse(params)


def test_group_params_schema_allows_studio_prefill():
    from dtk_engine import transform_schema

    impute = transform_schema("impute")["properties"]
    assert impute["by"]["x-dtk-widget"] == "column"
    assert impute["by"]["x-dtk-semantic"] == "group_id"
    assert impute["by"]["x-dtk-when"] == {"strategy": list(GROUP_STRATEGIES)}
    assert impute["order"]["x-dtk-when"] == {"strategy": ["group_prev", "group_interp"]}
    assert impute["fallback"]["x-dtk-when"] == {"strategy": list(GROUP_STRATEGIES)}
    ffill = transform_schema("ffill")["properties"]
    assert ffill["by"]["x-dtk-semantic"] == "group_id"
