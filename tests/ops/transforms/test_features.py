import numpy as np
import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform


def run(op_name, df, **params):
    t = get_transform(op_name)
    return t.fit_apply(df, t.parse(params))


def fit_then_apply(op_name, train, test, **params):
    t = get_transform(op_name)
    p = t.parse(params)
    return t.apply(test, p, t.fit(train, p))


def test_derive_arith():
    df = pd.DataFrame({"a": [6.0, 4.0], "b": [3.0, 2.0]})
    assert run("derive", df, a="a", b="b", op="ratio")["a_ratio_b"].tolist() == [2, 2]
    assert run("derive", df, a="a", b="b", op="difference")[
        "a_difference_b"
    ].tolist() == [3, 2]
    assert run("derive", df, a="a", b="b", op="product", name="p")["p"].tolist() == [
        18,
        8,
    ]
    assert df.columns.tolist() == ["a", "b"]


def test_derive_ratio_no_inf():
    df = pd.DataFrame({"a": [5.0, 5.0, 5.0], "b": [0.0, 0.5, -0.1]})
    out = run("derive", df, a="a", b="b", op="ratio")["a_ratio_b"]
    assert np.isfinite(out).all()
    assert out.tolist() == [5.0, 5.0, -5.0]
    out = run("derive", df, a="a", b="b", op="ratio", min_denominator=0.25)["a_ratio_b"]
    assert out.tolist() == [20.0, 10.0, -20.0]


def test_derive_days_between():
    df = pd.DataFrame({"s": ["2024-01-01", "2024-01-10"], "e": ["2024-01-04", "bad"]})
    out = run("derive", df, a="s", b="e", op="days_between")["s_days_between_e"]
    assert out[0] == 3 and np.isnan(out[1])


def test_derive_rejects_free_code():
    with pytest.raises(KeyParamsError):
        get_transform("derive").parse({"a": "a", "b": "b", "op": "a + b"})


def test_datetime_parts():
    df = pd.DataFrame({"t": ["2024-03-02 23:30", "2024-03-04 01:00", None]})  # Sat, Mon
    out = run("datetime_parts", df, column="t")
    assert out["t_hour"].tolist()[:2] == [23, 1]
    assert out["t_dayofweek"].tolist()[:2] == [5, 0]
    assert out["t_month"].tolist()[:2] == [3, 3]
    assert out["t_year"].tolist()[:2] == [2024, 2024]
    assert out["t_is_weekend"].tolist()[:2] == [1, 0]
    assert pd.isna(out["t_is_weekend"][2])
    only = run("datetime_parts", df, column="t", parts=["month"])
    assert only.columns.tolist() == ["t", "t_month"]


def test_cyclical_23h_near_0h():
    df = pd.DataFrame({"h": [0, 23, 12]})
    out = run("cyclical", df, column="h", period=24)
    assert out.columns.tolist() == ["h", "h_sin", "h_cos"]
    d_23_0 = np.hypot(out.h_sin[1] - out.h_sin[0], out.h_cos[1] - out.h_cos[0])
    d_12_0 = np.hypot(out.h_sin[2] - out.h_sin[0], out.h_cos[2] - out.h_cos[0])
    assert d_23_0 < 0.3 < d_12_0
    assert out.h_sin[0] == 0 and out.h_cos[0] == 1


def test_bin_cut():
    df = pd.DataFrame({"x": [0, 5, 10, 20]})
    out = run("bin", df, column="x", mode="cut", edges=[0, 5, 10], name="b")
    assert out["b"].isna().tolist() == [True, False, False, True]  # left-open bins
    out = run(
        "bin", df, column="x", mode="cut", edges=[-1, 5, 10, 30], labels=["l", "m", "h"]
    )
    assert out["x_bin"].astype(str).tolist() == ["l", "l", "m", "h"]


def test_bin_qcut_fitted_on_train():
    train = pd.DataFrame({"x": np.arange(1.0, 101.0)})
    test = pd.DataFrame({"x": [-50.0, 30.0, 500.0]})
    t = get_transform("bin")
    p = t.parse({"column": "x", "mode": "qcut", "q": 4})
    state = t.fit(train, p)
    assert len(state["edges"]) == 3
    out = t.apply(test, p, state)
    assert out["x_bin"].tolist() == [0, 1, 3]  # out-of-range test values still binned


def test_bin_params_validation():
    t = get_transform("bin")
    for bad in (
        {"column": "x", "mode": "cut"},
        {"column": "x", "mode": "cut", "edges": [3, 1]},
        {"column": "x", "mode": "qcut"},
        {"column": "x", "mode": "qcut", "q": 4, "labels": ["a"]},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_group_agg_uses_train_stats_only():
    train = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    test = pd.DataFrame({"g": ["a", "b", "zzz"], "v": [1000.0, 1000.0, 1000.0]})
    out = fit_then_apply(
        "group_agg", train, test, group="g", value="v", aggs=["mean", "count"]
    )
    assert out["v_mean_by_g"].tolist()[:2] == [2.0, 10.0]
    assert out["v_count_by_g"].tolist()[:2] == [2.0, 1.0]
    assert out["v_mean_by_g"].isna().tolist() == [False, False, True]  # unseen -> NaN


def test_group_agg_state_json_safe_and_numeric_keys():
    train = pd.DataFrame({"g": [1, 1, 2], "v": [1.0, 3.0, 5.0]})
    t = get_transform("group_agg")
    p = t.parse({"group": "g", "value": "v", "aggs": ["std", "median"]})
    state = t.fit(train, p)  # single-row group -> NaN std must serialize
    out = t.apply(pd.DataFrame({"g": [1, 2, 3]}), p, state)
    assert out["v_median_by_g"].tolist()[:2] == [2.0, 5.0]
    assert out["v_std_by_g"].isna().tolist() == [False, True, True]


def test_group_agg_refuses_target():
    with pytest.raises(KeyParamsError, match="out-of-fold"):
        get_transform("group_agg").parse(
            {"group": "g", "value": "y", "aggs": ["mean"], "target": "y"}
        )


def test_interactions():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = run("interactions", df, columns=["a", "b", "c"])
    assert out.columns.tolist() == ["a", "b", "c", "a*b", "a*c", "b*c"]
    assert out["b*c"].tolist() == [15, 24]
    out = run("interactions", df, columns=["a", "b"], interaction_only=False)
    assert out.columns.tolist()[3:] == ["a*a", "a*b", "b*b"]


def test_interactions_limits():
    t = get_transform("interactions")
    with pytest.raises(KeyParamsError):
        t.parse({"columns": [f"c{i}" for i in range(11)]})
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["a"]})


def test_sklearn_door_group_agg():
    train = pd.DataFrame({"g": ["a", "b"], "v": [1.0, 2.0]})
    tr = DtkTransformer("group_agg", group="g", value="v", aggs=["mean"]).fit(train)
    out = tr.transform(pd.DataFrame({"g": ["b", "c"]}))
    assert out["v_mean_by_g"].isna().tolist() == [False, True]


def test_polynomial_degree2_readable_names():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "z": ["x", "y", "z"]})
    test = pd.DataFrame({"a": [10.0], "b": [2.0], "z": ["t"]})
    out = fit_then_apply(
        "polynomial", train, test, columns=["a", "b"], degree=2, include_bias=False
    )
    assert out.columns.tolist() == ["a", "b", "a^2", "a*b", "b^2", "z"]
    assert out["a^2"].tolist() == [100.0]
    assert out["a*b"].tolist() == [20.0]
    assert out["b^2"].tolist() == [4.0]
    # Train side mirrors sklearn.
    train_out = run("polynomial", train, columns=["a", "b"], degree=2)
    assert train_out.columns.tolist() == ["a", "b", "a^2", "a*b", "b^2", "z"]
    assert train_out["a*b"].tolist() == [4.0, 10.0, 18.0]


def test_polynomial_matches_sklearn_and_caps():
    from sklearn.preprocessing import PolynomialFeatures

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = run("polynomial", df, columns=["a", "b"], degree=2, interaction_only=True)
    ref = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    expected = ref.fit_transform(df[["a", "b"]])
    assert np.allclose(out[["a", "b", "a*b"]].to_numpy(), expected)

    t = get_transform("polynomial")
    p = t.parse({"columns": ["a", "b"], "degree": 2, "max_output_columns": 3})
    with pytest.raises(ValueError, match="cap is 3"):
        t.fit(df, p)


def test_power_transform_fitted_on_train_only():
    from sklearn.preprocessing import PowerTransformer

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=80) + 3, "y": rng.normal(size=80) + 5})
    test = pd.DataFrame({"x": [100.0], "y": [200.0]})
    t = get_transform("power_transform")
    p = t.parse({"columns": ["x", "y"], "method": "yeo-johnson", "standardize": True})
    state = t.fit(train, p)
    assert "lambdas" in state and "mean" in state and "scale" in state
    import json

    json.dumps(state)
    out = t.apply(test, p, state)
    ref = PowerTransformer(method="yeo-johnson", standardize=True).fit(train)
    assert np.allclose(out.to_numpy(), ref.transform(test))
    # Refitting on test would give different lambdas; frozen state wins.
    assert not np.allclose(t.fit(test, p)["lambdas"], state["lambdas"])


def test_power_transform_box_cox_and_no_standardize():
    train = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0]})
    out = run(
        "power_transform",
        train,
        columns=["x"],
        method="box-cox",
        standardize=False,
    )
    from sklearn.preprocessing import PowerTransformer

    ref = PowerTransformer(method="box-cox", standardize=False).fit_transform(
        train[["x"]]
    )
    assert np.allclose(out[["x"]].to_numpy(), ref)


def test_quantile_transform_fitted_on_train():
    from sklearn.preprocessing import QuantileTransformer

    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.normal(size=100), "b": rng.uniform(size=100)})
    test = pd.DataFrame({"a": [-10.0, 0.0, 10.0], "b": [0.0, 0.5, 1.0]})
    t = get_transform("quantile_transform")
    p = t.parse(
        {
            "columns": ["a", "b"],
            "output_distribution": "normal",
            "n_quantiles": 50,
        }
    )
    state = t.fit(train, p)
    out = t.apply(test, p, state)
    ref = QuantileTransformer(
        n_quantiles=50,
        output_distribution="normal",
        random_state=0,
        subsample=int(1e9),
    ).fit(train)
    assert np.allclose(out.to_numpy(), ref.transform(test))
    assert len(state["quantiles"]) == state["n_quantiles"]


def test_spline_fitted_on_train_and_readable_names():
    from sklearn.preprocessing import SplineTransformer

    rng = np.random.default_rng(2)
    train = pd.DataFrame({"a": rng.uniform(0, 10, 60), "keep": np.arange(60)})
    test = pd.DataFrame({"a": [0.0, 5.0, 10.0], "keep": [0, 1, 2]})
    t = get_transform("spline")
    p = t.parse({"columns": ["a"], "n_knots": 4, "degree": 2})
    state = t.fit(train, p)
    out = t.apply(test, p, state)
    assert "keep" in out.columns
    assert all(c.startswith("a_sp_") for c in out.columns if c != "keep")
    ref = SplineTransformer(n_knots=4, degree=2, knots="quantile", include_bias=True)
    ref.fit(train[["a"]])
    assert np.allclose(
        out.drop(columns=["keep"]).to_numpy(), ref.transform(test[["a"]])
    )
    # Cap
    p_cap = t.parse({"columns": ["a"], "n_knots": 4, "degree": 2, "max_output_columns": 2})
    with pytest.raises(ValueError, match="cap is 2"):
        t.fit(train, p_cap)


def test_new_feature_ops_dtk_transformer():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]})
    test = pd.DataFrame({"a": [5.0], "b": [6.0]})
    tr = DtkTransformer("polynomial", columns=["a", "b"], degree=2).fit(train)
    assert tr.transform(test).columns.tolist() == ["a", "b", "a^2", "a*b", "b^2"]
    tr = DtkTransformer("power_transform", columns=["a"]).fit(train)
    assert "lambdas" in tr.state_
    out = tr.transform(test)
    assert list(out.columns) == ["a", "b"]


def _visits():
    return pd.DataFrame(
        {
            "p": ["a", "a", "a", "b", "b", "c", "c"],
            "age": [60.0, 50.0, 55.0, 40.0, 41.0, 30.0, 31.0],
            "v": [3.0, 1.0, 2.0, np.nan, 9.0, np.nan, np.nan],
        }
    )


def test_group_agg_min_max_first_last():
    df = _visits()
    out = run(
        "group_agg", df, group="p", value="v", order="age",
        aggs=["min", "max", "first", "last"],
    )
    assert out["v_min_by_p"].tolist()[:5] == [1.0, 1.0, 1.0, 9.0, 9.0]
    assert out["v_max_by_p"].tolist()[:3] == [3.0, 3.0, 3.0]
    # ordered by age, not by row position; the NaN of group b is skipped
    assert out["v_first_by_p"].tolist()[:5] == [1.0, 1.0, 1.0, 9.0, 9.0]
    assert out["v_last_by_p"].tolist()[:3] == [3.0, 3.0, 3.0]


def test_group_agg_all_nan_group_gives_nan():
    out = run(
        "group_agg", _visits(), group="p", value="v", order="age",
        aggs=["min", "max", "first", "last"],
    )
    for agg in ("min", "max", "first", "last"):
        assert out[f"v_{agg}_by_p"].iloc[5:].isna().all()


def test_group_agg_first_age_equals_min_age_per_patient():
    df = _visits().rename(columns={"v": "x"}).assign(x=lambda d: d["age"])
    out = run("group_agg", df, group="p", value="x", order="age", aggs=["min", "first"])
    assert (out["x_min_by_p"] == out["x_first_by_p"]).all()
    assert out.groupby("p")["x_first_by_p"].nunique().eq(1).all()


def test_group_agg_first_last_order_rules():
    df = pd.DataFrame(
        {"g": ["a", "a", "a"], "o": [2.0, np.nan, 1.0], "v": [20.0, 99.0, 10.0]}
    )
    out = run("group_agg", df, group="g", value="v", order="o", aggs=["first", "last"])
    assert out["v_first_by_g"].tolist() == [10.0] * 3  # NaN-order row ignored
    assert out["v_last_by_g"].tolist() == [20.0] * 3
    with pytest.raises(KeyParamsError, match="order"):
        get_transform("group_agg").parse({"group": "g", "value": "v", "aggs": ["first"]})


def test_group_agg_first_uses_train_only_and_refuses_target():
    train = _visits()
    test = pd.DataFrame({"p": ["a", "zzz"], "age": [1.0, 1.0], "v": [-5.0, -5.0]})
    out = fit_then_apply(
        "group_agg", train, test, group="p", value="v", order="age", aggs=["first"]
    )
    assert out["v_first_by_p"].isna().tolist() == [False, True]
    assert out["v_first_by_p"].iloc[0] == 1.0
    with pytest.raises(KeyParamsError, match="out-of-fold"):
        get_transform("group_agg").parse(
            {"group": "p", "value": "y", "aggs": ["max"], "target": "y"}
        )


def test_group_agg_first_last_order_equals_value():
    df = pd.DataFrame({"p": ["a", "a", "b", "b"], "age": [60.0, 55.0, 70.0, 72.0]})
    out = run("group_agg", df, group="p", value="age", aggs=["first", "min"], order="age")
    assert out["age_first_by_p"].tolist() == out["age_min_by_p"].tolist() == [
        55.0, 55.0, 70.0, 70.0,
    ]


def test_group_agg_first_last_order_equals_group():
    df = pd.DataFrame({"p": [3, 1, 1], "v": [1.0, 2.0, 3.0]})
    out = run("group_agg", df, group="p", value="v", aggs=["last"], order="p")
    assert out["v_last_by_p"].tolist() == [1.0, 3.0, 3.0]
