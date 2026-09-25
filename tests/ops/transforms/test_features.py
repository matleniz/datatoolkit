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
