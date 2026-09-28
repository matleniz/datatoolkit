import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def test_drop_columns():
    t = get_transform("drop_columns")
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = t.fit_apply(df, t.parse({"columns": ["a"]}))
    assert out.columns.tolist() == ["b"]
    assert df.columns.tolist() == ["a", "b"]  # input untouched


def test_drop_columns_missing_ok():
    t = get_transform("drop_columns")
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        t.fit_apply(df, t.parse({"columns": ["z"]}))
    out = t.fit_apply(df, t.parse({"columns": ["z"], "missing_ok": True}))
    assert out.columns.tolist() == ["a"]


def test_drop_columns_params_strict():
    t = get_transform("drop_columns")
    with pytest.raises(KeyParamsError):
        t.parse({"columns": []})
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["a"], "colums": ["a"]})


def test_drop_columns_replay_both():
    steps = [Step(op="drop_columns", target="both", params={"columns": ["a"]})]
    train = pd.DataFrame({"a": [1], "b": [2]})
    test = pd.DataFrame({"a": [3], "b": [4]})
    assert replay(steps, "test", test, train).to_dict("list") == {"b": [4]}


# --- B2 ops: each via DtkTransformer and via replay on "both" -----------------

import numpy as np

from dtk_engine.pipeline import DtkTransformer


def _both(op, params, train, test):
    """Replay one "both" step on test (fitted on train)."""
    return replay([Step(op=op, target="both", params=params)], "test", test, train)


def test_rename():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = DtkTransformer("rename", mapping={"a": "x"}).fit_transform(df)
    assert out.columns.tolist() == ["x", "b"]
    assert _both("rename", {"mapping": {"a": "x"}}, df, df).columns.tolist() == [
        "x",
        "b",
    ]
    with pytest.raises(KeyError):
        DtkTransformer("rename", mapping={"z": "y"}).fit_transform(df)
    ok = DtkTransformer("rename", mapping={"z": "y"}, missing_ok=True)
    assert ok.fit_transform(df).columns.tolist() == ["a", "b"]


def test_cast():
    df = pd.DataFrame({"a": ["1", "2"], "b": [1.0, 2.0]})
    out = DtkTransformer("cast", dtypes={"a": "int64"}).fit_transform(df)
    assert out["a"].dtype == np.int64
    assert _both("cast", {"dtypes": {"a": "int64"}}, df, df)["a"].dtype == np.int64
    with pytest.raises(ValueError):
        DtkTransformer("cast", dtypes={"a": "int64"}).fit_transform(
            pd.DataFrame({"a": ["x"]})
        )
    with pytest.raises(KeyParamsError):
        get_transform("cast").parse({"dtypes": {"a": "not_a_dtype"}})


def test_drop_duplicates_keeps_most_recent():
    df = pd.DataFrame(
        {"id": [1, 1, 2, 2], "ts": [2, 1, 1, 2], "v": ["new1", "old1", "old2", "new2"]}
    )
    p = {"subset": ["id"], "sort_by": ["ts"], "keep": "last"}
    out = DtkTransformer("drop_duplicates", **p).fit_transform(df)
    assert out["v"].tolist() == ["new1", "new2"]  # original order preserved
    assert _both("drop_duplicates", p, df, df)["v"].tolist() == ["new1", "new2"]
    first = DtkTransformer(
        "drop_duplicates", subset=["id"], sort_by=["ts"], keep="first"
    ).fit_transform(df)
    assert first["v"].tolist() == ["old1", "old2"]
    none = DtkTransformer("drop_duplicates", subset=["id"], keep="none")
    assert len(none.fit_transform(df)) == 0


def test_drop_duplicates_requires_sort_by():
    with pytest.raises(KeyParamsError):
        get_transform("drop_duplicates").parse({"keep": "first"})


def test_standardize_text():
    df = pd.DataFrame({"c": [" Paris ", "PARIS", "paris", None, "Lyon"]})
    p = {"columns": ["c"], "lower": True, "mapping": {"paris": "FR-Paris"}}
    out = DtkTransformer("standardize_text", **p).fit_transform(df)
    assert out["c"].tolist()[:3] == ["FR-Paris"] * 3
    assert pd.isna(out["c"][3]) and out["c"][4] == "lyon"
    assert _both("standardize_text", p, df, df)["c"][0] == "FR-Paris"
    with pytest.raises(TypeError):
        DtkTransformer("standardize_text", columns=["n"]).fit_transform(
            pd.DataFrame({"n": [1, 2]})
        )


def test_standardize_text_unify_separators():
    df = pd.DataFrame({"c": ["site-a", "site_a", "Site  A", "site.a"]})
    p = {"columns": ["c"], "lower": True, "unify_separators": True}
    out = DtkTransformer("standardize_text", **p).fit_transform(df)
    assert out["c"].tolist() == ["site a"] * 4
    assert _both("standardize_text", p, df, df)["c"].tolist() == ["site a"] * 4
    # unify_separators=False (default): the punctuation variants stay distinct
    assert DtkTransformer(
        "standardize_text", columns=["c"], lower=True
    ).fit_transform(df)["c"].nunique() == 4


def test_to_numeric_currency_and_percent():
    cases = [
        ("$2.39 ", {}, 2.39),
        ("$1,250.00", {"thousands": ","}, 1250.0),
        ("12,5 %", {"decimal": ",", "percent": True}, 0.125),
        ("1 250,00 EUR", {"decimal": ",", "thousands": " "}, 1250.0),
    ]
    for raw, extra, expected in cases:
        p = {"columns": ["v"], **extra}
        df = pd.DataFrame({"v": [raw]})
        out = DtkTransformer("to_numeric", **p).fit_transform(df)
        assert out["v"].iloc[0] == pytest.approx(expected), raw
        assert _both("to_numeric", p, df, df)["v"].iloc[0] == pytest.approx(expected)


def test_to_numeric_missing_and_errors():
    df = pd.DataFrame({"v": ["$1.00", None, "nope"]})
    out = DtkTransformer("to_numeric", columns=["v"], errors="coerce").fit_transform(df)
    assert out["v"].tolist()[0] == pytest.approx(1.0)
    assert pd.isna(out["v"].iloc[1]) and pd.isna(out["v"].iloc[2])
    with pytest.raises(ValueError):
        DtkTransformer("to_numeric", columns=["v"], errors="raise").fit_transform(df)


def test_to_numeric_params_strict():
    t = get_transform("to_numeric")
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["v"], "decimal": ",", "thousands": ","})


def test_drop_high_missing():
    train = pd.DataFrame(
        {
            "a": [1, None, None, None],
            "b": [1, 2, 3, None],
            "y": [1, None, None, None],
        }
    )
    test = pd.DataFrame({"a": [1, 2], "b": [3, 4], "y": [5, 6]})
    p = {"threshold": 0.5, "target": "y"}
    t = DtkTransformer("drop_high_missing", **p).fit(train)
    # a=75%>50% dropped; b=25%<50% kept; y=75%>50% but excluded as target
    assert t.state_ == {"dropped": ["a"]}
    assert t.transform(test).columns.tolist() == ["b", "y"]
    assert _both("drop_high_missing", p, train, test).columns.tolist() == ["b", "y"]


def test_drop_high_missing_exclude():
    train = pd.DataFrame({"a": [1, None, None], "b": [1, None, None]})
    p = {"threshold": 0.5, "exclude": ["a"]}
    t = DtkTransformer("drop_high_missing", **p).fit(train)
    assert t.state_ == {"dropped": ["b"]}


def test_parse_dates():
    df = pd.DataFrame({"d": ["2024-01-02", "2024-02-03"]})
    out = DtkTransformer("parse_dates", columns=["d"], format="%Y-%m-%d").fit_transform(
        df
    )
    assert str(out["d"].dtype).startswith("datetime64")
    assert str(_both("parse_dates", {"columns": ["d"]}, df, df)["d"].dtype).startswith(
        "datetime64"
    )
    with pytest.raises(ValueError):  # never coerced to NaT
        DtkTransformer("parse_dates", columns=["d"], format="%Y-%m-%d").fit_transform(
            pd.DataFrame({"d": ["nope"]})
        )


def test_replace_sentinels():
    df = pd.DataFrame({"a": [1, -999, 3], "b": ["x", "N/A", "y"]})
    p = {"sentinels": {"a": [-999], "b": ["N/A"]}}
    out = DtkTransformer("replace_sentinels", **p).fit_transform(df)
    assert out["a"].isna().tolist() == [False, True, False]
    assert out["b"].isna().tolist() == [False, True, False]
    assert _both("replace_sentinels", p, df, df)["a"].isna().sum() == 1
    assert df["a"].tolist() == [1, -999, 3]  # input untouched


def test_drop_missing_target():
    train = pd.DataFrame({"y": [1.0, None, 3.0], "x": [1, 2, 3]})
    t = DtkTransformer("drop_missing_target", target="y").fit(train)
    assert t.state_ == {"dropped": 1}
    assert t.transform(train)["x"].tolist() == [1, 3]
    unlabelled = pd.DataFrame({"x": [7, 8]})
    assert len(t.transform(unlabelled)) == 2
    out = _both("drop_missing_target", {"target": "y"}, train, train)
    assert out["x"].tolist() == [1, 3]


def test_filter_rows():
    df = pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "x", "y"]})
    p = {
        "conditions": [
            {"column": "a", "op": "gt", "value": 1},
            {"column": "b", "op": "eq", "value": "x"},
        ]
    }
    out = DtkTransformer("filter_rows", **p).fit_transform(df)
    assert out["a"].tolist() == [3.0]
    assert _both("filter_rows", p, df, df)["a"].tolist() == [3.0]
    either = {**p, "combine": "or"}
    assert len(DtkTransformer("filter_rows", **either).fit_transform(df)) == 4 - 1
    notna = {"conditions": [{"column": "a", "op": "notna"}]}
    assert len(DtkTransformer("filter_rows", **notna).fit_transform(df)) == 3
    isin = {"conditions": [{"column": "b", "op": "isin", "value": ["y"]}]}
    assert len(DtkTransformer("filter_rows", **isin).fit_transform(df)) == 2


def test_filter_rows_params_strict():
    t = get_transform("filter_rows")
    with pytest.raises(KeyParamsError):
        t.parse({"conditions": [{"column": "a", "op": "gt"}]})
    with pytest.raises(KeyParamsError):
        t.parse({"conditions": [{"column": "a", "op": "eval", "value": 1}]})
    with pytest.raises(KeyParamsError):
        t.parse({"conditions": [{"column": "a", "op": "isin", "value": 1}]})


def test_clip_fits_on_train():
    train = pd.DataFrame({"a": np.arange(101, dtype=float)})
    test = pd.DataFrame({"a": [-50.0, 50.0, 500.0]})
    p = {"columns": ["a"], "lower": 10, "upper": 90}
    t = DtkTransformer("clip", **p).fit(train)
    assert t.state_ == {"bounds": {"a": [10.0, 90.0]}}
    assert t.transform(test)["a"].tolist() == [10.0, 50.0, 90.0]
    assert _both("clip", p, train, test)["a"].tolist() == [10.0, 50.0, 90.0]
    with pytest.raises(KeyParamsError):
        get_transform("clip").parse({"columns": ["a"], "lower": 90, "upper": 10})


def test_align_to_train():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(10, 2, 500)})
    test = pd.DataFrame({"a": rng.normal(15, 4, 500)})
    p = {"columns": ["a"], "mode": "shift_mean"}
    shifted = _both("align_to_train", p, train, test)["a"]
    assert shifted.mean() == pytest.approx(train["a"].mean())
    assert shifted.std() == pytest.approx(test["a"].std())
    p = {"columns": ["a"], "mode": "standardize_to_train"}
    t = DtkTransformer("align_to_train", **p).fit(train)
    out = t.transform(test)["a"]
    assert out.mean() == pytest.approx(train["a"].mean())
    assert out.std() == pytest.approx(train["a"].std())
    assert _both("align_to_train", p, train, test)["a"].std() == pytest.approx(
        train["a"].std()
    )
    # identity on train itself
    assert t.transform(train)["a"].to_numpy() == pytest.approx(train["a"].to_numpy())


# --- align_to_train: complete version -----------------------------------------


def _align_frames(seed=1, n=400):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame({"a": rng.normal(10, 2, n)})
    test = pd.DataFrame({"a": rng.normal(15, 4, n)})
    return train, test


def _run(mode, train, test, **extra):
    p = {"columns": ["a"], "mode": mode, **extra}
    return _both("align_to_train", p, train, test)["a"]


def _iqr(s):
    return s.quantile(0.75) - s.quantile(0.25)


def test_align_shift_median():
    train, test = _align_frames()
    out = _run("shift_median", train, test)
    assert out.median() == pytest.approx(train["a"].median())
    assert out.std() == pytest.approx(test["a"].std())


def test_align_standardize_and_alias():
    train, test = _align_frames()
    out = _run("standardize", train, test)
    assert out.mean() == pytest.approx(train["a"].mean())
    assert out.std() == pytest.approx(train["a"].std())
    alias = _run("standardize_to_train", train, test)
    assert alias.to_numpy() == pytest.approx(out.to_numpy())


def test_align_robust():
    train, test = _align_frames()
    out = _run("robust", train, test)
    assert out.median() == pytest.approx(train["a"].median())
    assert _iqr(out) == pytest.approx(_iqr(train["a"]))


def test_align_quantile_recovers_monotone_distortion():
    train, _ = _align_frames()
    rng = np.random.default_rng(5)
    base = rng.normal(10, 2, 400)
    test = pd.DataFrame({"a": np.exp(base / 4) + 3})  # monotone distortion
    test.loc[[3, 7], "a"] = np.nan
    t = DtkTransformer("align_to_train", columns=["a"], mode="quantile").fit(train)
    assert len(t.state_["global"]["a"]["quantiles"]) == 101
    out = t.transform(test)["a"]
    assert out.isna().sum() == 2
    for q in (0.1, 0.5, 0.9):
        assert out.quantile(q) == pytest.approx(train["a"].quantile(q), abs=0.3)
    assert out.mean() == pytest.approx(train["a"].mean(), abs=0.2)
    # monotone: order preserved
    assert out.dropna().rank().equals(test["a"].dropna().rank())
    # identity (within interpolation error) on train
    same = t.transform(train)["a"]
    assert same.to_numpy() == pytest.approx(train["a"].to_numpy(), abs=0.6)


def test_align_quantile_ties_use_average_ranks():
    train = pd.DataFrame({"a": np.arange(101, dtype=float)})
    test = pd.DataFrame({"a": [1.0] * 30 + [2.0] * 30})
    out = (
        DtkTransformer("align_to_train", columns=["a"], mode="quantile")
        .fit(train)
        .transform(test)["a"]
    )
    assert out.iloc[:30].nunique() == 1
    assert out.iloc[0] < out.iloc[-1]


@pytest.mark.parametrize(
    "mode", ["shift_mean", "shift_median", "standardize", "robust"]
)
def test_align_identity_on_train(mode):
    train, _ = _align_frames()
    t = DtkTransformer("align_to_train", columns=["a"], mode=mode).fit(train)
    assert t.transform(train)["a"].to_numpy() == pytest.approx(train["a"].to_numpy())


def test_align_per_group_with_unseen_group():
    rng = np.random.default_rng(2)
    train = pd.DataFrame(
        {
            "g": ["x"] * 200 + ["y"] * 200,
            "a": np.r_[rng.normal(0, 1, 200), rng.normal(100, 1, 200)],
        }
    )
    test = pd.DataFrame(
        {
            "g": ["x"] * 100 + ["y"] * 100 + ["z"] * 100,
            "a": np.r_[
                rng.normal(5, 1, 100), rng.normal(50, 1, 100), rng.normal(7, 1, 100)
            ],
        }
    )
    p = {"columns": ["a"], "group": "g", "mode": "shift_mean"}
    t = DtkTransformer("align_to_train", **p).fit(train)
    assert set(t.state_["groups"]["a"]) == {"x", "y"}
    out = t.transform(test)
    assert out[out.g == "x"]["a"].mean() == pytest.approx(
        train[train.g == "x"]["a"].mean()
    )
    assert out[out.g == "y"]["a"].mean() == pytest.approx(
        train[train.g == "y"]["a"].mean()
    )
    # unseen group: global shift, mean of the whole test column lands on train's
    z_shift = out[out.g == "z"]["a"].mean() - test[test.g == "z"]["a"].mean()
    glob = train["a"].mean() - test["a"].mean()
    assert z_shift == pytest.approx(glob)
    assert _both("align_to_train", p, train, test)["a"].to_numpy() == pytest.approx(
        out["a"].to_numpy()
    )


def test_align_small_group_falls_back_to_global():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"g": ["x"] * 100 + ["y"] * 100, "a": rng.normal(0, 1, 200)})
    test = pd.DataFrame({"g": ["x"] * 100 + ["y"] * 5, "a": rng.normal(3, 1, 105)})
    p = {"columns": ["a"], "group": "g"}
    out = DtkTransformer("align_to_train", **p).fit(train).transform(test)
    glob = train["a"].mean() - test["a"].mean()
    small = out[out.g == "y"]["a"] - test[test.g == "y"]["a"]
    assert small.to_numpy() == pytest.approx(glob)


def test_align_small_frame_skip_and_raise():
    train, _ = _align_frames()
    test = pd.DataFrame({"a": [100.0, 101.0, 102.0]})
    out = DtkTransformer("align_to_train", columns=["a"]).fit(train).transform(test)
    assert out["a"].tolist() == [100.0, 101.0, 102.0]
    t = DtkTransformer("align_to_train", columns=["a"], on_small="raise").fit(train)
    with pytest.raises(ValueError, match="min_rows"):
        t.transform(test)
    lowered = DtkTransformer("align_to_train", columns=["a"], min_rows=2).fit(train)
    assert lowered.transform(test)["a"].mean() == pytest.approx(train["a"].mean())


@pytest.mark.parametrize("mode", ["standardize", "robust"])
def test_align_degenerate_scale_shifts_only(mode):
    train = pd.DataFrame({"a": [5.0] * 50})
    test = pd.DataFrame({"a": np.arange(50, dtype=float)})
    out = _run(mode, train, test)
    assert out.to_numpy() == pytest.approx(test["a"].to_numpy() - 24.5 + 5.0)
    # constant frame against a spread train: shift only, no division by zero
    spread, const = test, pd.DataFrame({"a": [7.0] * 50})
    res = _run(mode, spread, const)
    assert np.isfinite(res).all()
    assert res.nunique() == 1


def test_align_params_validation():
    t = get_transform("align_to_train")
    assert (
        t.parse({"columns": ["a"], "mode": "standardize_to_train"}).mode
        == "standardize"
    )
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["a"], "mode": "nope"})
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["a"], "group": "a"})
    with pytest.raises(KeyParamsError):
        t.parse({"columns": ["a"], "min_rows": 0})
