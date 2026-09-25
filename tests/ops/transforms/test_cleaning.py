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
