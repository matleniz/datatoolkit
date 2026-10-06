import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay

DF = pd.DataFrame(
    {" Age ": [1], "userID": [2], "Total Sales ($)": [3], "HTTPCode": [4]}
)


def _names(df=DF, **params):
    t = get_transform("rename_columns_bulk")
    return t.fit_apply(df, t.parse(params)).columns.tolist()


def test_rules():
    assert _names(rule="strip")[0] == "Age"
    assert _names(rule="lower")[:2] == [" age ", "userid"]
    assert _names(rule="upper")[1] == "USERID"
    assert _names(rule="snake_case") == ["age", "user_id", "total_sales", "http_code"]


def test_prefix_suffix_with_rule():
    out = _names(rule="snake_case", prefix="x_", suffix="_y")
    assert out[1] == "x_user_id_y"
    assert _names(prefix="p_") == [
        "p_ Age ",
        "p_userID",
        "p_Total Sales ($)",
        "p_HTTPCode",
    ]


def test_columns_subset_and_data_kept():
    t = get_transform("rename_columns_bulk")
    out = t.fit_apply(DF, t.parse({"rule": "upper", "columns": ["userID"]}))
    assert out.columns.tolist() == [" Age ", "USERID", "Total Sales ($)", "HTTPCode"]
    assert out["USERID"].tolist() == [2]
    assert DF.columns[1] == "userID"  # input untouched


def test_missing_column():
    with pytest.raises(KeyError):
        _names(rule="lower", columns=["nope"])
    assert (
        _names(rule="upper", columns=["nope", "userID"], missing_ok=True)[1] == "USERID"
    )


def test_duplicates_refused():
    df = pd.DataFrame({"A": [1], "a": [2]})
    with pytest.raises(ValueError, match="duplicate"):
        _names(df, rule="lower")
    # collision with an untouched column
    with pytest.raises(ValueError, match="duplicate"):
        _names(df, rule="lower", columns=["A"])


def test_params_rejected():
    t = get_transform("rename_columns_bulk")
    for bad in (
        {},
        {"rule": "title"},
        {"rule": "lower", "columns": ["a", "a"]},
        {"rule": "lower", "colums": []},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_transformer_train_test():
    train = pd.DataFrame({"A B": [1]})
    test = pd.DataFrame({"A B": [2]})
    t = DtkTransformer("rename_columns_bulk", rule="snake_case").fit(train)
    assert t.transform(test).columns.tolist() == ["a_b"]


def test_replay_both_train_and_test():
    steps = [
        Step(
            op="rename_columns_bulk",
            target="both",
            params={"rule": "lower", "prefix": "f_"},
        )
    ]
    train = pd.DataFrame({"A": [1]})
    test = pd.DataFrame({"A": [2]})
    assert replay(steps, "train", train, train).columns.tolist() == ["f_a"]
    assert replay(steps, "test", test, train).columns.tolist() == ["f_a"]
