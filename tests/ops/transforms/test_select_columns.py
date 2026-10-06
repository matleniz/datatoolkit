import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def _frame(offset=0):
    return pd.DataFrame({"a": [1 + offset], "b": [2 + offset], "c": [3 + offset]})


def test_select_columns_keeps_given_order():
    t = get_transform("select_columns")
    df = _frame()
    out = t.fit_apply(df, t.parse({"columns": ["c", "a"]}))
    assert out.columns.tolist() == ["c", "a"]
    assert df.columns.tolist() == ["a", "b", "c"]  # input untouched


def test_select_columns_missing_ok():
    t = get_transform("select_columns")
    with pytest.raises(KeyError):
        t.fit_apply(_frame(), t.parse({"columns": ["a", "z"]}))
    out = t.fit_apply(_frame(), t.parse({"columns": ["z", "b"], "missing_ok": True}))
    assert out.columns.tolist() == ["b"]


def test_select_columns_params_strict():
    t = get_transform("select_columns")
    for bad in (
        {"columns": []},
        {"columns": ["a", "a"]},
        {"columns": ["a"], "colums": ["a"]},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_select_columns_dtk_transformer_train_then_test():
    tr = DtkTransformer("select_columns", columns=["b", "a"]).fit(_frame())
    assert tr.transform(_frame(10)).to_dict("list") == {"b": [12], "a": [11]}


def test_select_columns_replay_train_and_test():
    steps = [Step(op="select_columns", target="both", params={"columns": ["c", "a"]})]
    train, test = _frame(), _frame(10)
    assert replay(steps, "train", train, train).to_dict("list") == {"c": [3], "a": [1]}
    assert replay(steps, "test", test, train).to_dict("list") == {"c": [13], "a": [11]}
