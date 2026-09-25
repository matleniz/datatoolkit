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
