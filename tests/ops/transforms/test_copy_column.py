import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def _frame(offset=0):
    return pd.DataFrame({c: [i + offset] for i, c in enumerate("abc")})


def _copy(**params):
    t = get_transform("copy_column")
    return t.fit_apply(_frame(), t.parse({"column": "b", "name": "z", **params}))


def test_copy_column_appends_by_default():
    df = _frame()
    out = get_transform("copy_column").fit_apply(
        df, get_transform("copy_column").parse({"column": "b", "name": "z"})
    )
    assert out.columns.tolist() == ["a", "b", "c", "z"]
    assert out["z"].tolist() == out["b"].tolist()
    assert df.columns.tolist() == ["a", "b", "c"]  # input untouched


def test_copy_column_position_reuses_reorder_semantics():
    assert _copy(position="first").columns.tolist() == list("zabc")
    assert _copy(position="last").columns.tolist() == list("abcz")
    assert _copy(position="before", anchor="a").columns.tolist() == list("zabc")
    assert _copy(position="after", anchor="b").columns.tolist() == list("abzc")


def test_copy_column_is_a_real_copy():
    out = _copy()
    out.loc[0, "z"] = 99
    assert out.loc[0, "b"] == 1


def test_copy_column_runtime_errors():
    t = get_transform("copy_column")
    with pytest.raises(ValueError, match="already exists"):
        t.fit_apply(_frame(), t.parse({"column": "a", "name": "c"}))
    with pytest.raises(KeyError):
        t.fit_apply(_frame(), t.parse({"column": "q", "name": "z"}))
    with pytest.raises(KeyError):
        t.fit_apply(
            _frame(), t.parse({"column": "a", "name": "z", "position": "after", "anchor": "q"})
        )


def test_copy_column_params_strict():
    t = get_transform("copy_column")
    for bad in (
        {"column": "a"},
        {"column": "a", "name": ""},
        {"column": "a", "name": "z", "position": "after"},
        {"column": "a", "name": "z", "anchor": "b"},
        {"column": "a", "name": "z", "position": "first", "anchor": "b"},
        {"column": "a", "name": "z", "position": "after", "anchor": "z"},
        {"column": "a", "name": "z", "position": "middle"},
        {"column": "a", "name": "z", "nme": "z"},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_copy_column_dtk_transformer_train_then_test():
    tr = DtkTransformer(
        "copy_column", column="a", name="a2", position="after", anchor="a"
    ).fit(_frame())
    assert tr.transform(_frame(10)).to_dict("list") == {
        "a": [10], "a2": [10], "b": [11], "c": [12]
    }


def test_copy_column_replay_train_and_test():
    steps = [Step(op="copy_column", target="both", params={"column": "c", "name": "d", "position": "first"})]
    train, test = _frame(), _frame(10)
    assert replay(steps, "train", train, train).columns.tolist() == list("dabc")
    out = replay(steps, "test", test, train)
    assert out.columns.tolist() == list("dabc")
    assert out["d"].tolist() == [12]
