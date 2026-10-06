import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def _frame():
    return pd.DataFrame(
        {"a": [2, 1, None, 1, 2], "b": ["x", "y", "z", "x", "w"]},
        index=[10, 11, 12, 13, 14],
    )


def test_sort_rows_ascending_na_last_keeps_index():
    out = DtkTransformer("sort_rows", by=["a"]).fit_transform(_frame())
    assert out.index.tolist() == [11, 13, 10, 14, 12]  # stable among ties
    assert out["a"].isna().tolist() == [False] * 4 + [True]


def test_sort_rows_descending_na_first():
    out = DtkTransformer(
        "sort_rows", by=["a"], ascending=False, na_position="first"
    ).fit_transform(_frame())
    assert out.index.tolist() == [12, 10, 14, 11, 13]


def test_sort_rows_per_column_order():
    out = DtkTransformer(
        "sort_rows", by=["a", "b"], ascending=[True, False]
    ).fit_transform(_frame())
    assert out.index.tolist() == [11, 13, 10, 14, 12]


def test_sort_rows_input_untouched_and_missing_column():
    df = _frame()
    DtkTransformer("sort_rows", by=["a"]).fit_transform(df)
    assert df.index.tolist() == [10, 11, 12, 13, 14]
    with pytest.raises(KeyError):
        DtkTransformer("sort_rows", by=["zz"]).fit_transform(df)


def test_sort_rows_params_strict():
    t = get_transform("sort_rows")
    for bad in (
        {"by": []},
        {"by": ["a", "a"]},
        {"by": ["a"], "ascending": [True, False]},
        {"by": ["a"], "na_position": "middle"},
        {"by": ["a"], "bogus": 1},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_sort_rows_replay_train_and_test():
    steps = [Step(op="sort_rows", target="both", params={"by": ["a", "b"]})]
    train = pd.DataFrame({"a": [3, 1, 2], "b": list("xyz")})
    test = pd.DataFrame({"a": [2, 2, 1], "b": list("zwv")})
    assert replay(steps, "train", train, train)["a"].tolist() == [1, 2, 3]
    out = replay(steps, "test", test, train)
    assert out["b"].tolist() == ["v", "w", "z"]
    assert out.index.tolist() == [2, 1, 0]
