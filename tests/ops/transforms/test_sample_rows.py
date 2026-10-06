import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay


def _frame(n=20):
    return pd.DataFrame({"a": range(n)}, index=range(100, 100 + n))


def test_sample_rows_head():
    out = DtkTransformer("sample_rows", mode="head", n=3).fit_transform(_frame())
    assert out["a"].tolist() == [0, 1, 2]
    assert out.index.tolist() == [100, 101, 102]
    out = DtkTransformer("sample_rows", mode="head", frac=0.25).fit_transform(_frame())
    assert len(out) == 5


def test_sample_rows_random_is_seeded_ordered_and_keeps_ids():
    df = _frame()
    p = {"n": 7, "random_state": 3}
    a = DtkTransformer("sample_rows", **p).fit_transform(df)
    b = DtkTransformer("sample_rows", **p).fit_transform(df)
    assert a.equals(b) and len(a) == 7
    assert a.index.is_monotonic_increasing and a.index.is_unique
    assert (a["a"] == a.index - 100).all()  # row ids travel with their rows
    other = DtkTransformer("sample_rows", n=7, random_state=4).fit_transform(df)
    assert not a.index.equals(other.index)
    assert len(df) == 20  # input untouched


def test_sample_rows_n_larger_than_frame_keeps_all():
    out = DtkTransformer("sample_rows", n=99, random_state=0).fit_transform(_frame(5))
    assert out["a"].tolist() == [0, 1, 2, 3, 4]


def test_sample_rows_params_strict():
    t = get_transform("sample_rows")
    for bad in (
        {"random_state": 0},  # neither n nor frac
        {"n": 2, "frac": 0.5, "random_state": 0},
        {"n": 2},  # random without seed
        {"mode": "head", "n": 2, "random_state": 0},
        {"n": 0, "random_state": 0},
        {"frac": 1.5, "random_state": 0},
        {"mode": "tail", "n": 2},
        {"n": 2, "random_state": 0, "bogus": 1},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_sample_rows_replay_train_and_test():
    steps = [
        Step(op="sample_rows", target="both", params={"n": 4, "random_state": 1})
    ]
    train, test = _frame(10), _frame(6)
    assert len(replay(steps, "train", train, train)) == 4
    out = replay(steps, "test", test, train)
    assert len(out) == 4 and out.index.isin(test.index).all()
    only_train = [Step(op="sample_rows", target="train", params=steps[0].params)]
    assert len(replay(only_train, "test", test, train)) == 6
