import pandas as pd
import pytest
from dtk_engine.errors import SourceError
from dtk_engine.workspace import Step
from dtk_engine.workspace import replay as replay_mod  # the module
from dtk_engine.workspace.replay import replay, transform


@pytest.fixture
def ops(monkeypatch):
    """Temporary transforms: `add` adds params["n"]; `center` subtracts fit's mean."""
    registry = {}
    monkeypatch.setattr(replay_mod, "_TRANSFORMS", registry)

    @transform("add")
    def add(df, params, fit):
        return df + params["n"]

    @transform("center")
    def center(df, params, fit):
        ref = df if fit is None else fit
        return df - ref["v"].mean()

    return registry


def test_no_op_registered_yet():
    assert replay_mod.all_transforms() == []


def test_duplicate_transform(ops):
    with pytest.raises(ValueError, match="duplicate"):
        transform("add")(lambda df, p, f: df)


def test_unknown_op_is_clear_error():
    steps = [Step(op="drop_columns", target="train")]
    with pytest.raises(SourceError, match="unknown transform op 'drop_columns'"):
        replay(steps, "train", pd.DataFrame({"v": [1]}))


def test_empty_steps_is_identity():
    df = pd.DataFrame({"v": [1, 2]})
    assert replay([], "test", df) is df


def test_targets(ops):
    steps = [
        Step(op="add", target="train", params={"n": 1}),
        Step(op="add", target="test", params={"n": 10}),
    ]
    df = pd.DataFrame({"v": [0]})
    assert replay(steps, "train", df)["v"].tolist() == [1]
    assert replay(steps, "test", df)["v"].tolist() == [10]


def test_both_fits_on_train_as_of_the_step(ops):
    train = pd.DataFrame({"v": [0.0, 2.0]})  # mean 1, then 11 after the train step
    test = pd.DataFrame({"v": [20.0]})
    steps = [
        Step(op="add", target="train", params={"n": 10}),
        Step(op="center", target="both"),
    ]
    assert replay(steps, "train", train)["v"].tolist() == [-1.0, 1.0]
    assert replay(steps, "test", test, train)["v"].tolist() == [9.0]
    with pytest.raises(SourceError, match="needs the train frame"):
        replay(steps, "test", test)
