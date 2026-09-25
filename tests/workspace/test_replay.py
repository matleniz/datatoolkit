import pandas as pd
import pytest

from dtk_engine import transform_registry as registry
from dtk_engine.errors import KeyParamsError, SourceError, UnknownTransformError
from dtk_engine.transform_registry import TransformParams, transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay, replay_fitted


class AddParams(TransformParams):
    n: float


@pytest.fixture
def ops(monkeypatch):
    """Temporary transforms: `add` adds params.n; `center` subtracts the fitted mean."""
    fits = []
    monkeypatch.setattr(registry, "_TRANSFORMS", {})

    @transform("add", params_model=AddParams)
    def add(df, params, state):
        return df + params.n

    def fit_center(df, params):
        fits.append(df["v"].tolist())
        return {"mean": float(df["v"].mean())}

    @transform("center", params_model=TransformParams, fit=fit_center)
    def center(df, params, state):
        return df - state["mean"]

    return fits


def test_unknown_op_is_clear_error():
    steps = [Step(op="not_an_op", target="train")]
    for role in ("train", "test"):
        with pytest.raises(UnknownTransformError, match="unknown transform op"):
            replay(steps, role, pd.DataFrame({"v": [1]}))
    with pytest.raises(UnknownTransformError):
        replay_fitted(steps, pd.DataFrame({"v": [1]}))


def test_invalid_params_fail_before_any_work(ops):
    steps = [
        Step(op="center", target="train"),
        Step(op="add", target="train", params={"m": 1}),
    ]
    with pytest.raises(KeyParamsError, match="transform 'add'"):
        replay(steps, "train", pd.DataFrame({"v": [1.0]}))
    assert ops == []  # center never fitted


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


def test_test_only_step_fits_on_test(ops):
    steps = [Step(op="center", target="test")]
    assert replay(steps, "test", pd.DataFrame({"v": [4.0, 6.0]}))["v"].tolist() == [
        -1.0,
        1.0,
    ]


def test_both_fits_on_train_as_of_the_step(ops):
    train = pd.DataFrame({"v": [0.0, 2.0]})  # mean 1, then 11 after the train step
    test = pd.DataFrame({"v": [20.0]})
    steps = [
        Step(op="add", target="train", params={"n": 10}),
        Step(op="center", target="both"),
    ]
    assert replay(steps, "train", train)["v"].tolist() == [-1.0, 1.0]
    # test uses the train statistic (11), never its own mean (20)
    assert replay(steps, "test", test, train)["v"].tolist() == [9.0]
    assert ops[-1] == [10.0, 12.0]
    with pytest.raises(SourceError, match="needs the train frame"):
        replay(steps, "test", test)


def test_chained_both_steps_refit_on_transformed_train(ops):
    train = pd.DataFrame({"v": [0.0, 4.0]})
    test = pd.DataFrame({"v": [10.0]})
    steps = [
        Step(op="add", target="both", params={"n": 1}),
        Step(op="center", target="both"),
    ]
    # second step fits on train after +1: mean 3
    assert replay(steps, "test", test, train)["v"].tolist() == [8.0]


def test_op_failure_names_the_step():
    steps = [Step(op="drop_columns", target="train", params={"columns": ["nope"]})]
    with pytest.raises(SourceError, match=r"step 0 \('drop_columns' on train\)"):
        replay(steps, "train", pd.DataFrame({"v": [1]}))


def test_invalid_params_are_not_source_errors(ops):
    steps = [Step(op="add", target="both", params={"n": "x"})]
    # no train frame: the bad params win over the missing-train SourceError
    with pytest.raises(KeyParamsError):
        replay(steps, "test", pd.DataFrame({"v": [1.0]}))


def test_replay_fitted_matches_replay(ops):
    steps = [
        Step(op="add", target="train", params={"n": 1}),
        Step(op="center", target="both"),
        Step(op="add", target="test", params={"n": 10}),
        Step(op="add", target="train", params={"n": 100}),
    ]
    train, test = pd.DataFrame({"v": [0.0, 2.0]}), pd.DataFrame({"v": [5.0]})
    tr, te, fitted = replay_fitted(steps, train, test)
    pd.testing.assert_frame_equal(tr, replay(steps, "train", train))
    pd.testing.assert_frame_equal(te, replay(steps, "test", test, train))
    assert tr["v"].tolist() == [99.0, 101.0] and te["v"].tolist() == [13.0]
    assert [f["fitted_on"] for f in fitted] == ["train", "train", "test", "train"]
    assert fitted[1]["state"] == {"mean": 2.0}
    # no test frame: the "test" step is recorded as not fitted
    _, te, fitted = replay_fitted(steps, train)
    assert te is None and fitted[2] == {"fitted_on": None, "state": {}}
