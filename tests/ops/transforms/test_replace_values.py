import numpy as np
import pandas as pd
import pytest

from dtk_engine.errors import KeyParamsError
from dtk_engine.pipeline import DtkTransformer
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace.models import Step
from dtk_engine.workspace.replay import replay


def _run(df, mapping):
    t = get_transform("replace_values")
    return t.fit_apply(df, t.parse({"mapping": mapping}))


def test_text_mapping_and_untouched_values():
    df = pd.DataFrame({"s": ["a", "b", "c"], "n": [1, 2, 3]})
    out = _run(df, {"s": {"a": "x", "b": None}})
    assert out["s"].tolist()[0] == "x"
    assert pd.isna(out["s"].tolist()[1]) and out["s"].tolist()[2] == "c"
    assert out["n"].tolist() == [1, 2, 3]
    assert df["s"].tolist() == ["a", "b", "c"]  # input untouched


def test_numeric_keys_match_numbers_on_int_column():
    df = pd.DataFrame({"sex": [1, 2, 1, 3]})
    out = _run(df, {"sex": {"1": "M", "2": "F"}})
    assert out["sex"].tolist() == ["M", "F", "M", 3]


def test_numeric_keys_match_floats_and_text_key_is_inert():
    df = pd.DataFrame({"x": [1.0, 2.5, np.nan]})
    out = _run(df, {"x": {"1": 10, "2.5": 0, "abc": 99}})
    assert out["x"].tolist()[:2] == [10.0, 0.0]
    assert out["x"].dtype == float and np.isnan(out["x"].iloc[2])


def test_numeric_to_numeric_keeps_numeric_dtype_and_null_gives_nan():
    out = _run(pd.DataFrame({"x": [1, 2, 3]}), {"x": {"1": 10, "3": None}})
    assert out["x"].dtype == float
    assert out["x"].iloc[0] == 10 and np.isnan(out["x"].iloc[2])


def test_text_keys_do_not_match_numbers_in_text_column():
    out = _run(pd.DataFrame({"c": ["1", "01", "a"]}), {"c": {"1": "one"}})
    assert out["c"].tolist() == ["one", "01", "a"]


def test_bool_column():
    out = _run(
        pd.DataFrame({"b": [True, False]}), {"b": {"true": "yes", "false": "no"}}
    )
    assert out["b"].tolist() == ["yes", "no"]


def test_simultaneous_swap_not_chained():
    out = _run(pd.DataFrame({"s": ["a", "b"]}), {"s": {"a": "b", "b": "a"}})
    assert out["s"].tolist() == ["b", "a"]


def test_ambiguous_keys_rejected():
    with pytest.raises(ValueError, match="same value"):
        _run(pd.DataFrame({"x": [1]}), {"x": {"1": "a", "1.0": "b"}})


def test_missing_column_raises():
    with pytest.raises(KeyError):
        _run(pd.DataFrame({"x": [1]}), {"z": {"1": "a"}})


def test_params_strict():
    t = get_transform("replace_values")
    for bad in (
        {"mapping": {}},
        {"mapping": {"x": {}}},
        {"mapping": {"x": {"1": [2]}}},
        {"mapping": {"x": {"1": 2}}, "extra": 1},
    ):
        with pytest.raises(KeyParamsError):
            t.parse(bad)


def test_transformer_train_test():
    train = pd.DataFrame({"sex": [1, 2]})
    test = pd.DataFrame({"sex": [2, 1, 9]})
    t = DtkTransformer("replace_values", mapping={"sex": {"1": "M", "2": "F"}}).fit(
        train
    )
    assert t.transform(test)["sex"].tolist() == ["F", "M", 9]


def test_replay_both_train_and_test():
    steps = [
        Step(
            op="replace_values",
            target="both",
            params={"mapping": {"sex": {"1": "M", "2": "F"}}},
        )
    ]
    train = pd.DataFrame({"sex": [1, 2]})
    test = pd.DataFrame({"sex": [2, 1]})
    assert replay(steps, "train", train, train)["sex"].tolist() == ["M", "F"]
    assert replay(steps, "test", test, train)["sex"].tolist() == ["F", "M"]
