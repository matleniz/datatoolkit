import json

import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.transform_registry import get_transform


def test_defaults_on_demo_data():
    res = run_key("feature_selection", {})
    m = res["metrics"]
    assert m["task"] == "classification" and m["n_features"] >= 3
    titles = [t["title"] for t in res["tables"]]
    assert "feature_scores" in titles and "suggested_steps" in titles
    assert "filter" in res["text"] and "interactions" in res["text"]
    assert "impute" in res["text"]  # Age has missing values
    assert len(res["figures"]) == 2


def test_wrapper_and_deterministic():
    a = run_key("feature_selection", {"wrapper": True})
    b = run_key("feature_selection", {"wrapper": True})
    assert a == b
    scores = next(t for t in a["tables"] if t["title"] == "feature_scores")
    assert "rfecv_rank" in scores["records"][0]


def test_api_door_planted_and_suggested_steps_are_valid():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "dup": a * 3 + rng.normal(scale=0.01, size=n),
            "noise": rng.normal(size=n),
            "const": 1.0,
            "y": (a > 0).astype(int),
        }
    )
    res = api.select_features(df, target="y")
    assert res.metrics["top_feature"] in {"a", "dup"}
    assert res.metrics["n_collinear_pairs"] == 1
    assert res.metrics["n_near_constant"] == 1
    steps = next(t for t in res.tables if t.title == "suggested_steps").records
    assert steps[0]["op"] == "drop_low_variance"
    for step in steps:  # each suggestion is a valid, runnable step
        t = get_transform(step["op"])
        out = t.fit_apply(df, t.parse(step["params"]))
        assert "y" in out
    json.dumps(res.model_dump())


def test_unknown_target():
    with pytest.raises(ValueError, match="target"):
        api.select_features(api.load(TRAIN_CSV), target="nope")


@pytest.mark.parametrize(
    "params",
    [
        {"target": "nope"},
        {"columns": ["Name"]},
        {"columns": ["Survived"]},
        {"columns": ["nope"]},
        {"target": "Name", "task": "regression"},
    ],
)
def test_bad_params_raise_key_params_error(params):
    with pytest.raises(KeyParamsError):
        run_key("feature_selection", params)


def test_no_numeric_feature_gives_encode_first_result():
    df = pd.DataFrame(
        {
            "city": ["Paris", "Lyon", "Nice"] * 10,
            "size": ["S", "M", "L"] * 10,
            "amount": np.arange(30.0),
        }
    )
    res = api.select_features(df, target="amount")
    assert res.metrics["n_features"] == 0
    assert res.metrics["n_non_numeric_columns"] == 2
    steps = next(t for t in res.tables if t.title == "encode_first_steps").records
    assert {s["op"] for s in steps} >= {"onehot", "ordinal"}
    assert "re-run feature_selection" in res.text
    json.dumps(res.model_dump())


def test_sample_reported_above_threshold(monkeypatch):
    from dtk_engine.keys.feature_selection import selection_result

    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "noise": rng.normal(size=300), "y": a * 2})
    assert "n_scored_rows" not in selection_result(df, "y").metrics
    monkeypatch.setattr("dtk_engine.ops.selection.SCORE_SAMPLE_SIZE", 120)
    res = selection_result(df, "y")
    assert res.metrics["n_scored_rows"] == 120 and res.metrics["n_rows"] == 300
    assert "sample of 120 of 300 rows" in res.text
    assert res.metrics["top_feature"] == "a"
    assert selection_result(df, "y").model_dump() == res.model_dump()


def test_sample_stratified_keeps_rare_class():
    from dtk_engine.ops.selection import SCORE_SAMPLE_SIZE, _sample_index

    rng = np.random.default_rng(3)
    y = (rng.random(30_000) < 0.005).astype(int)
    keep = _sample_index(y, "classification", 0)
    assert len(keep) == SCORE_SAMPLE_SIZE and (np.diff(keep) > 0).all()
    share = y[keep].mean()
    assert abs(share - y.mean()) < 0.0005
    assert (y[keep] == 1).sum() >= 40


def test_sample_falls_back_when_a_class_has_one_row():
    from dtk_engine.ops.selection import SCORE_SAMPLE_SIZE, _sample_index

    y = np.zeros(30_000, dtype=int)
    y[0] = 1
    keep = _sample_index(y, "classification", 0)
    assert len(keep) == SCORE_SAMPLE_SIZE
    reg = _sample_index(np.random.default_rng(0).normal(size=30_000), "regression", 0)
    assert len(reg) == SCORE_SAMPLE_SIZE
