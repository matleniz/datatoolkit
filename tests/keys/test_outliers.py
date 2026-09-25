import pytest

from dtk_engine import api, run_key
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError


def test_defaults_on_demo_data():
    res = run_key("outliers", {})
    m = res["metrics"]
    assert m["n_numeric_columns"] >= 1
    assert m["n_rows_flagged"] >= 1
    assert m["contamination"] == 0.01
    assert "remove" in res["text"] and "clip" in res["text"]


def test_deterministic():
    assert run_key("outliers", {})["metrics"] == run_key("outliers", {})["metrics"]


def test_contamination_is_explicit_numeric():
    with pytest.raises(KeyParamsError):
        run_key("outliers", {"contamination": "auto"})


def test_api_door():
    res = api.outliers(api.load(TRAIN_CSV), contamination=0.05)
    assert res.metrics["n_rows_flagged"] >= 1
