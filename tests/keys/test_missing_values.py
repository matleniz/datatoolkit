import pytest

from dtk_engine import api, run_key
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV


def _tables(res):
    return {t["title"]: t["records"] for t in res["tables"]}


def test_defaults_on_demo_data():
    res = run_key("missing_values", {})
    m = res["metrics"]
    assert m["n_columns_with_missing"] >= 1
    rates = _tables(res)["missing_rates"]
    assert rates[0]["pct_missing"] >= rates[-1]["pct_missing"]
    assert "n_test_spikes" not in m


def test_with_test_and_target():
    res = run_key(
        "missing_values",
        {
            "source": {"kind": "csv", "path": TRAIN_CSV},
            "test": {"kind": "csv", "path": TEST_CSV},
            "target": "Survived",
        },
    )
    assert "n_test_spikes" in res["metrics"]
    assert res["metrics"]["n_rows_missing_target"] == 0
    assert "test_value_spikes" in _tables(res)


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        api.missing(api.load(TRAIN_CSV), target="nope")


def test_api_door_renders():
    assert "missing_rates" in api.missing(api.load(TRAIN_CSV))._repr_html_()
