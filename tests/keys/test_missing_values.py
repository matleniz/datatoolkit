import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.transform_registry import get_transform


def _tables(res):
    return {t["title"]: t["records"] for t in res["tables"]}


def test_defaults_on_demo_data():
    res = run_key("missing_values", {})
    m = res["metrics"]
    assert m["n_columns_with_missing"] >= 1
    rates = _tables(res)["missing_rates"]
    assert rates[0]["pct_missing"] >= rates[-1]["pct_missing"]
    assert "n_test_spikes" not in m
    assert m["n_missing_cells"] >= 1
    assert 0 < m["pct_missing_cells"] < 100
    n_cells = m["n_rows"] * m["n_columns"]
    assert m["n_missing_cells"] == pytest.approx(
        m["pct_missing_cells"] / 100 * n_cells, abs=1
    )


def test_suggested_steps_drop_high_missing():
    df = pd.DataFrame(
        {
            "a": [1] * 3 + [None] * 7,  # 70% missing
            "b": list(range(10)),
            "y": list(range(10)),
        }
    )
    result = api.missing(df, target="y")
    steps = next(t for t in result.tables if t.title == "suggested_steps").records
    assert len(steps) == 1
    step = steps[0]
    assert step["op"] == "drop_high_missing"
    assert step["params"]["target"] == "y"
    get_transform(step["op"]).parse(step["params"])  # a valid step

    clean = api.missing(df[["b", "y"]])
    clean_steps = next(t for t in clean.tables if t.title == "suggested_steps").records
    assert clean_steps == []


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


def test_columns_param_scopes_per_column_and_row_stats():
    """Titanic: missing_values on Age (+ Cabin) only reports those columns."""
    res = run_key("missing_values", {"columns": ["Age", "Cabin"]})
    rates = _tables(res)["missing_rates"]
    assert {r["column"] for r in rates} == {"Age", "Cabin"}
    assert res["metrics"]["n_columns"] == 2
    # Per-row histogram only counts gaps among the chosen columns (max k = 2).
    per_row = _tables(res)["missing_per_row"]
    assert max(r["n_missing"] for r in per_row) <= 2
    sentinels = _tables(res)["sentinels"]
    assert all(r["column"] in {"Age", "Cabin"} for r in sentinels)


def test_columns_unknown_raises():
    with pytest.raises(KeyParamsError, match="not in the frame"):
        run_key("missing_values", {"columns": ["nope"]})


def test_unknown_target_raises():
    with pytest.raises(ValueError):
        api.missing(api.load(TRAIN_CSV), target="nope")


def test_api_door_renders():
    res = api.missing(api.load(TRAIN_CSV), columns=["Age"])
    assert res.metrics["n_columns"] == 1
    assert "missing_rates" in res._repr_html_()


def test_sort_and_threshold():
    df = pd.DataFrame(
        {
            "almost": [1] * 9 + [None],
            "half": [1] * 5 + [None] * 5,
            "full": [None] * 10,
            "ok": list(range(10)),
        }
    )
    by_name = api.missing(df, sort="column")
    names = [r["column"] for r in next(t.records for t in by_name.tables if t.title == "missing_rates")]
    assert names == sorted(names)

    filtered = api.missing(df, threshold=50)
    rates = [
        r for r in next(t.records for t in filtered.tables if t.title == "missing_rates")
    ]
    assert {r["column"] for r in rates} == {"half", "full"}
    assert filtered.metrics["threshold"] == 50
    assert filtered.metrics["pct_missing_cells"] > 0
