import pytest

from dtk_engine import api, run_key
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError


def _table(res, title):
    return next(t["records"] for t in res["tables"] if t["title"] == title)


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


def test_columns_param_scopes_to_selection():
    """Titanic: outliers on Age alone lists only that column (MAT-159 / MAT-146)."""
    res = run_key("outliers", {"columns": ["Age"]})
    table = _table(res, "outliers_per_column")
    assert [r["column"] for r in table] == ["Age"]
    assert res["metrics"]["n_numeric_columns"] == 1
    flagged = _table(res, "flagged_rows")
    assert flagged and set(flagged[0]) >= {"row", "score", "Age"}
    assert "Fare" not in flagged[0]


def test_columns_unknown_or_non_numeric_raises():
    with pytest.raises(KeyParamsError, match="not in the frame"):
        run_key("outliers", {"columns": ["nope"]})
    with pytest.raises(KeyParamsError, match="must be numeric"):
        run_key("outliers", {"columns": ["Sex"]})


def test_api_door():
    res = api.outliers(api.load(TRAIN_CSV), contamination=0.05, columns=["Fare"])
    assert res.metrics["n_numeric_columns"] == 1
    assert res.metrics["n_rows_flagged"] >= 1
