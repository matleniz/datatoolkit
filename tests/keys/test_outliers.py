import pandas as pd
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


def test_method_scopes_detectors():
    iqr_only = run_key("outliers", {"columns": ["Fare"], "method": "iqr"})
    assert iqr_only["metrics"]["method"] == "iqr"
    assert iqr_only["metrics"]["n_rows_flagged"] == 0
    assert iqr_only["metrics"]["n_columns_with_z_outliers"] == 0
    assert any("box plot" in f["title"] for f in iqr_only["figures"])

    z_only = run_key("outliers", {"columns": ["Fare"], "method": "zscore"})
    assert z_only["metrics"]["n_columns_with_iqr_outliers"] == 0
    assert not any(f["title"].startswith("% outliers") for f in z_only["figures"])

    if_only = run_key(
        "outliers", {"columns": ["Fare"], "method": "isolation_forest", "contamination": 0.05}
    )
    assert if_only["metrics"]["n_rows_flagged"] >= 1
    assert if_only["metrics"]["n_columns_with_iqr_outliers"] == 0


def _mains(res):
    return [f for f in res["figures"] if f.get("main")]


def test_single_column_box_plot_and_headline():
    res = run_key("outliers", {"columns": ["Fare"], "method": "iqr"})
    assert res["headline"].endswith("above 65.6") or "above" in res["headline"]
    assert " outliers in Fare (" in res["headline"]
    (main,) = _mains(res)
    traces = main["plotly"]["data"]
    assert traces[0]["type"] == "box" and traces[0]["orientation"] == "h"
    assert any(t["type"] == "scatter" for t in traces)
    assert main["plotly"]["layout"]["shapes"]  # fence line
    boxes = _table(res, "box_stats")
    assert boxes[0]["column"] == "Fare" and boxes[0]["n_above"] > 0


def test_multi_column_bars_sorted_and_multiples():
    df = pd.DataFrame(
        {
            "a": list(range(100)) + [1000] * 5,
            "b": list(range(100)) + [900, 1000, 1100, 5, 6],
            "c": list(range(105)),
        }
    )
    res = api.outliers(df, method="iqr").model_dump(mode="json")
    assert "columns with outliers; most:" in res["headline"]
    (main,) = _mains(res)
    bar = main["plotly"]["data"][0]
    assert bar["type"] == "bar" and bar["orientation"] == "h"
    assert list(bar["x"]) == sorted(bar["x"], reverse=True) and min(bar["x"]) > 0
    assert any("Box plots" in f["title"] for f in res["figures"])


def test_no_outliers_no_main_figure():
    df = api.load(TRAIN_CSV)[["Pclass"]].assign(x=lambda d: range(len(d)))
    res = api.outliers(df, columns=["x"], method="iqr")
    assert res.headline == "No outliers outside the IQR fences"
    assert not [f for f in res.figures if f.main]


def test_sampled_outlier_points_capped():
    from dtk_engine.ops.outliers import OUTLIER_POINTS_MAX, outlier_points

    df = pd.DataFrame({"a": list(range(1000)) + [10_000] * 900})
    assert len(outlier_points(df, "a", 0, 1000)) == OUTLIER_POINTS_MAX
