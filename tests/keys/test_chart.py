"""Tests for the ``chart`` analysis key (MAT-172)."""

import pandas as pd
import pytest

from dtk_engine import api, key_schema, list_keys, run_key
from dtk_engine.errors import KeyParamsError


def test_listed_and_schema():
    keys = {k["id"]: k for k in list_keys()}
    assert "chart" in keys
    assert keys["chart"]["category"] == "analysis"
    assert keys["chart"]["needs_target"] is False
    schema = key_schema("chart")
    props = schema["properties"]
    assert props["chart"]["enum"] == [
        "histogram",
        "box",
        "violin",
        "bar",
        "count",
        "scatter",
        "line",
        "heatmap",
        "density_heatmap",
        "pie",
        "scatter_matrix",
    ]
    for name in ("x", "y", "color", "facet_row", "facet_col", "size"):
        assert props[name]["x-dtk-widget"] == "column"
        assert props[name]["x-dtk-source"] == "source"
    assert props["columns"]["x-dtk-widget"] == "columns"
    assert "agg" in props and "trendline" in props and "sample_size" in props


def test_defaults_on_demo_data():
    res = run_key("chart", {})
    assert res["metrics"]["chart"] == "histogram"
    assert res["metrics"]["n_rows"] == res["metrics"]["n_rows_source"]
    assert len(res["figures"]) == 1
    assert "plotly" in res["figures"][0]
    assert res["figures"][0]["plotly"]["data"]


def test_box_fare_by_pclass_colored_by_survived():
    res = run_key(
        "chart",
        {
            "chart": "box",
            "x": "Pclass",
            "y": "Fare",
            "color": "Survived",
        },
    )
    assert res["metrics"]["chart"] == "box"
    fig = res["figures"][0]["plotly"]
    assert fig["data"]
    assert fig["data"][0]["type"] == "box"


def test_scatter_with_trendline():
    res = run_key(
        "chart",
        {
            "chart": "scatter",
            "x": "Age",
            "y": "Fare",
            "trendline": True,
        },
    )
    assert res["metrics"]["trendline"] == 1
    types = {t["type"] for t in res["figures"][0]["plotly"]["data"]}
    assert "scatter" in types
    # OLS overlay is a second scatter (lines) trace
    assert len(res["figures"][0]["plotly"]["data"]) >= 2


def test_several_chart_types_render():
    cases = [
        {"chart": "histogram", "x": "Age"},
        {"chart": "violin", "x": "Pclass", "y": "Age"},
        {"chart": "bar", "x": "Pclass", "y": "Fare", "agg": "mean"},
        {"chart": "count", "x": "Sex"},
        {"chart": "line", "x": "Pclass", "y": "Fare", "agg": "median"},
        {"chart": "density_heatmap", "x": "Age", "y": "Fare"},
        {"chart": "heatmap", "x": "Age", "y": "Fare"},
        {"chart": "pie", "x": "Embarked"},
        {"chart": "scatter_matrix", "columns": ["Age", "Fare", "Pclass"]},
    ]
    for params in cases:
        res = run_key("chart", params)
        assert res["metrics"]["chart"] == params["chart"]
        assert res["figures"][0]["plotly"]["data"]


def test_sampling():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    res = api.chart(df, chart="scatter", x="a", y="b", sample_size=20)
    assert res.metrics["n_rows"] == 20
    assert res.metrics["n_rows_source"] == 100
    assert res.metrics["n_sampled_out"] == 80
    assert "Sampled 20" in res.text


def test_dataset_source_version(tmp_path, monkeypatch):
    from dtk_engine import save_workspace
    from dtk_engine.demo_data import TRAIN_CSV

    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    save_workspace(
        {
            "name": "w",
            "datasets": {"train": {"x": {"kind": "csv", "path": TRAIN_CSV}}},
            "steps": [],
        }
    )
    res = run_key(
        "chart",
        {
            "source": {
                "kind": "dataset",
                "workspace": "w",
                "role": "train",
                "version": 0,
            },
            "chart": "histogram",
            "x": "Age",
        },
    )
    assert res["metrics"]["chart"] == "histogram"
    assert res["figures"][0]["plotly"]["data"]


def test_binary_color_discrete_traces_and_no_coloraxis():
    """Binary / low-cardinality numeric color gets discrete traces and no coloraxis (MAT-251)."""
    res = run_key(
        "chart",
        {
            "chart": "scatter",
            "x": "Age",
            "y": "Fare",
            "color": "Survived",
            "trendline": True,
        },
    )
    fig = res["figures"][0]["plotly"]
    layout = fig.get("layout", {})
    # No continuous colorbar / coloraxis
    assert "coloraxis" not in layout

    # Points are split into discrete traces per category ('0' and '1')
    # and trendline OLS overlays match the category groups.
    trace_names = [t.get("name") for t in fig["data"]]
    assert trace_names == ["0", "1", "OLS (0)", "OLS (1)"]


def test_continuous_numeric_color_unchanged():
    """High-cardinality numeric color preserves continuous coloraxis and single trace (MAT-251)."""
    res = run_key(
        "chart",
        {
            "chart": "scatter",
            "x": "Age",
            "y": "Fare",
            "color": "Fare",
        },
    )
    fig = res["figures"][0]["plotly"]
    layout = fig.get("layout", {})
    assert "coloraxis" in layout
    assert len(fig["data"]) == 1


def test_color_cardinality_threshold():
    """Columns with <= 10 unique values are discrete; > 10 stay continuous (MAT-251)."""
    # Exactly 10 unique values -> discrete
    df10 = pd.DataFrame(
        {"x": range(20), "y": range(20), "c10": [i % 10 for i in range(20)]}
    )
    res10 = api.chart(df10, chart="scatter", x="x", y="y", color="c10")
    fig10 = res10.figures[0].plotly
    assert "coloraxis" not in fig10.get("layout", {})
    assert len(fig10["data"]) == 10

    # 11 unique values -> continuous
    df11 = pd.DataFrame(
        {"x": range(22), "y": range(22), "c11": [i % 11 for i in range(22)]}
    )
    res11 = api.chart(df11, chart="scatter", x="x", y="y", color="c11")
    fig11 = res11.figures[0].plotly
    assert "coloraxis" in fig11.get("layout", {})
    assert len(fig11["data"]) == 1


def test_boolean_color_discrete():
    """Boolean column treated as discrete categorical color (MAT-251)."""
    df_bool = pd.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "flag": [True, False, True, False]}
    )
    res_bool = api.chart(df_bool, chart="scatter", x="x", y="y", color="flag")
    fig_bool = res_bool.figures[0].plotly
    assert "coloraxis" not in fig_bool.get("layout", {})
    assert len(fig_bool["data"]) == 2
    assert {t.get("name") for t in fig_bool["data"]} == {"True", "False"}


def test_errors():
    with pytest.raises(KeyParamsError, match="not in the frame"):
        run_key("chart", {"chart": "histogram", "x": "nope"})
    with pytest.raises(KeyParamsError, match="requires x and y"):
        run_key("chart", {"chart": "scatter", "x": "Age"})
    with pytest.raises(KeyParamsError):
        run_key("chart", {"chart": "not_a_chart", "x": "Age"})

