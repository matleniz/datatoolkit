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


def test_errors():
    with pytest.raises(KeyParamsError, match="not in the frame"):
        run_key("chart", {"chart": "histogram", "x": "nope"})
    with pytest.raises(KeyParamsError, match="requires x and y"):
        run_key("chart", {"chart": "scatter", "x": "Age"})
    with pytest.raises(KeyParamsError):
        run_key("chart", {"chart": "not_a_chart", "x": "Age"})
