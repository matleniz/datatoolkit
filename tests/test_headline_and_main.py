"""Tests for Result.headline and Figure.main contract (MAT-244)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dtk_engine import contract
from dtk_engine.result import Figure, Result


def test_result_model_defaults():
    """Verify default headline is empty string and figure main defaults to False."""
    res = Result()
    assert res.headline == ""
    assert res.figures == []

    fig = Figure(title="test", plotly={"data": []})
    assert fig.main is False


def test_result_single_main_figure_enforced():
    """At most one figure can have main=True in Result."""
    f1 = Figure(title="f1", plotly={}, main=True)
    f2 = Figure(title="f2", plotly={}, main=True)

    with pytest.raises(ValueError, match="At most one figure can have main=True"):
        Result(figures=[f1, f2])


def test_add_figure_switches_main():
    """add_figure(..., main=True) resets prior figures so at most one remains main."""
    res = Result()
    res.add_figure("f1", {"data": []}, main=True)
    assert len(res.figures) == 1
    assert res.figures[0].main is True

    res.add_figure("f2", {"data": []}, main=False)
    assert res.figures[0].main is True
    assert res.figures[1].main is False

    res.add_figure("f3", {"data": []}, main=True)
    assert res.figures[0].main is False
    assert res.figures[1].main is False
    assert res.figures[2].main is True


ANALYSIS_KEYS_WITH_FIGURES = [
    ("correlations", {}),
    ("missing_values", {}),
    ("outliers", {}),
    ("column_distribution", {}),
    ("target_analysis", {"target": "Survived"}),
    ("train_test_check", {}),
    ("feature_selection", {"target": "Survived"}),
    ("dataset_overview", {}),
]


@pytest.mark.parametrize("key_id,params", ANALYSIS_KEYS_WITH_FIGURES)
def test_keys_have_headline_and_single_main_figure(key_id: str, params: dict):
    """Every analysis key produces a non-empty headline and exactly 1 main figure."""
    data = contract.run_key(key_id, params)

    # JSON round-trip
    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    result = Result.model_validate(deserialized)

    # Headline check
    assert isinstance(result.headline, str)
    assert len(result.headline.strip()) > 0, f"{key_id} produced empty headline"

    # Exactly one main figure
    assert len(result.figures) > 0, f"{key_id} produced no figures"
    main_figures = [f for f in result.figures if f.main]
    assert len(main_figures) == 1, (
        f"{key_id} expected exactly 1 main figure, got {len(main_figures)}: "
        f"{[f.title for f in main_figures]}"
    )


def test_expected_main_figure_per_key():
    """Verify each key marks the exact requested figure as main."""
    # correlations -> main = heatmap
    res = contract.run_key("correlations", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert "correlation" in main_figs[0]["title"]

    # missing_values -> main = % missing per column
    res = contract.run_key("missing_values", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == "% missing per column"

    # outliers -> main = % outside fences per column
    res = contract.run_key("outliers", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == "% outliers per column (IQR)"

    # column_distribution -> main = focused / first column
    res = contract.run_key("column_distribution", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == res["figures"][0]["title"]

    # target_analysis -> main = per-feature figure of the top feature
    res = contract.run_key("target_analysis", {"target": "Survived"})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    top_feat = res["metrics"]["top_feature"]
    assert top_feat in main_figs[0]["title"]

    # train_test_check -> main = % missing train vs test
    res = contract.run_key("train_test_check", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == "% missing train vs test"

    # feature_selection -> main = feature scores
    res = contract.run_key("feature_selection", {"target": "Survived"})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert "Mutual information per feature" in main_figs[0]["title"]

    # dataset_overview -> main = % missing per column
    res = contract.run_key("dataset_overview", {})
    main_figs = [f for f in res["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == "% missing per column"


def test_chart_key_has_no_headline_but_main_figure():
    """chart key must not produce a headline, but its sole figure is marked main."""
    res = contract.run_key("chart", {"chart": "histogram", "x": "Age"})
    assert res.get("headline") == ""
    assert len(res["figures"]) == 1
    assert res["figures"][0]["main"] is True


def test_missing_values_clean_data_headline():
    """Clean data with 0 missing values outputs 'No missing values'."""
    from dtk_engine.keys.missing_values import missing_result

    clean_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    res = missing_result(clean_df)
    assert res.headline == "No missing values"


def test_outliers_clean_data_headline():
    """Data with 0 outliers outputs 'No outliers outside the IQR fences'."""
    from dtk_engine.keys.outliers import outliers_result

    clean_df = pd.DataFrame({"a": [10, 11, 12, 10, 11, 12, 11, 10, 12, 11]})
    res = outliers_result(clean_df)
    assert res.headline == "No outliers outside the IQR fences"


def test_http_api_passes_headline_and_main(tmp_path, monkeypatch):
    """Verify HTTP POST /api/keys/{id}/run preserves headline and main figure flag."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))

    client = TestClient(create_app())
    resp = client.post("/api/keys/missing_values/run", json={"params": {}})
    assert resp.status_code == 200
    data = resp.json()

    assert "headline" in data
    assert isinstance(data["headline"], str)
    assert len(data["headline"]) > 0

    assert "figures" in data
    main_figs = [f for f in data["figures"] if f.get("main")]
    assert len(main_figs) == 1
    assert main_figs[0]["title"] == "% missing per column"
