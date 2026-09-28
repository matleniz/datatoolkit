"""Workspace.charts persistence (MAT-185)."""

import pytest
from pydantic import ValidationError

from dtk_engine.contract import get_workspace, save_workspace
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace.models import ChartSpec, Workspace


def _base_ws(name="w_charts", **kw):
    return {
        "name": name,
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": TRAIN_CSV},
                "target_column": "Survived",
            }
        },
        **kw,
    }


def test_chart_spec_round_trip():
    spec = ChartSpec(name="Fare by Pclass", params={"chart": "box", "x": "Pclass", "y": "Fare"})
    assert spec.name == "Fare by Pclass"
    assert spec.params["chart"] == "box"


def test_chart_spec_strict_extra_forbidden():
    with pytest.raises(ValidationError):
        ChartSpec(name="c", params={}, extra_field=1)


def test_chart_spec_empty_name_refused():
    with pytest.raises(ValidationError):
        ChartSpec(name="", params={})


def test_workspace_charts_default_empty():
    ws = Workspace.model_validate(_base_ws())
    assert ws.charts == []
    assert ws.model_dump(mode="json")["charts"] == []


def test_workspace_charts_unique_names():
    ws = Workspace.model_validate(
        _base_ws(
            charts=[
                {"name": "Fare by Pclass", "params": {"chart": "box", "x": "Pclass", "y": "Fare"}},
                {"name": "Age hist", "params": {"chart": "histogram", "x": "Age"}},
            ]
        )
    )
    assert [c.name for c in ws.charts] == ["Fare by Pclass", "Age hist"]


def test_workspace_charts_duplicate_names_refused():
    with pytest.raises(ValidationError, match="duplicate chart name"):
        Workspace.model_validate(
            _base_ws(
                charts=[
                    {"name": "same", "params": {"chart": "box"}},
                    {"name": "same", "params": {"chart": "histogram"}},
                ]
            )
        )


def test_save_workspace_rejects_duplicate_charts(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    with pytest.raises(KeyParamsError, match="duplicate chart name"):
        save_workspace(
            _base_ws(
                charts=[
                    {"name": "dup", "params": {"chart": "box"}},
                    {"name": "dup", "params": {"chart": "bar"}},
                ]
            )
        )


def test_save_and_get_workspace_with_charts(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    charts = [
        {
            "name": "Fare by Pclass",
            "params": {"chart": "box", "x": "Pclass", "y": "Fare", "color": "Survived"},
        },
        {
            "name": "Age hist",
            "params": {"chart": "histogram", "x": "Age"},
        },
    ]
    saved = save_workspace(_base_ws(charts=charts))
    assert saved["charts"] == charts

    fetched = get_workspace("w_charts")
    assert fetched["charts"] == charts
