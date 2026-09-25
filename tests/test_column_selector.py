"""The column-selector schema convention (params.columns_field / column_field)."""

import pytest

from dtk_engine import key_schema, list_keys, run_key, save_workspace, source_columns
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError

SELECTOR_KEYS = ["column_distribution", "target_analysis", "correlations"]


def _selectors(schema):
    return {
        name: prop
        for name, prop in schema["properties"].items()
        if "x-dtk-widget" in prop
    }


@pytest.mark.parametrize("key_id", SELECTOR_KEYS)
def test_hints_point_at_a_source_param(key_id):
    schema = key_schema(key_id)
    selectors = _selectors(schema)
    assert "columns" in selectors
    assert selectors["columns"]["x-dtk-widget"] == "columns"
    assert selectors["columns"]["type"] == "array"
    assert selectors["columns"]["default"] == []
    for prop in selectors.values():
        assert prop["x-dtk-widget"] in ("columns", "column")
        assert prop["x-dtk-dtype"] in ("any", "numeric")
        assert prop["x-dtk-source"] in schema["properties"]


def test_every_hint_is_well_formed():
    for k in list_keys():
        for prop in _selectors(key_schema(k["id"])).values():
            assert {"x-dtk-widget", "x-dtk-source", "x-dtk-dtype"} <= set(prop)


def test_correlations_offers_numeric_columns_only():
    prop = key_schema("correlations")["properties"]["columns"]
    assert prop["x-dtk-dtype"] == "numeric"


def test_source_columns():
    cols = source_columns({"kind": "csv", "path": TRAIN_CSV})
    by_name = {c["name"]: c for c in cols}
    assert by_name["Age"]["numeric"] and not by_name["Sex"]["numeric"]
    assert cols[0]["name"] == "PassengerId"
    with pytest.raises(KeyParamsError):
        source_columns({"kind": "nope"})


def test_selector_keys_run_on_a_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    save_workspace(
        {
            "name": "w",
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": TRAIN_CSV},
                    "target_column": "Survived",
                },
                "test": {"x": {"kind": "csv", "path": TEST_CSV}},
            },
        }
    )
    train = {"kind": "dataset", "workspace": "w", "role": "train"}
    test = {"kind": "dataset", "workspace": "w", "role": "test"}
    names = [c["name"] for c in source_columns(train)]
    assert "Survived" in names
    res = run_key(
        "column_distribution",
        {
            "source": train,
            "test": test,
            "compare": "train_vs_test",
            "columns": ["Fare"],
        },
    )
    assert res["metrics"]["n_groups"] == 2
    res = run_key("target_analysis", {"source": train, "columns": ["Age", "Sex"]})
    assert res["metrics"]["n_features"] == 2
    res = run_key("correlations", {"source": train, "target": "Survived"})
    assert res["metrics"]["n_columns"] >= 3
