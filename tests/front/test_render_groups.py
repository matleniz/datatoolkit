import pandas as pd
from dtk_engine import run_key
from dtk_engine.result import Result
from dtk_streamlit.render import filter_options, filter_records, group_items


def _res(tables=(), figures=()):
    return {"tables": list(tables), "figures": list(figures)}


def test_flat_when_nothing_grouped():
    res = _res([{"title": "t", "records": []}], [{"title": "f", "plotly": {}}])
    assert group_items(res) == {}
    assert group_items({}) == {}


def test_groups_in_first_appearance_order_overview_first():
    res = _res(
        tables=[
            {"title": "a", "records": [], "group": "Numeric"},
            {"title": "b", "records": []},
            {"title": "c", "records": [], "group": "Categorical"},
        ],
        figures=[{"title": "f", "plotly": {}, "group": "Numeric"}],
    )
    groups = group_items(res)
    assert list(groups) == ["Overview", "Numeric", "Categorical"]
    assert [t["title"] for t in groups["Overview"]["tables"]] == ["b"]
    assert [f["title"] for f in groups["Numeric"]["figures"]] == ["f"]


def test_overview_omitted_when_empty_and_explicit_overview_merges():
    res = _res(tables=[{"title": "a", "records": [], "group": "X"}])
    assert list(group_items(res)) == ["X"]
    res = _res(
        tables=[
            {"title": "a", "records": [], "group": "X"},
            {"title": "b", "records": [], "group": "Overview"},
            {"title": "c", "records": []},
        ]
    )
    groups = group_items(res)
    assert list(groups) == ["Overview", "X"]
    assert [t["title"] for t in groups["Overview"]["tables"]] == ["b", "c"]


def test_filter_records_on_column_field():
    records = [
        {"column": "a", "value": 1},
        {"column": "b", "value": 2},
        {"column": "a", "value": 3},
    ]
    assert filter_options(records) == ["a", "b"]
    assert filter_records(records, []) == records
    assert [r["value"] for r in filter_records(records, ["a"])] == [1, 3]
    assert filter_options([{"x": 1}]) == []


def test_result_group_backward_compatible():
    res = Result()
    res.add_table("t", pd.DataFrame({"a": [1]}))
    res.add_table("g", pd.DataFrame({"a": [1]}), group="G")
    assert [t.group for t in res.tables] == [None, "G"]
    assert Result.model_validate_json(res.model_dump_json()) == res


def test_dataset_overview_result_is_groupable():
    groups = group_items(run_key("dataset_overview", {}))
    assert next(iter(groups)) == "Overview" and "Numeric" in groups
