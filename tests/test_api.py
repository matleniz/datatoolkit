import pandas as pd
import pytest

from dtk_engine import Result, api, run_key
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, SourceError


def test_load_path_and_spec(tmp_path):
    assert len(api.load(TRAIN_CSV)) == 41
    path = tmp_path / "semi.csv"
    path.write_text("a;b\n1,5;2\n")
    df = api.load({"kind": "csv", "path": str(path), "decimal": ","})
    assert df["a"].tolist() == [1.5]


def test_load_errors(tmp_path):
    with pytest.raises(SourceError, match="no reader for '.xyz'"):
        api.load(tmp_path / "f.xyz")
    with pytest.raises(KeyParamsError):
        api.load({"kind": "csv"})
    with pytest.raises(SourceError, match="not found"):
        api.load(tmp_path / "missing.csv")


def test_overview_plain_dataframe():
    df = pd.DataFrame({"n": [1.0, 2.0, None], "c": ["a", "b", "a"]})
    res = api.overview(df)
    assert isinstance(res, Result)
    assert res.metrics["rows"] == 3 and res.metrics["cols"] == 2


def test_overview_matches_key():
    via_key = run_key("dataset_overview", {})
    assert api.overview(api.load(TRAIN_CSV)).model_dump(mode="json") == via_key


def test_check_matches_key():
    via_key = run_key("train_test_check", {})
    res = api.check(api.load(TRAIN_CSV), api.load(TEST_CSV))
    assert res.model_dump(mode="json") == via_key


def test_transform():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert api.transform(df, "drop_columns", columns=["a"]).columns.tolist() == ["b"]
    assert "drop_columns" in [t["op"] for t in api.list_transforms()]


def test_repr_html():
    res = api.overview(api.load(TRAIN_CSV))
    html = res._repr_html_()
    assert "rows" in html and "Overview / columns" in html
    assert html.count("cdn.plot.ly") == 1  # plotly.js loaded once
    assert "of 41 rows" not in html  # head table has 5 rows only
    assert Result(text="<b>")._repr_html_() == "<pre>&lt;b&gt;</pre>"
