import pandas as pd

from dtk_engine import api, run_key


def test_duplicates_defaults_on_demo():
    res = run_key("duplicates", {})
    assert res["metrics"]["n_exact_duplicates"] == 1
    assert {t["title"] for t in res["tables"]} >= {"exact duplicates"}


def test_inconsistencies_defaults_on_demo():
    res = run_key("inconsistencies", {})
    assert res["metrics"]["columns_checked"] > 0


def test_api_duplicates_subset_conflicts():
    df = pd.DataFrame({"k": [1, 1, 2], "v": ["a", "b", "c"]})
    res = api.duplicates(df, subset=["k"])
    assert res.metrics["n_conflict_groups"] == 1
    assert res.metrics["n_exact_duplicates"] == 0


def test_api_inconsistencies():
    df = pd.DataFrame({"c": ["Paris", "paris ", "Lyon"]})
    res = api.inconsistencies(df)
    assert res.metrics["n_merged_variants"] == 1


def test_api_inconsistencies_mixed_date_formats():
    df = pd.DataFrame({"d": ["2020-01-01", "01/15/2020", "15-01-2020", "garbage"]})
    res = api.inconsistencies(df)
    assert res.metrics["n_mixed_date_format_columns"] == 1
    assert "mixed date formats" in {t.title for t in res.tables}
    assert "parse_dates" in res.text
