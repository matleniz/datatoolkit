import numpy as np
import pandas as pd
import pytest

from dtk_engine.keys.dataset_overview import overview_result
from dtk_engine.ops import alerts as A


def _profile(**cols):
    rows = [
        {
            "column": name,
            "semantic_type": c.get("semantic_type", "numeric"),
            "pct_missing": c.get("pct_missing", 0.0),
            "n_unique": c.get("n_unique", 10),
        }
        for name, c in cols.items()
    ]
    return pd.DataFrame(rows, columns=["column", "semantic_type", "pct_missing", "n_unique"])


def _numeric(**skews):
    return pd.DataFrame({"column": list(skews), "skew": list(skews.values())})


def _alerts(profile=None, numeric=None, n_rows=100, n_dups=0):
    profile = _profile(a={}) if profile is None else profile
    numeric = _numeric() if numeric is None else numeric
    return A.dataset_alerts(profile, numeric, n_rows, n_dups)


def _find(df, check):
    return df[df["check"] == check].to_dict("records")


def test_no_alert_on_clean_input():
    out = _alerts()
    assert list(out.columns) == ["severity", "check", "column", "message"]
    assert out.empty


@pytest.mark.parametrize(
    ("pct", "expected"),
    [
        (A.MISSING_INFO_PCT - 0.01, None),
        (A.MISSING_INFO_PCT, "info"),
        (A.MISSING_WARNING_PCT - 0.01, "info"),
        (A.MISSING_WARNING_PCT, "warning"),
    ],
)
def test_missing_thresholds(pct, expected):
    found = _find(_alerts(_profile(a={"pct_missing": pct})), "missing")
    assert [f["severity"] for f in found] == ([expected] if expected else [])


def test_entirely_missing_is_one_error_not_constant():
    out = _alerts(_profile(a={"pct_missing": 100.0, "semantic_type": "constant"}))
    assert [(r["severity"], r["check"]) for r in out.to_dict("records")] == [("error", "empty")]


def test_constant_column():
    out = _alerts(_profile(a={"semantic_type": "constant", "n_unique": 1}))
    assert [(r["severity"], r["check"], r["column"]) for r in out.to_dict("records")] == [
        ("warning", "constant", "a")
    ]


@pytest.mark.parametrize(
    ("pct", "expected"),
    [(0.0, "warning"), (0.01, "info"), (95.0, "info"), (99.9, "info")],
)
def test_constant_with_missing_is_a_presence_flag(pct, expected):
    prof = _profile(a={"semantic_type": "constant", "n_unique": 1, "pct_missing": pct})
    found = _find(_alerts(prof), "constant")
    assert [f["severity"] for f in found] == [expected]
    if expected == "info":
        assert "presence flag" in found[0]["message"]


def test_presence_flag_keeps_its_missing_alert():
    prof = _profile(a={"semantic_type": "constant", "n_unique": 1, "pct_missing": 95.0})
    out = _alerts(prof)
    assert [(r["severity"], r["check"]) for r in out.to_dict("records")] == [
        ("warning", "missing"),
        ("info", "constant"),
    ]


def test_id_like_column():
    out = _alerts(_profile(a={"semantic_type": "id_like"}))
    assert [(r["severity"], r["check"]) for r in out.to_dict("records")] == [("info", "id_like")]


@pytest.mark.parametrize(
    ("n_unique", "fires"),
    [(A.HIGH_CARDINALITY_UNIQUE - 1, False), (A.HIGH_CARDINALITY_UNIQUE, True)],
)
def test_high_cardinality_threshold(n_unique, fires):
    prof = _profile(a={"semantic_type": "categorical", "n_unique": n_unique})
    assert bool(_find(_alerts(prof), "high_cardinality")) is fires


def test_high_cardinality_only_for_categorical():
    prof = _profile(a={"semantic_type": "numeric", "n_unique": 10_000})
    assert not _find(_alerts(prof), "high_cardinality")


@pytest.mark.parametrize(
    ("n_dups", "expected"),
    [(0, None), (1, "info"), (4, "info"), (5, "warning")],
)
def test_duplicate_thresholds(n_dups, expected):
    found = _find(_alerts(n_rows=100, n_dups=n_dups), "duplicates")
    assert [f["severity"] for f in found] == ([expected] if expected else [])
    assert A.DUPLICATE_ROWS_WARNING_PCT == 5.0


@pytest.mark.parametrize(
    ("skew", "expected"),
    [
        (A.SKEW_INFO - 0.01, None),
        (A.SKEW_INFO, "info"),
        (-A.SKEW_INFO, "info"),
        (-10.0, "info"),
        (10.0, "info"),
        (float("nan"), None),
    ],
)
def test_skew_thresholds(skew, expected):
    found = _find(_alerts(numeric=_numeric(a=skew)), "skew")
    assert [f["severity"] for f in found] == ([expected] if expected else [])


def test_sorted_error_warning_info():
    prof = _profile(
        a={"semantic_type": "id_like"},
        b={"pct_missing": 100.0},
        c={"semantic_type": "constant"},
    )
    assert list(_alerts(prof)["severity"]) == ["error", "warning", "info"]


def test_overview_result_has_alerts_table_on_real_frame():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "const": 1,
            "mostly_nan": [np.nan] * 300 + [1.0] * 100,
            "heavy": rng.lognormal(0, 2, n),
            "cat": [f"c{i % 60}" for i in range(n)],
        }
    )
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    res = overview_result(df).model_dump()
    table = next(t for t in res["tables"] if t["title"] == "alerts")
    got = {(r["check"], r["column"]) for r in table["records"]}
    assert {
        ("id_like", "id"),
        ("constant", "const"),
        ("missing", "mostly_nan"),
        ("skew", "heavy"),
        ("high_cardinality", "cat"),
        ("duplicates", None),
    } <= got
    assert table["group"] == "Overview"


def test_overview_alerts_table_empty_but_present_on_demo():
    res = overview_result(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.5]})).model_dump()
    assert any(t["title"] == "alerts" for t in res["tables"])
