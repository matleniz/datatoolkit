import numpy as np
import pandas as pd
import pytest

from dtk_engine.ops.compare import (
    CATEGORICAL_DRIFT_FIELDS,
    ISSUE_FIELDS,
    KS_WARNING,
    PSI_INFO,
    PSI_WARNING,
    SEVERITIES,
    SMD_WARNING,
    auto_id_columns,
    categorical_drift,
    category_shift,
    compare_columns,
    drift_columns,
    find_issues,
    histogram_pair,
    is_row_counter,
    ks_statistic,
    numeric_drift,
    numeric_shift,
    overlap,
    psi,
    schema_diff,
)


def test_schema_diff():
    train = pd.DataFrame(columns=["id", "a", "b", "y"])
    test = pd.DataFrame(columns=["b", "a", "id", "extra"])
    diff = schema_diff(train, test)
    assert diff["only_train"] == ["y"]
    assert diff["only_test"] == ["extra"]
    assert diff["common"] == ["id", "a", "b"]
    assert diff["order_differs"]
    assert not schema_diff(train, train)["order_differs"]


def test_numeric_shift():
    shift = numeric_shift(pd.Series([0.0, 10.0, None]), pd.Series([-1.0, 5, 11, 20]))
    assert (shift["min_train"], shift["max_train"]) == (0.0, 10.0)
    assert (shift["min_test"], shift["max_test"]) == (-1.0, 20.0)
    assert shift["pct_test_out_of_range"] == 75.0
    assert shift["mean_train"] == 5.0


def test_numeric_shift_skips_non_numeric():
    assert numeric_shift(pd.Series([1.0, 2.0]), pd.Series(["1", "x"])) == {}
    assert numeric_shift(pd.Series([True, False]), pd.Series([True, True])) == {}


def test_category_shift_compares_as_strings():
    shift = category_shift(
        pd.Series(["S", "C", "C", None]), pd.Series(["S", "Q", "Q", 1])
    )
    assert shift["n_unseen_categories"] == 2
    assert shift["unseen_categories"] == "1, Q"
    assert shift["unseen_category_counts"] == "Q (2), 1 (1)"
    assert shift["_only_in_test"] == [{"value": "Q", "count": 2}, {"value": "1", "count": 1}]
    assert shift["pct_test_rows_unseen"] == 75.0
    assert shift["train_only_categories"] == "C"
    assert shift["near_match_hint"] is None
    # 1 vs "1" is a dtype issue, not a new category
    assert (
        category_shift(pd.Series([1, 2]), pd.Series(["1", "2"]))["n_unseen_categories"]
        == 0
    )


def test_category_shift_near_match_trailing_punctuation():
    """UCI adult-style labels: test values differ only by a trailing '.'."""
    train = pd.Series(["<=50K", ">50K", "<=50K", ">50K"])
    test = pd.Series(["<=50K.", ">50K.", "<=50K.", ">50K.", "<=50K."])
    shift = category_shift(train, test)
    assert shift["n_unseen_categories"] == 2
    assert shift["pct_test_rows_unseen"] == 100.0
    assert {r["value"] for r in shift["_only_in_test"]} == {"<=50K.", ">50K."}
    assert shift["_near_matches"] == [
        {"test": "<=50K.", "train": "<=50K"},
        {"test": ">50K.", "train": ">50K"},
    ]
    assert "standardize_text" in shift["near_match_hint"]
    assert "<=50K." in shift["near_match_hint"]


@pytest.mark.parametrize(
    ("train", "test", "expected"),
    [
        (range(5), range(3), True),
        (range(1, 6), range(1, 4), True),
        ([1, 2, 2, 3], [0, 1], True),  # duplicated rows keep a counter a counter
        (range(5), range(10, 13), False),  # test does not restart
        ([1, 5, 9], [1, 5], False),
        (["a", "b"], ["a"], False),
    ],
)
def test_is_row_counter(train, test, expected):
    assert is_row_counter(pd.Series(list(train)), pd.Series(list(test))) is expected


def test_overlap_rows_and_ids():
    train = pd.DataFrame({"pid": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    test = pd.DataFrame({"pid": [3, 3, 4, 5], "x": [3.0, 9.0, 9.0, 9.0]})
    table = overlap(train, test, ["pid", "absent"]).set_index("kind")
    rows, ids = table.loc["rows"], table.loc["id"]
    assert rows["n_test_in_train"] == 1 and rows["pct_test_in_train"] == 25.0
    assert ids["column"] == "pid"
    assert (ids["n_test"], ids["n_test_in_train"]) == (3, 1)
    assert not ids["row_counter"]


def test_overlap_rows_ignore_id_and_counter_columns():
    train = pd.DataFrame({"Index": range(5), "pid": range(10, 15), "x": list("abcde")})
    test = pd.DataFrame({"Index": range(3), "pid": [90, 91, 92], "x": list("zzc")})
    rows = overlap(train, test, ["pid"]).set_index("kind").loc["rows"]
    assert rows["n_test_in_train"] == 1


def test_overlap_skips_rows_when_only_ids_are_common():
    df = pd.DataFrame({"pid": [1, 2, 3]})
    assert list(overlap(df, df, ["pid"])["kind"]) == ["id"]


def test_compare_columns_union_and_auto_ids():
    train = pd.DataFrame(
        {
            "pid": range(10, 30),
            "cat": ["a", "b"] * 10,
            "num": [1.5, 2.5, 3.5, 4.5] * 5,
            "y": 0,
        }
    )
    test = pd.DataFrame(
        {"pid": range(30, 40), "cat": ["a", "c"] * 5, "num": [None] * 5 + [50] * 5}
    ).assign(new=1)
    cols = compare_columns(train, test).set_index("column")
    assert list(cols.index) == ["pid", "cat", "num", "y", "new"]
    assert not cols.at["y", "in_test"] and not cols.at["new", "in_train"]
    assert cols.at["num", "pct_missing_delta"] == 50.0
    assert cols.at["num", "pct_test_out_of_range"] == 100.0
    assert pd.isna(cols.at["pid", "pct_test_out_of_range"])  # ids skip ranges
    assert cols.at["cat", "n_unseen_categories"] == 1
    assert auto_id_columns(cols.reset_index()) == ["pid"]


def test_auto_id_columns_accepts_group_id():
    columns = pd.DataFrame(
        {
            "column": ["patient_id", "x", "only"],
            "in_train": [True, True, True],
            "in_test": [True, True, False],
            "semantic_train": ["group_id", "numeric", "id_like"],
            "semantic_test": ["categorical", "numeric", None],
        }
    )
    assert auto_id_columns(columns) == ["patient_id"]


def _issues(train, test, id_columns, missing_ids=()):
    cols = compare_columns(train, test)
    return find_issues(train, test, cols, overlap(train, test, id_columns), missing_ids)


def test_find_issues_severities_and_order():
    train = pd.DataFrame(
        {"pid": [101, 205, 309], "age": [1.0, None, 3.0], "target": [0, 1, 0]}
    )
    test = pd.DataFrame(
        {"age": ["1", "2", "3"], "pid": [309, 412, 518], "leak": [1, 2, 3]}
    )
    issues = _issues(train, test, ["pid"], ["ghost"])
    assert list(issues.columns) == ISSUE_FIELDS
    found = {(r.severity, r.check, r.column) for r in issues.itertuples()}
    assert ("error", "schema", "age") in found  # dtype mismatch
    assert ("error", "overlap", "pid") in found  # entity leak
    assert ("error", "overlap", "ghost") in found  # requested id missing
    assert ("warning", "schema", "leak") in found  # only in test
    assert ("warning", "missing", "age") in found
    assert ("info", "schema", "target") in found  # probable target
    assert any(r.check == "schema" and pd.isna(r.column) for r in issues.itertuples())
    ranks = issues["severity"].map({"error": 0, "warning": 1, "info": 2})
    assert ranks.is_monotonic_increasing


def test_numeric_dtype_mismatch_is_info():
    train = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    test = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["1", "2", "3"]})
    issues = _issues(train, test, [])
    dtype = issues[issues["message"].str.startswith("dtype")].set_index("column")
    assert dtype.at["a", "severity"] == "info"
    assert "numeric on both sides" in dtype.at["a", "message"]
    assert dtype.at["b", "severity"] == "error"


def test_row_counter_overlap_is_info_not_leak():
    train = pd.DataFrame({"idx": range(10), "v": [1.5] * 10})
    test = pd.DataFrame({"idx": range(4), "v": [2.5] * 4})
    issues = _issues(train, test, ["idx"])
    idx = issues[issues["column"] == "idx"]
    assert list(idx["severity"]) == ["info"]


def test_no_issue_on_identical_tables():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    issues = _issues(df, df.copy(), [])
    # identical tables only share their rows
    assert list(issues["check"]) == ["overlap"]


def test_ks_and_psi_identical_and_disjoint():
    a = np.arange(100, dtype=float)
    assert ks_statistic(a, a) == 0.0
    assert psi(a, a) == pytest.approx(0.0)
    assert ks_statistic(a, a + 1000) == 1.0
    assert psi(a, a + 1000) > 1.0  # everything lands in the last train bin
    # constant train column: a shifted test is still detected
    assert psi(np.ones(20), np.full(20, 5.0)) > 1.0


def test_numeric_drift_stats_and_skips():
    train = pd.DataFrame({"x": np.arange(1000.0), "few": [1.0] + [None] * 999})
    test = pd.DataFrame({"x": np.arange(1000.0) + 500, "few": np.arange(1000.0)})
    out = numeric_drift(train, test, ["x", "few"]).set_index("column")
    x = out.loc["x"]
    assert x["smd"] == pytest.approx(500 / np.std(np.arange(1000.0), ddof=1))
    assert x["ks"] == pytest.approx(0.5)
    assert x["psi"] > PSI_WARNING
    assert x["pct_test_above_train_p99"] == 51.0  # test values 990..1499
    assert x["pct_test_below_train_p1"] == 0.0
    assert x["pct_test_outside_train_range"] == 50.0
    assert (x["p1_train"], x["median_test"]) == (pytest.approx(9.99), 999.5)
    assert out.loc["few", "skipped"] == "< 2 non-null values in train"
    assert pd.isna(out.loc["few", "psi"])
    assert pd.isna(x["skipped"])


def test_numeric_drift_ignores_nan():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, None, None]})
    same = numeric_drift(train, train, ["x"]).iloc[0]
    assert same["ks"] == 0.0 and same["smd"] == 0.0 and same["mean_train"] == 2.0


def test_categorical_drift_long_table():
    train = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 4 + [None]})
    test = pd.DataFrame({"c": ["a"] * 2 + ["b"] * 2 + ["z"] * 6})
    out = categorical_drift(train, test, ["c"], top=2)
    assert list(out.columns) == CATEGORICAL_DRIFT_FIELDS
    assert out["tvd"].iloc[0] == pytest.approx(0.6)  # full distribution, not top 2
    assert set(out["value"]) == {"z", "a"}  # top 2 by max share
    row = out.set_index("value").loc["a"]
    assert (row["pct_train"], row["pct_test"], row["diff"]) == (60.0, 20.0, -40.0)
    assert categorical_drift(train, test, []).empty


def test_drift_columns_exclude_ids_and_counters():
    train = pd.DataFrame(
        {
            "Index": range(30),
            "pid": [f"P{i}" for i in range(30)],
            "x": np.linspace(0, 1, 30),
            "cat": ["a", "b"] * 15,
        }
    )
    columns = compare_columns(train, train)
    numeric, categorical = drift_columns(train, train, columns)
    assert numeric == ["x"] and categorical == ["cat"]


def test_histogram_pair_shares_bins_and_normalizes():
    edges, d_train, d_test = histogram_pair(
        pd.Series([0.0, 1.0, 2.0]), pd.Series([1.0, 9.0, None]), bins=5
    )
    assert len(edges) == 6 and edges[0] == 0.0 and edges[-1] == 9.0
    assert d_train.sum() == pytest.approx(1.0) and d_test.sum() == pytest.approx(1.0)


def test_find_issues_drift_severities():
    train = pd.DataFrame({"x": np.arange(1000.0), "c": ["a"] * 500 + ["b"] * 500})
    test = pd.DataFrame({"x": np.arange(1000.0) + 500, "c": ["a"] * 900 + ["b"] * 100})
    columns = compare_columns(train, test)
    num = numeric_drift(train, test, ["x"])
    cat = categorical_drift(train, test, ["c"])
    issues = find_issues(train, test, columns, overlap(train, test, []), (), num, cat)
    drift = issues[issues["check"] == "drift"].set_index("column")
    assert drift.loc["x", "severity"] == "warning"
    assert "PSI" in drift.loc["x", "message"] and "KS" in drift.loc["x", "message"]
    assert drift.loc["c", "severity"] == "warning"
    assert len(drift) == 2  # one finding per column
    assert list(issues["severity"]) == sorted(issues["severity"], key=SEVERITIES.index)


def test_find_issues_drift_info_band():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 5000)})
    test = pd.DataFrame({"x": rng.normal(0, 1.1, 5000)})
    num = numeric_drift(train, test, ["x"]).iloc[0]
    assert num["smd"] < SMD_WARNING and num["ks"] < KS_WARNING
    assert num["psi"] < PSI_INFO
    out = find_issues(
        train,
        test,
        compare_columns(train, test),
        overlap(train, test, []),
        (),
        numeric_drift(train, test, ["x"]),
    )
    drift = out[out["check"] == "drift"]
    assert drift["severity"].tolist() == ["info"]  # 3-5% outside train p1-p99


def test_find_issues_without_drift_tables_unchanged():
    train = pd.DataFrame({"x": [1.0, 2.0]})
    columns = compare_columns(train, train)
    out = find_issues(train, train, columns, overlap(train, train, []))
    assert "drift" not in set(out["check"])
