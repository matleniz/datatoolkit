import pandas as pd
import pytest
from dtk_engine.ops.compare import (
    ISSUE_FIELDS,
    auto_id_columns,
    category_shift,
    compare_columns,
    find_issues,
    is_row_counter,
    numeric_shift,
    overlap,
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
    assert shift["pct_test_rows_unseen"] == 75.0
    assert shift["train_only_categories"] == "C"
    # 1 vs "1" is a dtype issue, not a new category
    assert (
        category_shift(pd.Series([1, 2]), pd.Series(["1", "2"]))["n_unseen_categories"]
        == 0
    )


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
