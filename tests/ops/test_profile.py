import numpy as np
import pandas as pd
import pytest
from dtk_engine.ops.profile import (
    ID_MIN_NON_NULL,
    SEMANTIC_TYPES,
    column_profile,
    pct_numeric_parsable,
    semantic_type,
)

N = ID_MIN_NON_NULL


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.5, 2.5, 3.0], "numeric"),
        ([True, False, True, True], "boolean"),
        ([0, 1, 1, 0], "boolean"),
        (pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]), "datetime"),
        (["2024-01-01", "2024-02-01", "2024-03-01"], "datetime"),
        (["a", "b", "a", "b", "a", "c"], "categorical"),
        (["the quick fox", "a lazy dog", "hello world", "foo bar"], "text"),
        ([f"id{i}" for i in range(N)], "id_like"),
        (list(range(10, 10 + N)), "id_like"),
        # Index 0..n-1 with one duplicated row is still an identifier.
        ([*range(N), 0], "id_like"),
        # Too few non-null values to call it an identifier (demo `Cabin`).
        (["C85", "E46", "B28", "C123", "D33", *[None] * 36], "text"),
        ([10, 11, 12, 13], "numeric"),
        # All-distinct integers spread over a wide range: a feature, not an id.
        ([i * 997 + i % 7 for i in range(N)], "numeric"),
        # Entity key repeated over rows (e.g. patient_id).
        ([f"P{i % 150:04d}" for i in range(1500)], "group_id"),
        ([i % 150 for i in range(1500)], "group_id"),
        # Few categories stay categorical; sparse spread integers stay numeric.
        ([f"c{i % 20}" for i in range(1500)], "categorical"),
        ([(i % 150) * 1000 for i in range(1500)], "numeric"),
        # Year strings are not dates.
        (["2024", "2023", "2024", "2022", "2023", "2024"], "categorical"),
        # Numbers polluted by a token stay text-like.
        (["12", "unknown", "31", "45", "7"], "text"),
        (["x", "x", None], "constant"),
        ([None, None], "constant"),
    ],
)
def test_semantic_type(values, expected):
    assert semantic_type(pd.Series(values)) == expected
    assert expected in SEMANTIC_TYPES


def test_column_profile():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 3.0], "b": ["x", "y", "x", "x"]})
    prof = column_profile(df).set_index("column")
    assert list(prof.index) == ["a", "b"]
    assert prof.loc["a", "n_missing"] == 1
    assert prof.loc["a", "pct_missing"] == 25.0
    assert prof.loc["a", "n_unique"] == 2
    assert prof.loc["a", "semantic_type"] == "numeric"
    assert prof.loc["b", "semantic_type"] == "categorical"
    assert prof.loc["b", "sample_values"] == "x, y"


def test_column_profile_empty_frame():
    prof = column_profile(pd.DataFrame({"a": []}))
    assert prof.loc[0, "pct_missing"] == 0.0


def test_column_profile_pct_numeric_parsable():
    df = pd.DataFrame(
        {
            "age": ["12", "unknown", "31", " 45", None],
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": ["a", "b", "a", "b", "a"],
            "flag": [True, False, True, True, False],
        }
    )
    prof = column_profile(df).set_index("column")
    assert prof.loc["age", "pct_numeric_parsable"] == 75.0
    assert prof.loc["age", "semantic_type"] == "text"
    assert prof.loc["num", "pct_numeric_parsable"] == 100.0
    assert prof.loc["cat", "pct_numeric_parsable"] == 0.0
    assert prof.loc["flag", "pct_numeric_parsable"] == 0.0


def test_pct_numeric_parsable_empty():
    assert pct_numeric_parsable(pd.Series([None, None], dtype=object)) == 0.0
