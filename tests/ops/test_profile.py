import numpy as np
import pandas as pd
import pytest
from dtk_engine.ops.profile import SEMANTIC_TYPES, column_profile, semantic_type


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
        (["id1", "id2", "id3", "id4"], "id_like"),
        ([10, 11, 12, 13], "id_like"),
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
