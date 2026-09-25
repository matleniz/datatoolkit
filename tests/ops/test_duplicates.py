import pandas as pd

from dtk_engine.ops.duplicates import (
    GROUP_COLUMN,
    conflicts,
    duplicate_groups,
    exact_duplicates,
)

DF = pd.DataFrame(
    {
        "id": [1, 1, 2, 3, 3, 3],
        "city": ["a", "a", "b", "c", "d", "c"],
        "v": [1, 1, 2, 3, 3, 3],
    }
)


def test_exact():
    n, rows = exact_duplicates(DF)
    assert n == 2
    assert len(rows) == 4


def test_groups_keep_false():
    g = duplicate_groups(DF, ["id"])
    assert len(g) == 5
    assert g[GROUP_COLUMN].nunique() == 2


def test_conflicts_name_disagreeing_columns():
    by_col, rows = conflicts(DF, ["id"])
    assert by_col["column"].tolist() == ["city"]
    assert by_col["n_groups"].tolist() == [1]
    assert set(rows["id"]) == {3}


def test_no_duplicates():
    g = duplicate_groups(DF, ["v", "city"])
    assert set(g["id"]) == {1, 3}
    by_col, rows = conflicts(DF.drop_duplicates("id"), ["id"])
    assert by_col.empty and rows.empty
