import numpy as np
import pandas as pd

from dtk_engine.ops.outliers import (
    flagged_rows,
    isolation_forest,
    numeric_columns,
    univariate_outliers,
)


def _frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(0, 1, 500), "y": rng.normal(10, 2, 500)})
    df.loc[0, "x"] = 50.0
    df.loc[1, ["x", "y"]] = [-40.0, -30.0]
    return df


def test_univariate_counts_planted_outliers():
    df = _frame()
    t = univariate_outliers(df, ["x", "y"]).set_index("column")
    assert t.loc["x", "n_iqr"] >= 2 and t.loc["x", "n_z"] == 2
    assert t.loc["x", "upper_fence"] > 1 and t.loc["x", "pct_z"] == 0.4
    strict = univariate_outliers(df, ["x"], iqr_k=100).set_index("column")
    assert strict.loc["x", "n_iqr"] == 0  # wider fences flag fewer


def test_constant_and_empty_columns():
    df = pd.DataFrame({"c": [1.0] * 10, "e": [np.nan] * 10})
    t = univariate_outliers(df, ["c", "e"])
    assert t["n_z"].tolist() == [0, 0] and t["count"].tolist() == [10, 0]


def test_isolation_forest_flags_planted_rows_deterministically():
    df = _frame()
    a = isolation_forest(df, ["x", "y"], contamination=0.01, random_state=0)
    b = isolation_forest(df, ["x", "y"], contamination=0.01, random_state=0)
    assert a.equals(b)
    assert {0, 1} <= set(a.index[a["flagged"]])
    assert a["flagged"].sum() == 5
    rows = flagged_rows(df, a, ["x", "y"])
    assert rows["row"].iloc[0] in (0, 1)


def test_isolation_forest_nothing_to_fit():
    assert isolation_forest(pd.DataFrame({"a": [np.nan] * 5}), ["a"], 0.1, 0).empty


def test_numeric_columns_skip_ids_and_text():
    df = pd.DataFrame(
        {
            "id": range(100),
            "v": np.random.default_rng(0).normal(size=100),
            "t": ["a"] * 100,
        }
    )
    assert numeric_columns(df) == ["v"]
