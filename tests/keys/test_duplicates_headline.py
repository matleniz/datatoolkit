"""The duplicates key always states a one-line headline (#232)."""

import pandas as pd

from dtk_engine.keys.duplicates import duplicates_result


def _frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({"id": range(n), "value": [i % 7 for i in range(n)]})


def test_no_duplicates_headline():
    result = duplicates_result(_frame())
    assert result.headline == "No exact duplicate rows; no duplicate groups on id (auto)"


def test_exact_only_headline_explicit_empty_subset():
    df = pd.concat([_frame(), _frame().head(5)], ignore_index=True)
    result = duplicates_result(df, [])
    assert result.headline == (
        "5 exact duplicate rows (4.8 %); no identity columns detected"
    )


def test_auto_subset_headline():
    df = pd.concat([_frame(), _frame().head(5)], ignore_index=True)
    result = duplicates_result(df)
    assert result.headline == "5 exact duplicate rows (4.8 %); 5 groups on id (auto)"


def test_explicit_subset_with_conflicts_headline():
    df = pd.DataFrame({"k": [1, 1, 2, 3], "v": ["a", "b", "c", "c"]})
    result = duplicates_result(df, ["k"])
    assert result.headline == (
        "No exact duplicate rows; 1 group on k, 1 with conflicting values"
    )


def test_empty_frame_headline():
    result = duplicates_result(pd.DataFrame({"id": [], "v": []}))
    assert result.headline.startswith("No rows")
