"""Integer column labels (header=None CSVs, hand-built frames): column names are
strings from the sources / notebook boundary on, so every overview group works."""

import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key

N = 40


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            0: range(N),
            1: np.linspace(0, 1, N) ** 2,
            2: ["a", "b"] * (N // 2),
            3: pd.date_range("2024-01-01", periods=N),
            4: [f"a longer free text sentence number {i}" for i in range(N)],
            5: [None] * N,
        }
    )


def _headerless(tmp_path) -> dict:
    path = tmp_path / "h.csv"
    _frame().drop(columns=[3, 5]).to_csv(path, header=False, index=False)
    return {"kind": "csv", "path": str(path), "header": None}


def _assert_overview(res) -> None:
    if isinstance(res, dict):  # run_key returns the JSON payload
        tables = {t["title"]: t["records"] for t in res["tables"]}
    else:
        tables = {t.title: t.records for t in res.tables}
    assert [r["column"] for r in tables["columns"]][:2] == ["0", "1"]
    assert "1" in {r["column"] for r in tables["numeric stats"]}
    assert "text stats" in tables and "alerts" in tables


def test_headerless_csv_loads_with_string_labels(tmp_path):
    df = api.load(_headerless(tmp_path))
    assert list(df.columns) == ["0", "1", "2", "3"]


def test_overview_on_headerless_csv_via_key(tmp_path):
    res = run_key("dataset_overview", {"source": _headerless(tmp_path)})
    _assert_overview(res)


def test_overview_on_int_labelled_frame():
    _assert_overview(api.overview(_frame()))


@pytest.mark.parametrize(
    "call",
    [
        lambda df: api.duplicates(df),
        lambda df: api.inconsistencies(df),
        lambda df: api.missing(df),
        lambda df: api.outliers(df),
        lambda df: api.correlations(df),
        lambda df: api.distribution(df),
    ],
)
def test_other_keys_on_int_labelled_frame(call):
    call(_frame())


def test_string_labels_unchanged_frame_is_not_copied():
    df = pd.DataFrame({"a": [1]})
    from dtk_engine.sources import string_columns

    assert string_columns(df) is df
