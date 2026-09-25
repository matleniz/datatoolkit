"""Object columns holding lists / dicts / arrays (nested JSON, Parquet) or bytes
(GeoParquet WKB): every key returns a Result instead of crashing."""

import json

import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.ops.duplicates import exact_duplicates
from dtk_engine.ops.missing import sentinel_counts
from dtk_engine.ops.profile import (
    SAMPLE_MAX_CHARS,
    column_profile,
    hashable_frame,
    object_kind,
    semantic_type,
)
from dtk_engine.transform_registry import get_transform

N = 30


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(N),
            "tags": [["a", "b"] if i % 2 else ["c"] for i in range(N)],
            "user": [{"name": f"u{i % 3}"} for i in range(N)],
            "arr": [np.array([i % 4, 1]) for i in range(N)],
            "geometry": [
                b"\x01\x01\x00\x00\x00" + bytes([0x80 + i % 5]) * 30 for i in range(N)
            ],
            "cat": ["x", "y", "z"] * (N // 3),
            "y": rng.normal(size=N),
        }
    )
    df.loc[3, "tags"] = None
    return df


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([[1, 2], [3], None, [1, 2]], "nested"),
        ([{"a": 1}, {"a": 2}], "nested"),
        ([np.array([1, 2]), np.array([3])], "nested"),
        ([(1, 2), (3, 4)], "nested"),
        ([b"\x80\x01", b"\x80\x02", None], "binary"),
        ([[1], [1], None], "constant"),
    ],
)
def test_semantic_type_nested_and_binary(values, expected):
    assert semantic_type(pd.Series(values, dtype=object)) == expected


def test_object_kind_ignores_plain_columns():
    assert object_kind(pd.Series(["a", None])) is None
    assert object_kind(pd.Series([1.0, 2.0])) is None


def test_column_profile_counts_and_short_samples():
    prof = column_profile(_frame()).set_index("column")
    assert prof.loc["tags", "n_unique"] == 2
    assert prof.loc["arr", "n_unique"] == 4
    assert prof.loc["arr", "sample_values"].startswith("[0, 1]")
    assert prof.loc["geometry", "semantic_type"] == "binary"
    assert prof.loc["geometry", "pct_numeric_parsable"] == 0.0
    samples = prof.loc["geometry", "sample_values"].split(", ")
    assert all(len(s) <= SAMPLE_MAX_CHARS for s in samples)


def test_hashable_frame_compares_by_value():
    df = pd.DataFrame({"a": [np.array([1, 2]), np.array([1, 2]), [1, 2]]})
    assert hashable_frame(df)["a"].nunique() == 1
    plain = pd.DataFrame({"a": [1, 2]})
    assert hashable_frame(plain) is plain


def test_duplicates_on_nested_rows():
    df = _frame()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    n, rows = exact_duplicates(df)
    assert n == 1 and len(rows) == 2
    t = get_transform("drop_duplicates")
    assert len(t.fit_apply(df, t.parse({"keep": "none"}))) == len(df) - 2


def test_sentinels_skip_binary_and_nested():
    df = pd.DataFrame({"g": [b"\x80", b"\x81"], "l": [["unknown"], []]})
    assert sentinel_counts(df).empty


@pytest.mark.parametrize(
    "call",
    [
        api.overview,
        api.missing,
        api.duplicates,
        api.inconsistencies,
        api.outliers,
        lambda df: api.check(df, df.copy()),
        lambda df: api.advise(df, target="y"),
        lambda df: api.select_features(df, target="y"),
    ],
)
def test_every_api_door_survives(call):
    json.dumps(call(_frame()).model_dump())


def test_overview_types_and_advisor_drops():
    df = _frame()
    types = {
        r["column"]: r["semantic_type"] for r in api.overview(df).tables[0].records
    }
    assert (types["tags"], types["user"], types["arr"]) == ("nested",) * 3
    assert types["geometry"] == "binary"
    recs = api.advise(df, target="y").tables[0].records
    drops = {r["column"] for r in recs if r["op"] == "drop_columns"}
    assert {"tags", "user", "arr", "geometry"} <= drops


def test_run_key_on_nested_parquet(tmp_path):
    path = tmp_path / "nested.parquet"
    _frame().to_parquet(path)
    source = {"kind": "parquet", "path": str(path)}
    for key_id in (
        "dataset_overview",
        "missing_values",
        "duplicates",
        "inconsistencies",
        "outliers",
        "preprocessing_advisor",
    ):
        json.dumps(run_key(key_id, {"source": source}))
