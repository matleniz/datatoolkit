"""DuckDB pushdown (``workspace.large``, extra ``large``): profile, shape and export of a
``CsvSource(engine="duckdb")`` role without a pandas frame; same answers as pandas."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

from dtk_engine import contract, save_workspace
from dtk_engine.sources import load
from dtk_engine.workspace import large

EXACT = ("kind", "count", "missing", "distinct", "histogram", "iqr_bounds", "outliers", "skewed")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(10, 3, n).round(2),
            "skew": rng.exponential(2, n).round(3),
            "n": rng.integers(0, 50, n),
            "cat": rng.choice(["a", "b", "c", "d"], n),
            "flag": rng.choice([True, False], n),
            'we"ird': rng.integers(0, 3, n),
        }
    )
    df.loc[::17, "x"] = np.nan
    df.loc[::11, "cat"] = None
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    return path


def ws_for(path, engine, steps=(), **extra):
    return {
        "name": "w",
        "datasets": {"train": {"x": {"kind": "csv", "path": str(path), "engine": engine}}},
        "steps": list(steps),
        **extra,
    }


FILTER = {
    "op": "filter_rows",
    "target": "train",
    "params": {
        "conditions": [
            {"column": "n", "op": "ge", "value": 5},
            {"column": "cat", "op": "notin", "value": ["d"]},
            {"column": "x", "op": "ne", "value": 10.0},
        ]
    },
}
DROP = {"op": "drop_columns", "target": "both", "params": {"columns": ["skew"]}}


def by_name(profiles):
    return {c["name"]: c for c in profiles["columns"]}


@pytest.mark.parametrize("steps", [[], [FILTER], [FILTER, DROP]])
def test_profile_matches_pandas(csv, steps):
    duck = by_name(contract.column_profiles(ws_for(csv, "duckdb", steps), "train"))
    pand = by_name(contract.column_profiles(ws_for(csv, "pandas", steps), "train"))
    assert duck.keys() == pand.keys()
    for name in pand:
        for key in EXACT:
            assert duck[name][key] == pytest.approx(pand[name][key]), (name, key)
        as_counts = lambda p: {json.dumps(t["value"]): t["count"] for t in p["top_values"] or []}
        assert as_counts(duck[name]) == as_counts(pand[name]), name


def test_profile_does_not_build_the_frame(csv, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pandas frame built")

    monkeypatch.setattr("dtk_engine.workspace.inspect._frame_at", boom)
    monkeypatch.setattr("dtk_engine.sources.csv_duckdb.read_csv_duckdb", boom)
    out = contract.column_profiles(ws_for(csv, "duckdb", [FILTER]), "train")
    assert len(out["columns"]) == 7 and "sampled" not in out


def test_sampled_heuristics_exact_counts(csv, monkeypatch):
    monkeypatch.setattr(large, "DEFAULT_SAMPLE", 50)
    out = contract.column_profiles(ws_for(csv, "duckdb"), "train")
    assert (out["sampled"], out["n_rows_source"], out["n_sampled_out"]) == (True, 400, 350)
    assert "estimates" in out["note"]
    assert by_name(out)["id"]["count"] == 400  # exact, not the sample's 50
    again = contract.column_profiles(ws_for(csv, "duckdb"), "train")
    assert out == again  # seeded
    named = contract.column_profiles(ws_for(csv, "duckdb"), "train", columns=["n", "x"])
    assert [c["name"] for c in named["columns"]] == ["n", "x"]


def test_shape_matches_pandas(csv):
    expected = load_shape(ws_for(csv, "pandas", [FILTER, DROP]))
    ws = contract.parse_workspace(ws_for(csv, "duckdb", [FILTER, DROP]))
    assert contract._role_shape(ws, "train") == expected


def load_shape(ws):
    parsed = contract.parse_workspace(ws)
    df = contract.workspace_frame(parsed, "train")
    return [len(df), df.shape[1]]


def test_export_parquet_matches_pandas(csv, tmp_path):
    save_workspace(ws_for(csv, "duckdb", [FILTER, DROP]))
    manifest = contract.export_workspace("w", str(tmp_path / "out"))
    got = pd.read_parquet(tmp_path / "out" / "processed" / "train.parquet")
    expected = contract.workspace_frame(
        contract.parse_workspace(ws_for(csv, "pandas", [FILTER, DROP])), "train"
    )
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    assert manifest["outputs"]["train"]["rows"] == len(expected)
    assert [s["fitted_on"] for s in manifest["steps"]] == ["train", "train"]


def test_export_with_csv_format_falls_back(csv, tmp_path, monkeypatch):
    monkeypatch.setattr(large, "export_parquet", lambda *a: pytest.fail("pushed down"))
    save_workspace(ws_for(csv, "duckdb", [FILTER]))
    contract.export_workspace("w", str(tmp_path / "out"), formats=["parquet", "csv"])
    assert (tmp_path / "out" / "processed" / "train.csv").exists()


@pytest.mark.parametrize(
    "step",
    [
        {"op": "sort_rows", "target": "train", "params": {"by": ["x"]}},
        {"op": "drop_duplicates", "target": "train", "params": {"keep": "none"}},
        {"op": "filter_rows", "target": "train", "params": {"conditions": [{"column": "cat", "op": "gt", "value": 3}]}},
        {"op": "filter_rows", "target": "train", "params": {"conditions": [{"column": "n", "op": "eq", "value": True}]}},
        {"op": "filter_rows", "target": "train", "params": {"conditions": [{"column": "n", "op": "isin", "value": []}]}},
        {"op": "filter_rows", "target": "train", "params": {"conditions": [{"column": "nope", "op": "isna"}]}},
        {"op": "drop_columns", "target": "train", "params": {"columns": ["nope"]}},
    ],
)
def test_unpushable_steps_fall_back(csv, step):
    ws = contract.parse_workspace(ws_for(csv, "duckdb", [step]))
    assert large.plan(ws, "train", ws.steps) is None


def test_label_file_merge_and_label_last_fall_back(csv, tmp_path):
    y = tmp_path / "y.csv"
    y.write_text("y\n" + "0\n" * 400)
    with_y = ws_for(csv, "duckdb")
    with_y["datasets"]["train"]["y"] = {"kind": "csv", "path": str(y)}
    assert large.plan(contract.parse_workspace(with_y), "train", []) is None
    last = contract.parse_workspace(ws_for(csv, "duckdb", label_last=True))
    assert large.plan(last, "train", []) is None


def test_values_are_parameters_not_sql(csv):
    evil = "x'; DROP TABLE t; --"
    step = {
        "op": "filter_rows",
        "target": "train",
        "params": {"conditions": [{"column": "cat", "op": "eq", "value": evil}]},
    }
    assert contract.column_profiles(ws_for(csv, "duckdb", [step]), "train")["columns"][0]["count"] == 0
    ws = contract.parse_workspace(ws_for(csv, "duckdb", [step]))
    plan = large.plan(ws, "train", ws.steps)
    assert evil not in plan.scan() and plan.params == [evil]
    assert large.shape(plan) == [0, 7]


def test_quoted_column_name_and_unknown_column_in_profile(csv):
    out = contract.column_profiles(ws_for(csv, "duckdb"), "train", columns=['we"ird'])
    assert out["columns"][0]["name"] == 'we"ird'
    with pytest.raises(Exception, match="columns not in the frame"):
        contract.column_profiles(ws_for(csv, "duckdb"), "train", columns=["x; DROP"])


def test_pandas_engine_never_imports_duckdb(csv):
    code = (
        "import sys, json; from dtk_engine import contract;"
        f"ws = {ws_for(csv, 'pandas', [FILTER])!r};"
        "contract.column_profiles(ws, 'train'); contract._role_shape(contract.parse_workspace(ws), 'train');"
        "print('duckdb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_loaded_frame_unchanged_by_pushdown(csv):
    a = load(contract.parse_workspace(ws_for(csv, "duckdb")).datasets.train.x)
    assert len(a) == 400
