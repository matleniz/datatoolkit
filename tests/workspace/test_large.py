"""DuckDB pushdown (``workspace.large``, extra ``large``): rows, profile, preview, shape and
export of a ``CsvSource(engine="duckdb")`` role without a pandas frame; same answers as
pandas (approximate fields flagged above ``large_stats.APPROX_FROM_ROWS`` rows)."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

from dtk_engine import contract, save_workspace
from dtk_engine.errors import KeyParamsError
from dtk_engine.sources import load
from dtk_engine.workspace import large, large_stats

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


@pytest.mark.parametrize("steps", [[], [FILTER, DROP]])
def test_column_notes_does_not_build_the_frame(csv, monkeypatch, steps):
    def boom(*a, **k):
        raise AssertionError("pandas frame built")

    monkeypatch.setattr("dtk_engine.workspace.inspect._frame_at", boom)
    monkeypatch.setattr("dtk_engine.sources.csv_duckdb.read_csv_duckdb", boom)
    ws = ws_for(csv, "duckdb", steps, notes={"columns": {"cat": "kept", "skew": "dropped"}})
    out = contract.column_notes(ws, "train")
    assert out == {"notes": {"cat": "kept"} if steps else {"cat": "kept", "skew": "dropped"}, "keys": {}}


def test_sample_rows_never_samples(csv):
    full = contract.column_profiles(ws_for(csv, "duckdb"), "train")
    asked = contract.column_profiles(ws_for(csv, "duckdb"), "train", sample_rows=50)
    assert (asked["sampled"], asked["n_rows_source"], asked["n_sampled_out"]) == (False, 400, 0)
    assert asked["columns"] == full["columns"] and "note" not in asked
    assert (full["engine"], full["n_rows"]) == ("duckdb", 400) and "sampled" not in full
    named = contract.column_profiles(ws_for(csv, "duckdb"), "train", columns=["n", "x"])
    assert [c["name"] for c in named["columns"]] == ["n", "x"]


@pytest.fixture
def rich(tmp_path):
    """Every text rule and dtype quirk the profile ports to SQL."""
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame(
        {
            "row_id": np.arange(n),
            "user_id": [f"u{i:05d}" for i in rng.permutation(n)],
            "price": rng.choice(["$1,250.00", "$3.50", "$990.10", "$12.00"], n),
            "pct": rng.choice(["12,5 %", "3,1 %", "7,0 %"], n),
            "num_text": rng.choice(["1,5", "2,25", "3"], n),
            "date": rng.choice(["2020-01-15", "15/01/2020", "Jan 3, 2021", "unknown"], n),
            "iso": rng.choice(["2020-01-15", "2021-02-01 10:00:00", "2019-12-31"], n),
            "city": rng.choice(["Paris", "paris ", "PARIS", "Lyon", "N/A", "-", "", "MISSING"], n),
            "age": rng.integers(0, 90, n).astype(float),
            "code": rng.choice([1, 2, 3, -999, 4, 5], n),
            "zeros": np.where(rng.random(n) < 0.5, 0, rng.integers(1, 100, n)),
            "flag": rng.choice([True, False], n),
            "flagn": rng.choice(["True", "False", ""], n),
            "grp": rng.integers(0, 150, n),
            "txt": [" ".join(rng.choice(["a", "b", "c"], 2)) for _ in range(n)],
            "const": 7,
            "ym": rng.choice(["2020-01", "2021-03"], n),
            "mixed": rng.choice(["12", "7", "x", "1900-01-01", "-1", "nan", "1_000"], n),
        }
    )
    df.loc[::7, "age"] = np.nan
    df.loc[::13, "grp"] = None
    df.loc[::9, "date"] = None
    path = tmp_path / "rich.csv"
    df.to_csv(path, index=False)
    return path


def comparable(profile):
    """A pandas-path profile field set (top values as a value -> count map)."""
    out = {k: v for k, v in profile.items() if k not in ("stats", "approximate")}
    out["top_values"] = {json.dumps(t["value"]): t["count"] for t in profile["top_values"] or []}
    return json.loads(json.dumps(out))


@pytest.mark.parametrize("grouped", [True, False])
def test_every_profile_field_matches_pandas(rich, monkeypatch, grouped):
    if not grouped:  # text rules on every cell instead of once per distinct value
        monkeypatch.setattr(large_stats, "DISTINCT_EXACT_MAX", 0)
    duck = by_name(contract.column_profiles(ws_for(rich, "duckdb"), "train"))
    pand = by_name(contract.column_profiles(ws_for(rich, "pandas"), "train"))
    assert duck.keys() == pand.keys()
    for name in pand:
        assert comparable(duck[name]) == comparable(pand[name]), name
        assert duck[name]["approximate"][0] == "kind"  # the SQL port of the kind rules
    assert by_name_value(duck, "user_id", "top_values")[:2] == by_name_value(pand, "user_id", "top_values")[:2]
    assert duck["age"]["stats"]["max"] == pytest.approx(pand_series(rich, "age").max())
    assert "stats" not in duck["city"]


def by_name_value(profiles, name, key):
    return profiles[name][key]


def pand_series(path, name):
    return pd.read_csv(path)[name]


def test_approximate_mode_flags_and_stays_close(rich, monkeypatch):
    monkeypatch.setattr(large_stats, "APPROX_FROM_ROWS", 0)
    monkeypatch.setattr(large_stats, "DISTINCT_EXACT_MAX", 10)
    duck = by_name(contract.column_profiles(ws_for(rich, "duckdb"), "train"))
    pand = by_name(contract.column_profiles(ws_for(rich, "pandas"), "train"))
    age, code, user = duck["age"], duck["code"], duck["user_id"]
    assert {"distinct", "stats.q1", "stats.median", "stats.q3", "iqr_bounds", "outliers", "skewed"} <= set(age["approximate"])
    assert "distinct" not in code["approximate"]  # few distinct values: counted exactly
    assert code["distinct"] == pand["code"]["distinct"]
    assert {"distinct", "top_values", "variants", "sentinel_candidates"} <= set(user["approximate"])
    assert user["distinct"] == pytest.approx(pand["user_id"]["distinct"], rel=0.05)
    assert age["stats"]["median"] == pytest.approx(pand_series(rich, "age").median(), rel=0.1)
    assert [t["count"] for t in user["top_values"]] == [1] * 20  # exact counts
    for name in ("count", "missing", "histogram"):
        assert age[name] == pand["age"][name]
    assert {k: v["kind"] for k, v in duck.items()} == {k: v["kind"] for k, v in pand.items()}


def rows_ws(path, engine, steps=()):
    return ws_for(path, engine, steps)


ROW_CASES = [
    {"offset": 0, "limit": 25},
    {"offset": 590, "limit": 25},
    {"offset": 123, "limit": 7, "columns": ["city", "age", "grp"]},
    {"offset": 3, "limit": 30, "sort": [{"column": "age", "desc": True}]},
    {"offset": 40, "limit": 30, "sort": [{"column": "city"}, {"column": "age", "desc": False}]},
    {"offset": 5, "limit": 30, "filter": {"conditions": [{"column": "city", "op": "isin", "value": ["Paris", "Lyon"]}]}},
    {
        "offset": 10, "limit": 30, "sort": [{"column": "grp", "desc": True}],
        "filter": {"conditions": [{"column": "age", "op": "ge", "value": 30}, {"column": "grp", "op": "isna"}], "combine": "or"},
    },
]  # fmt: skip


@pytest.mark.parametrize("case", ROW_CASES)
def test_rows_match_pandas_without_a_frame(rich, monkeypatch, case):
    pand = contract.workspace_rows(rows_ws(rich, "pandas"), "train", **case)

    def boom(*a, **k):
        raise AssertionError("pandas frame built")

    monkeypatch.setattr("dtk_engine.workspace.inspect._frame_at", boom)
    monkeypatch.setattr("dtk_engine.sources.csv_duckdb.read_csv_duckdb", boom)
    duck = contract.workspace_rows(rows_ws(rich, "duckdb"), "train", **case)
    monkeypatch.undo()
    assert duck["engine"] == "duckdb" and duck["rows"] == pand["rows"]
    assert (duck["total"], duck["total_unfiltered"], duck["version"]) == (pand["total"], pand["total_unfiltered"], pand["version"])
    meta = [{k: c[k] for k in ("name", "dtype", "kind", "semantic")} for c in duck["columns"]]
    # kinds come from the whole role, not from the rows a view filter keeps (pandas)
    whole = {k: v for k, v in case.items() if k != "filter"}
    assert meta == contract.workspace_rows(rows_ws(rich, "pandas"), "train", **whole)["columns"]
    assert all(c["approximate"] == ["kind", "semantic"] for c in duck["columns"])


def test_rows_after_steps_keep_raw_row_ids(csv):
    case = {"offset": 2, "limit": 50, "sort": [{"column": "x"}]}
    duck = contract.workspace_rows(rows_ws(csv, "duckdb", [FILTER, DROP]), "train", **case)
    pand = contract.workspace_rows(rows_ws(csv, "pandas", [FILTER, DROP]), "train", **case)
    assert duck["rows"] == pand["rows"] and duck["total"] == pand["total"] < 400
    assert "skew" not in duck["rows"][0]
    at_zero = contract.workspace_rows(rows_ws(csv, "duckdb", [FILTER, DROP]), "train", version=0, limit=1)
    assert at_zero["total"] == 400 and "skew" in at_zero["rows"][0]


def test_rows_view_filter_pandas_must_decide_falls_back(csv):
    flt = {"conditions": [{"column": "n", "op": "eq", "value": True}]}  # bool value, int column
    out = contract.workspace_rows(rows_ws(csv, "duckdb"), "train", filter=flt)
    assert "engine" not in out
    assert out == contract.workspace_rows(rows_ws(csv, "pandas"), "train", filter=flt)
    with pytest.raises(KeyParamsError, match="filter columns not in the frame"):
        contract.workspace_rows(rows_ws(csv, "duckdb"), "train", filter={"conditions": [{"column": "nope", "op": "isna"}]})
    with pytest.raises(KeyParamsError, match="sort columns not in the frame"):
        contract.workspace_rows(rows_ws(csv, "duckdb"), "train", sort=[{"column": "nope"}])


def test_preview_and_source_columns_match_pandas_without_a_frame(rich, monkeypatch):
    pand_preview = contract.preview_workspace(rows_ws(rich, "pandas"), "train", head_rows=4)
    pand_columns = contract.source_columns({"kind": "csv", "path": str(rich)})

    def boom(*a, **k):
        raise AssertionError("pandas frame built")

    monkeypatch.setattr("dtk_engine.workspace.dataset.workspace_frame", boom)
    monkeypatch.setattr("dtk_engine.sources.csv_duckdb.read_csv_duckdb", boom)
    assert contract.preview_workspace(rows_ws(rich, "duckdb"), "train", head_rows=4) == pand_preview
    assert contract.source_columns({"kind": "csv", "path": str(rich), "engine": "duckdb"}) == pand_columns


def test_stream_query_retries_with_fewer_threads(rich, monkeypatch):
    monkeypatch.setattr(large_stats, "APPROX_FROM_ROWS", 0)
    monkeypatch.setattr(large_stats, "DISTINCT_EXACT_MAX", 0)
    real, seen = large_stats._connect, []

    def connect(threads=None):
        seen.append(threads)
        if threads is not None:
            raise large.OutOfMemory("DuckDB ran out of its memory limit")
        return real(threads)

    monkeypatch.setattr(large_stats, "_connect", connect)
    ws = contract.parse_workspace(rows_ws(rich, "duckdb"))
    facts = large_stats.facts(large.plan(ws, "train", []), "user_id")
    assert facts["count"] == 600 and seen.count(None) >= 2 and any(seen)


def test_rows_over_http(rich, tmp_path):
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    body = {"workspace": rows_ws(rich, "duckdb"), "offset": 590, "limit": 20}
    got = TestClient(create_app()).post("/api/workspace/rows", json=body).json()
    assert got["engine"] == "duckdb" and got["total"] == 600 and len(got["rows"]) == 10
    assert got["rows"][0]["_rid"] == 590


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
