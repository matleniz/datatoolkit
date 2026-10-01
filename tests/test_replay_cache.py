"""Two-layer content-addressed cache: raw sources (A) and replay results (B)."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from dtk_engine import (
    align_report,
    column_profiles,
    contract,
    preview_step,
    workspace_rows,
)
from dtk_engine.cache import LRU
from dtk_engine.errors import SourceError
from dtk_engine.sources import registry
from dtk_engine.sources.registry import load
from dtk_engine.sources.spec import CsvSource, SqlSource
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import dataset


@pytest.fixture(autouse=True)
def _fresh_caches():
    registry._RAW_CACHE.clear()
    dataset._FRAMES.clear()
    contract._SHAPES.clear()
    yield
    registry._RAW_CACHE.clear()
    dataset._FRAMES.clear()
    contract._SHAPES.clear()


def _bump(path, seconds=5):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


def _ws(tmp_path, steps=None):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("id,x,y\n1,1.0,a\n2,2.0,b\n3,3.0,a\n")
    test.write_text("id,x,y\n4,4.0,a\n5,5.0,b\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(train)}},
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
        "steps": steps or [],
    }
    return ws, train, test


@pytest.fixture
def read_counter(monkeypatch):
    calls = []
    real = registry._READERS["csv"]

    def counting(spec):
        calls.append(spec.path)
        return real(spec)

    monkeypatch.setitem(registry._READERS, "csv", counting)
    return calls


# (a) rewritten file invalidates the raw cache
def test_rewritten_csv_invalidates_raw_cache(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a\n1\n2\n")
    spec = CsvSource(path=str(path))
    first = load(spec)
    path.write_text("a\n1\n2\n3\n")
    _bump(path)
    second = load(spec)
    assert len(first) == 2 and len(second) == 3


# (b) copy discipline
def test_load_returns_equal_but_distinct_frames(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a\n1\n2\n")
    spec = CsvSource(path=str(path))
    r1, r2 = load(spec), load(spec)
    assert r1 is not r2 and r1.equals(r2)
    r1.loc[0, "a"] = 99  # caller-side mutation must not reach the cache
    assert load(spec).equals(r2)


def test_replay_cache_is_mutation_safe(tmp_path):
    ws, *_ = _ws(tmp_path)  # no steps: replay hands back its input
    a = workspace_rows(ws, "train")
    from dtk_engine.workspace.inspect import _frame_at
    from dtk_engine.workspace.models import Workspace

    frame, _ = _frame_at(Workspace.model_validate(ws), "train", None)
    frame.loc[:, "x"] = -1
    assert workspace_rows(ws, "train") == a


def test_uncacheable_sources_are_never_cached(tmp_path, monkeypatch):
    monkeypatch.setitem(registry._READERS, "sql", lambda spec: pd.DataFrame({"a": [1]}))
    load(SqlSource(url_env="X", query="select 1"))
    assert len(registry._RAW_CACHE) == 0


def test_missing_file_still_raises(tmp_path):
    with pytest.raises((SourceError, OSError)):
        load(CsvSource(path=str(tmp_path / "missing.csv")))


# (c) unsaved-dict style: mutating steps between calls gives a fresh result
def test_changing_steps_is_not_stale(tmp_path):
    ws, *_ = _ws(tmp_path)
    before = workspace_rows(ws, "train")
    assert [c["name"] for c in before["columns"]] == ["id", "x", "y"]
    ws["steps"].append(
        {"op": "drop_columns", "target": "both", "params": {"columns": ["y"]}}
    )
    after = workspace_rows(ws, "train")
    assert [c["name"] for c in after["columns"]] == ["id", "x"]
    ws["steps"].clear()
    assert workspace_rows(ws, "train") == before


def test_file_edit_invalidates_replay_cache(tmp_path):
    ws, train, _ = _ws(tmp_path)
    assert workspace_rows(ws, "train")["total"] == 3
    train.write_text("id,x,y\n1,1.0,a\n")
    _bump(train)
    assert workspace_rows(ws, "train")["total"] == 1


# (d) call counts prove real hits
def test_second_identical_call_hits_both_layers(tmp_path, read_counter):
    ws, *_ = _ws(
        tmp_path,
        steps=[{"op": "drop_columns", "target": "both", "params": {"columns": ["y"]}}],
    )
    workspace_rows(ws, "train")
    first = len(read_counter)
    assert first >= 1
    workspace_rows(ws, "train")
    column_profiles(ws, "train")
    assert len(read_counter) == first  # no re-read from disk


def test_second_step_apply_count(tmp_path, monkeypatch):
    ws, *_ = _ws(
        tmp_path,
        steps=[{"op": "drop_columns", "target": "both", "params": {"columns": ["y"]}}],
    )
    t = get_transform("drop_columns")
    calls = []
    real = type(t).apply
    monkeypatch.setattr(
        type(t), "apply", lambda self, *a, **k: (calls.append(1), real(self, *a, **k))[1]
    )
    workspace_rows(ws, "train")
    n = len(calls)
    assert n >= 1
    workspace_rows(ws, "train")
    assert len(calls) == n


def test_preview_step_and_align_report_reuse(tmp_path, read_counter):
    steps = [
        {"op": "drop_columns", "target": "both", "params": {"columns": ["y"]}},
    ]
    ws, *_ = _ws(tmp_path, steps=steps)
    step = {"op": "drop_columns", "target": "both", "params": {"columns": ["x"]}}
    first_out = preview_step(ws, step, "train")
    align_first = align_report(ws)
    reads = len(read_counter)
    assert preview_step(ws, step, "train") == first_out
    assert align_report(ws) == align_first
    assert len(read_counter) == reads


def test_frame_lru_bounds():
    lru = LRU(max_entries=2, max_bytes=10**9)
    for k in "abc":
        lru.put(k, pd.DataFrame({"a": [1]}))
    assert len(lru) == 2 and lru.get("a") is None and lru.get("c") is not None
    tiny = LRU(max_entries=10, max_bytes=1)
    tiny.put("a", pd.DataFrame({"a": [1]}))
    assert len(tiny) == 0
    small = LRU(max_entries=10, max_bytes=300)
    for k in "abcd":
        small.put(k, pd.DataFrame({"a": range(10)}))
    assert small._bytes <= 300
