"""``column_profiles(sample_rows=...)``: seeded sample, disclosure fields, validation, cache."""

from __future__ import annotations

import pytest

from dtk_engine import column_profiles
from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace.inspect import _profile_one

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from dtk_engine.http import create_app


@pytest.fixture
def ws(tmp_path):
    train = tmp_path / "t.csv"
    train.write_text("a,b,y\n" + "".join(f"{i},{i % 7},{i % 2}\n" for i in range(200)))
    return {
        "name": "w",
        "datasets": {"train": {"x": {"kind": "csv", "path": str(train)}, "target_column": "y"}},
    }


def test_default_is_full_frame_and_unchanged(ws):
    out = column_profiles(ws, "train")
    assert out == column_profiles(ws, "train", sample_rows=None)
    assert set(out) == {"columns", "version"}
    assert {c["name"] for c in out["columns"]} == {"a", "b", "y"}


def test_sampled_response_fields(ws):
    out = column_profiles(ws, "train", sample_rows=50)
    assert (out["sampled"], out["n_rows_source"], out["n_sampled_out"]) == (True, 200, 150)
    assert "estimates" in out["note"] and "50 rows" in out["note"]
    assert all(c["count"] == 50 for c in out["columns"])
    assert out["version"] == 0


def test_sample_is_deterministic(ws):
    assert column_profiles(ws, "train", sample_rows=50) == column_profiles(
        ws, "train", sample_rows=50
    )


@pytest.mark.parametrize("cap", [200, 500])
def test_not_sampled_when_frame_fits(ws, cap):
    out = column_profiles(ws, "train", sample_rows=cap)
    full = column_profiles(ws, "train")
    assert out["sampled"] is False and out["n_sampled_out"] == 0 and "note" not in out
    assert out["columns"] == full["columns"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, 5.0, True, "10", []])
def test_invalid_sample_rows(ws, bad):
    with pytest.raises(KeyParamsError, match="sample_rows"):
        column_profiles(ws, "train", sample_rows=bad)


def test_cache_key_separates_sample_sizes(ws):
    cache = _profile_one.cache
    cache._data.clear()
    column_profiles(ws, "train", columns=["a"])
    assert len(cache._data) == 1
    column_profiles(ws, "train", columns=["a"], sample_rows=20)
    assert len(cache._data) == 2  # sampled series -> its own key
    column_profiles(ws, "train", columns=["a"], sample_rows=20)
    column_profiles(ws, "train", columns=["a"])
    assert len(cache._data) == 2  # both are cache hits


def test_http_sample_and_validation(ws):
    client = TestClient(create_app())
    r = client.post("/api/workspace/profiles", json={"workspace": ws, "role": "train"})
    assert r.status_code == 200 and set(r.json()) == {"columns", "version"}
    r = client.post(
        "/api/workspace/profiles", json={"workspace": ws, "role": "train", "sample_rows": 30}
    )
    assert r.status_code == 200
    assert (r.json()["sampled"], r.json()["n_sampled_out"]) == (True, 170)
    assert "estimates" in r.json()["note"]
    for bad in (0, -3, 1.5, True, "5"):
        r = client.post(
            "/api/workspace/profiles",
            json={"workspace": ws, "role": "train", "sample_rows": bad},
        )
        assert r.status_code == 422 and "sample_rows" in r.text
