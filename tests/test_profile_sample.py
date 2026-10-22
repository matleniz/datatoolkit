"""``column_profiles(sample_rows=...)``: seeded sample, disclosure fields, validation, cache."""

from __future__ import annotations

import pytest

from dtk_engine import column_profiles
from dtk_engine.errors import KeyParamsError

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


def test_default_is_full_frame(ws):
    out = column_profiles(ws, "train")
    assert out == column_profiles(ws, "train", sample_rows=None)
    assert out["n_rows"] == out["n_rows_used"] == out["n_rows_source"] == 200
    assert out["sampled"] is False and out["n_sampled_out"] == 0
    assert {c["name"] for c in out["columns"]} == {"a", "b", "y"}


def test_sampled_response_fields(ws):
    out = column_profiles(ws, "train", sample_rows=50)
    assert (out["n_rows"], out["n_rows_used"], out["sampled"]) == (200, 50, True)
    assert (out["n_rows_source"], out["n_sampled_out"]) == (200, 150)
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
    assert out == full and out["sampled"] is False


@pytest.mark.parametrize("bad", [0, -1, 2.5, 5.0, True, "10", []])
def test_invalid_sample_rows(ws, bad):
    with pytest.raises(KeyParamsError, match="sample_rows"):
        column_profiles(ws, "train", sample_rows=bad)


def test_cache_key_separates_sample_sizes(ws):
    full = column_profiles(ws, "train")
    small = column_profiles(ws, "train", sample_rows=20)
    assert column_profiles(ws, "train") == full
    assert small["columns"][0]["count"] == 20 and full["columns"][0]["count"] == 200


def test_http_sample_and_validation(ws):
    client = TestClient(create_app())
    r = client.post("/api/workspace/profiles", json={"workspace": ws, "role": "train"})
    assert r.status_code == 200 and r.json()["sampled"] is False
    r = client.post(
        "/api/workspace/profiles", json={"workspace": ws, "role": "train", "sample_rows": 30}
    )
    assert r.status_code == 200
    assert (r.json()["n_rows"], r.json()["n_rows_used"], r.json()["sampled"]) == (200, 30, True)
    for bad in (0, -3, 1.5, True, "5"):
        r = client.post(
            "/api/workspace/profiles",
            json={"workspace": ws, "role": "train", "sample_rows": bad},
        )
        assert r.status_code == 422 and "sample_rows" in r.text
