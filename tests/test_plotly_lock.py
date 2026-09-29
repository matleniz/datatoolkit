"""Tests for thread-safe Plotly figure construction and serialization (MAT-243).

Plotly Express / go.Figure construction and template cascade are not thread-safe.
When concurrent requests run through thread pools (e.g. POST /keys/{id}/run),
unlocked figure construction intermittently raises ValueError: Invalid value
in plotly/basedatatypes.py (_index_is).
``plotly_lock`` serializes figure construction while keeping the underlying
data computations parallel.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from dtk_engine import contract
from dtk_engine.result import Figure, Result, plotly_lock

pytest.importorskip("plotly")


def test_plotly_lock_reentrant_and_decorator():
    """Verify plotly_lock supports re-entrancy and decorator syntax."""
    calls = []

    @plotly_lock
    def inner():
        calls.append("inner")
        return 42

    with plotly_lock:
        calls.append("outer")
        val = inner()
        assert val == 42

    assert calls == ["outer", "inner"]


def test_add_figure_accepts_callable_and_figure():
    """add_figure handles both go.Figure and zero-arg callables."""
    import plotly.graph_objects as go

    result = Result()
    result.add_figure("fig1", go.Figure())
    result.add_figure("fig2", lambda: go.Figure())
    result.add_figure("fig3", {"data": [], "layout": {}})

    assert len(result.figures) == 3
    assert all(isinstance(f, Figure) for f in result.figures)
    assert result.figures[0].title == "fig1"
    assert result.figures[1].title == "fig2"
    assert result.figures[2].title == "fig3"


def test_concurrent_run_key_stress():
    """Stress test: concurrent run_key across keys constructing Plotly figures."""
    keys_and_params = [
        ("column_distribution", {}),
        ("missing_values", {}),
        ("correlations", {}),
        ("chart", {"chart": "scatter", "x": "Age", "y": "Fare", "color": "Survived"}),
        ("chart", {"chart": "histogram", "x": "Age"}),
        ("chart", {"chart": "box", "x": "Pclass", "y": "Age"}),
        ("target_analysis", {"target": "Survived"}),
    ]

    n_threads = 8
    n_iterations = 24

    def worker(idx: int) -> dict:
        key_id, params = keys_and_params[idx % len(keys_and_params)]
        res = contract.run_key(key_id, params)
        assert "figures" in res
        assert isinstance(res["figures"], list)
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker, i) for i in range(n_iterations)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == n_iterations


def test_concurrent_run_key_barrier_stress():
    """Stress test with barrier to force threads into Plotly construction simultaneously."""
    n_threads = 8
    barrier = threading.Barrier(n_threads)

    def worker(idx: int) -> dict:
        barrier.wait()
        # column_distribution builds ~20 figures per call, high race likelihood if unlocked
        res = contract.run_key("column_distribution", {})
        assert len(res["figures"]) > 0
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker, i) for i in range(n_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == n_threads


def test_concurrent_http_run_key(tmp_path, monkeypatch):
    """Verify HTTP POST /keys/{id}/run handles concurrent requests cleanly."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dtk_engine.http import create_app

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))

    client = TestClient(create_app())

    requests = [
        ("column_distribution", {}),
        ("missing_values", {}),
        ("chart", {"chart": "histogram", "x": "Age"}),
        ("correlations", {}),
    ]

    n_threads = 4
    n_calls = 16

    def worker(idx: int):
        key_id, params = requests[idx % len(requests)]
        resp = client.post(f"/api/keys/{key_id}/run", json={"params": params})
        assert resp.status_code == 200
        body = resp.json()
        assert "figures" in body
        return resp.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(worker, i) for i in range(n_calls)]
        statuses = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(statuses) == n_calls
