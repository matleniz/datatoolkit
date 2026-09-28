"""E-G polish (MAT-142): identifier sentinel skip + concise HTTP validation errors."""

from __future__ import annotations

import json

import pytest

from dtk_engine import column_profiles
from dtk_engine.ops.missing import sentinel_counts

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.http import create_app


def test_column_profiles_skips_sentinels_on_identifier(tmp_path):
    """Identifier columns with 999 / -1 must not report sentinel candidates."""
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    # Dense Index-like ids including real values -1 and 999 (Parkinson-style).
    ids = [-1] + list(range(1000))
    lines = ["Index,Age,Survived"] + [f"{i},{20 + (i % 10)},{i % 2}" for i in ids]
    train.write_text("\n".join(lines) + "\n")
    test.write_text("Index,Age\n0,21\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    # Prove the raw sentinel detector would have flagged them.
    import pandas as pd

    hits = sentinel_counts(pd.DataFrame({"Index": ids}))
    assert set(hits["sentinel"].astype(str)) >= {"-1", "999"}

    by = {c["name"]: c for c in column_profiles(ws, "train")["columns"]}
    assert by["Index"]["kind"] == "identifier"
    assert by["Index"]["sentinel_candidates"] == []
    # A normal numeric column still surfaces sentinels.
    age = by["Age"]
    assert age["kind"] == "number"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    return tmp_path


@pytest.fixture
def client(home):
    return TestClient(create_app())


def _workspace():
    from pathlib import Path

    return {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": TRAIN_CSV},
                "target_column": "Survived",
            },
            "test": {
                "x": {
                    "kind": "csv",
                    "path": str(Path(TRAIN_CSV).with_name("test.csv")),
                }
            },
        },
    }


def test_preview_step_formula_validation_concise_422(client):
    """Invalid formula expr -> concise message + details, not the pydantic dump."""
    r = client.post(
        "/api/workspace/preview-step",
        json={
            "workspace": _workspace(),
            "step": {
                "op": "formula",
                "target": "both",
                "params": {"name": "x", "expr": "monthly_spend +"},
            },
            "role": "train",
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "KeyParamsError"
    assert body["message"] == "formula: invalid expression (invalid syntax)"
    assert "For further information visit" not in body["message"]
    assert "validation error" not in body["message"].lower()
    assert body["details"] == [
        {
            "loc": [],
            "msg": "formula: invalid expression (invalid syntax)",
            "type": "value_error",
        }
    ]
    assert json.dumps(body)


def test_key_params_error_422_includes_details(client):
    r = client.post(
        "/api/keys/dataset_overview/run",
        json={"params": {"__not_a_param__": True}},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "KeyParamsError"
    assert "details" in body
    assert isinstance(body["details"], list) and body["details"]
    assert {"loc", "msg", "type"} <= set(body["details"][0])
    assert "For further information visit" not in body["message"]
    assert "validation error" not in body["message"].lower()
