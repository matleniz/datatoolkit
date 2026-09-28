"""HTTP API (dtk_engine.http) over the JSON contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.http import create_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    return tmp_path


@pytest.fixture
def client(home):
    return TestClient(create_app())


def _workspace(name="w", **extra):
    ws = {
        "name": name,
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
    ws.update(extra)
    return ws


def test_list_keys(client):
    r = client.get("/api/keys")
    assert r.status_code == 200
    keys = r.json()
    assert keys and "id" in keys[0]
    assert json.dumps(keys)


def test_key_schema_and_run(client):
    r = client.get("/api/keys/dataset_overview/schema")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)
    r = client.post("/api/keys/dataset_overview/run", json={"params": {}})
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body and "tables" in body
    assert json.dumps(body)


def test_list_transforms_and_schema(client):
    r = client.get("/api/transforms")
    assert r.status_code == 200
    ops = r.json()
    assert ops and "op" in ops[0]
    r = client.get("/api/transforms/drop_columns/schema")
    assert r.status_code == 200
    assert "columns" in r.json()["properties"]


def test_workspace_crud(client):
    assert client.get("/api/workspaces").json() == []
    ws = _workspace()
    r = client.put("/api/workspaces/w", json=ws)
    assert r.status_code == 200
    saved = r.json()
    assert saved["name"] == "w"
    assert client.get("/api/workspaces/w").json() == saved
    assert [x["name"] for x in client.get("/api/workspaces").json()] == ["w"]
    r = client.delete("/api/workspaces/w")
    assert r.status_code == 204
    assert client.get("/api/workspaces").json() == []


def test_workspace_export(client, tmp_path):
    client.put("/api/workspaces/w", json=_workspace())
    out = tmp_path / "export"
    r = client.post(
        "/api/workspaces/w/export",
        json={"out_dir": str(out), "overwrite": False},
    )
    assert r.status_code == 200
    manifest = r.json()
    assert (out / "manifest.json").exists()
    assert (out / "processed" / "train.parquet").exists()
    assert json.dumps(manifest)


def test_source_columns(client):
    r = client.post(
        "/api/source/columns",
        json={"spec": {"kind": "csv", "path": TRAIN_CSV}},
    )
    assert r.status_code == 200
    cols = r.json()
    assert {"name", "dtype", "numeric"} <= set(cols[0])
    assert any(c["name"] == "Survived" for c in cols)


def test_workspace_preview_rows_profiles_align(client):
    ws = _workspace(
        steps=[
            {
                "op": "drop_columns",
                "target": "both",
                "params": {"columns": ["Name"]},
            }
        ]
    )
    r = client.post(
        "/api/workspace/preview",
        json={"workspace": ws, "role": "train", "head_rows": 3},
    )
    assert r.status_code == 200
    preview = r.json()
    assert preview["shape"][0] == 41 and len(preview["head"]) == 3
    assert "Name" not in preview["columns"]

    r = client.post(
        "/api/workspace/rows",
        json={"workspace": ws, "role": "train", "offset": 0, "limit": 5},
    )
    assert r.status_code == 200
    rows = r.json()
    assert rows["total"] == 41 and len(rows["rows"]) == 5
    assert rows["rows"][0]["_rid"] == 0

    r = client.post(
        "/api/workspace/profiles",
        json={"workspace": ws, "role": "train"},
    )
    assert r.status_code == 200
    profiles = r.json()
    assert profiles and json.dumps(profiles)

    r = client.post("/api/workspace/align", json={"workspace": ws})
    assert r.status_code == 200
    align = r.json()
    assert json.dumps(align)


def test_preview_step(client):
    ws = _workspace()
    step = {
        "op": "drop_columns",
        "target": "both",
        "params": {"columns": ["Name"]},
    }
    r = client.post(
        "/api/workspace/preview-step",
        json={"workspace": ws, "step": step, "role": "train"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["shape"][0] == 41
    assert "Name" in body["removed_columns"]
    assert "state" in body and "fitted_on" in body
    assert json.dumps(body)


def test_upload_then_source_columns(client, home):
    csv = b"a,b\n1,2\n3,4\n"
    r = client.put("/api/uploads/tiny.csv", content=csv)
    assert r.status_code == 200
    path = r.json()["path"]
    assert Path(path).is_file()
    assert Path(path).is_absolute()
    assert str(home / "uploads") in path
    r = client.post(
        "/api/source/columns",
        json={"spec": {"kind": "csv", "path": path}},
    )
    assert r.status_code == 200
    assert [c["name"] for c in r.json()] == ["a", "b"]


def test_cors_vite_origin(client):
    r = client.options(
        "/api/keys",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_unknown_key_404(client):
    r = client.get("/api/keys/does-not-exist/schema")
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "UnknownKeyError" and "message" in body


def test_unknown_transform_404(client):
    r = client.get("/api/transforms/does-not-exist/schema")
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "UnknownTransformError" and "message" in body


def test_workspace_not_found_404(client):
    r = client.get("/api/workspaces/missing")
    assert r.status_code == 404
    body = r.json()
    assert body["type"] == "WorkspaceNotFoundError" and "message" in body


def test_key_params_error_422(client):
    r = client.post(
        "/api/keys/dataset_overview/run",
        json={"params": {"__not_a_param__": True}},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "KeyParamsError" and "message" in body
    assert "details" in body
    assert "For further information visit" not in body["message"]


def test_source_error_422(client):
    r = client.post(
        "/api/source/columns",
        json={"spec": {"kind": "csv", "path": "/no/such/file.csv"}},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["type"] == "SourceError" and "message" in body


def test_unexpected_error_500_no_traceback(home, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("dtk_engine.http.contract.list_keys", boom)
    # raise_server_exceptions=False: otherwise TestClient re-raises after the
    # Exception handler has already turned it into a 500 JSON body.
    client = TestClient(create_app(), raise_server_exceptions=False)
    r = client.get("/api/keys")
    assert r.status_code == 500
    body = r.json()
    assert body == {"type": "RuntimeError", "message": "kaboom"}
    assert "traceback" not in body
    assert "Traceback" not in r.text
