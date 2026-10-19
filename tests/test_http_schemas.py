"""OpenAPI request schemas vs. what the handlers really accept (datatoolkit-issues#251).

The schemas describe today's behaviour, quirks included (``"5"`` for an int,
any ``overwrite``, a body ``name`` that the path overrides). Each case below
gives a body, whether the route's schema allows it, and the handler's status;
a schema that drifts from its handler fails here. A schema-forbidden body is
only asserted to fail (status >= 400), the exact error is not part of the promise.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
jsonschema = pytest.importorskip("jsonschema")

from fastapi.testclient import TestClient

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.http import create_app

TEST_CSV = str(Path(TRAIN_CSV).with_name("test.csv"))
SRC = {"kind": "csv", "path": TRAIN_CSV}
STEP = {"op": "drop_columns", "target": "both", "params": {"columns": ["Cabin"]}}


def _ws(name="w"):
    return {
        "name": name,
        "datasets": {
            "train": {"x": SRC, "target_column": "Survived"},
            "test": {"x": {"kind": "csv", "path": TEST_CSV}},
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    return TestClient(create_app(), raise_server_exceptions=False)  # a 500 is a status here


@pytest.fixture
def ctx(client, tmp_path):
    """A stored workspace ``w``, an upload and an output dir already exported to."""
    assert client.put("/api/workspaces/w", json=_ws()).status_code == 200
    up = client.put("/api/uploads/notes.txt", content=b"hello")
    assert up.status_code == 200
    out = tmp_path / "out"
    assert client.post(
        "/api/workspaces/w/export", json={"out_dir": str(out)}
    ).status_code == 200
    return SimpleNamespace(upload=up.json()["path"], out=str(out), fresh=str(tmp_path / "fresh"))


def _schema_allows(spec: dict, route: tuple[str, str], body) -> bool:
    method, path = route
    node = spec["paths"][path][method]["requestBody"]["content"]["application/json"]["schema"]
    root = {**node, "components": spec["components"]}
    return jsonschema.Draft202012Validator(root).is_valid(body)


def _ws_ref(**extra):
    return lambda c: {"workspace": _ws(), **extra}


# route -> [(label, body or callable(ctx) -> body, schema allows, handler status)].
# Status None: any failure (>= 400).
CASES = {
    ("post", "/api/keys/{key_id}/run"): [
        ("params", {"params": {}}, True, 200),
        ("params is a list", {"params": []}, False, None),
        ("params missing", {}, False, None),
        ("unknown field kept", {"params": {}, "extra": 1}, True, 200),
    ],
    ("put", "/api/workspaces/{name}"): [
        ("no name", lambda c: {k: v for k, v in _ws().items() if k != "name"}, True, 200),
        ("name ignored", lambda c: _ws("other"), True, 200),
        ("name that breaks the pattern is ignored too", lambda c: _ws("../bad name"), True, 200),
        ("name of another type is ignored too", lambda c: _ws(7), True, 200),
        ("datasets missing", {}, False, None),
        ("unknown field", lambda c: {**_ws(), "bogus": 1}, False, None),
    ],
    ("post", "/api/workspaces/{name}/rename"): [
        ("new_name", {"new_name": "renamed"}, True, 200),
        ("new_name missing", {}, False, None),
        ("new_name not a string", {"new_name": 3}, False, None),
    ],
    ("post", "/api/workspaces/{name}/duplicate"): [
        ("new_name", {"new_name": "copy"}, True, 200),
        ("new_name missing", {}, False, None),
        ("new_name not a string", {"new_name": 3}, False, None),
    ],
    ("post", "/api/workspaces/{name}/export"): [
        ("overwrite true", lambda c: {"out_dir": c.out, "overwrite": True}, True, 200),
        ("overwrite 'yes' is truthy", lambda c: {"out_dir": c.out, "overwrite": "yes"}, True, 200),
        ("overwrite 1 is truthy", lambda c: {"out_dir": c.out, "overwrite": 1}, True, 200),
        ("overwrite 0 is falsy", lambda c: {"out_dir": c.out, "overwrite": 0}, True, 422),
        ("overwrite '' is falsy", lambda c: {"out_dir": c.out, "overwrite": ""}, True, 422),
        ("overwrite null is falsy", lambda c: {"out_dir": c.out, "overwrite": None}, True, 422),
        ("fresh dir, defaults", lambda c: {"out_dir": c.fresh}, True, 200),
        ("formats", lambda c: {"out_dir": c.fresh, "formats": ["csv"]}, True, 200),
        ("formats null", lambda c: {"out_dir": c.fresh, "formats": None}, True, 200),
        ("out_dir missing", {}, False, None),
        ("out_dir not a string", {"out_dir": 3}, False, None),
        ("formats not a list", lambda c: {"out_dir": c.fresh, "formats": 7}, False, None),
    ],
    ("post", "/api/workspaces/{name}/documents"): [
        ("path", lambda c: {"path": c.upload}, True, 200),
        ("path, name, note", lambda c: {"path": c.upload, "name": "n.txt", "note": "x"}, True, 200),
        ("path missing", {}, False, None),
        ("path not a string", {"path": 3}, False, None),
    ],
    ("post", "/api/documents/describe"): [
        ("path", lambda c: {"path": c.upload}, True, 200),
        ("path and name", lambda c: {"path": c.upload, "name": "n.txt"}, True, 200),
        ("path missing", {}, False, None),
        ("path not a string", {"path": 3}, False, None),
    ],
    ("post", "/api/source/columns"): [
        ("csv spec", {"spec": SRC}, True, 200),
        ("spec missing", {}, False, None),
        ("spec not an object", {"spec": "x"}, False, None),
        ("unknown source kind", {"spec": {"kind": "nope"}}, False, None),
    ],
    ("post", "/api/workspace/preview"): [
        ("defaults", _ws_ref(), True, 200),
        ("role test", _ws_ref(role="test"), True, 200),
        ("head_rows int", _ws_ref(head_rows=2), True, 200),
        ("head_rows as a string", _ws_ref(head_rows="2"), True, 200),
        ("head_rows as a float", _ws_ref(head_rows=2.9), True, 200),
        ("head_rows as a bool", _ws_ref(head_rows=True), True, 200),
        # KNOWN DEFECT, pinned as is: the schema allows any string, int("abc") is a 500.
        # When the handler is fixed (422), update this case on purpose.
        ("head_rows non-numeric string", _ws_ref(head_rows="abc"), True, 500),
        ("head_rows null", _ws_ref(head_rows=None), False, None),
        ("head_rows array", _ws_ref(head_rows=[2]), False, None),
        ("role unknown", _ws_ref(role="val"), False, 422),
        ("role null", _ws_ref(role=None), False, 422),
        ("workspace missing", {}, False, None),
        ("workspace without datasets", {"workspace": {"name": "w"}}, False, 422),
        ("workspace with an unknown field", lambda c: {"workspace": {**_ws(), "bogus": 1}}, False, 422),
        ("workspace without a name", lambda c: {"workspace": {"datasets": _ws()["datasets"]}}, False, 422),
    ],
    ("post", "/api/workspace/rows"): [
        ("defaults", _ws_ref(), True, 200),
        ("offset and limit as strings", _ws_ref(offset="1", limit="3"), True, 200),
        ("offset and limit as floats", _ws_ref(offset=1.0, limit=3.0), True, 200),
        ("limit null", _ws_ref(limit=None), False, None),
        ("offset object", _ws_ref(offset={}), False, None),
        ("version int", _ws_ref(version=0), True, 200),
        ("version null", _ws_ref(version=None), True, 200),
        ("version as a string", _ws_ref(version="0"), False, 422),
        ("version negative", _ws_ref(version=-1), False, 422),
        ("version as a float", _ws_ref(version=0.5), False, 422),
        ("columns", _ws_ref(columns=["Age"]), True, 200),
        ("columns empty", _ws_ref(columns=[]), True, 200),
        ("columns null", _ws_ref(columns=None), True, 200),
        ("columns not a list", _ws_ref(columns="Age"), False, 422),
        ("columns with a non-string", _ws_ref(columns=[1]), False, 422),
        ("filter and sort", _ws_ref(
            filter={"conditions": [{"column": "Age", "op": "gt", "value": 30}], "combine": "and"},
            sort=[{"column": "Age", "desc": True}]), True, 200),
        ("role unknown", _ws_ref(role="val"), False, 422),
        ("workspace not an object", {"workspace": []}, False, 422),
    ],
    ("post", "/api/workspace/profiles"): [
        ("defaults", _ws_ref(), True, 200),
        ("role test, columns", _ws_ref(role="test", columns=["Age"]), True, 200),
        ("version as a string", _ws_ref(version="0"), False, 422),
        ("version negative", _ws_ref(version=-1), False, 422),
        ("columns not a list", _ws_ref(columns="Age"), False, 422),
        ("role unknown", _ws_ref(role="val"), False, 422),
        ("workspace missing", {}, False, None),
    ],
    ("post", "/api/workspace/preview-step"): [
        ("step", lambda c: {"workspace": _ws(), "step": STEP}, True, 200),
        ("step with an id and a note", lambda c: {
            "workspace": _ws(), "step": {**STEP, "id": "s9", "note": "n"}}, True, 200),
        ("step missing", _ws_ref(), False, None),
        ("step without target", lambda c: {"workspace": _ws(), "step": {"op": "drop_columns"}}, False, 422),
        ("step with an unknown field", lambda c: {"workspace": _ws(), "step": {**STEP, "x": 1}}, False, 422),
        ("step not an object", lambda c: {"workspace": _ws(), "step": []}, False, 422),
        ("role unknown", lambda c: {"workspace": _ws(), "step": STEP, "role": "val"}, False, 422),
    ],
    ("post", "/api/workspace/column-notes"): [
        ("defaults", _ws_ref(), True, 200),
        ("version int", _ws_ref(version=0), True, 200),
        ("version as a string", _ws_ref(version="0"), False, 422),
        ("version negative", _ws_ref(version=-1), False, 422),
        ("role unknown", _ws_ref(role="val"), False, 422),
    ],
    ("post", "/api/workspace/align"): [
        ("workspace", _ws_ref(), True, 200),
        ("workspace missing", {}, False, None),
        ("workspace with an unknown field", lambda c: {"workspace": {**_ws(), "bogus": 1}}, False, 422),
    ],
}

PATHS = {
    "/api/keys/{key_id}/run": "/api/keys/dataset_overview/run",
    "/api/workspaces/{name}": "/api/workspaces/w",
    "/api/workspaces/{name}/rename": "/api/workspaces/w/rename",
    "/api/workspaces/{name}/duplicate": "/api/workspaces/w/duplicate",
    "/api/workspaces/{name}/export": "/api/workspaces/w/export",
    "/api/workspaces/{name}/documents": "/api/workspaces/w/documents",
}


def test_every_request_body_route_has_cases(client):
    spec = client.get("/openapi.json").json()
    documented = {
        (method, path)
        for path, item in spec["paths"].items()
        if not path.startswith("/api/ui/")  # validated by FastAPI itself
        for method, op in item.items()
        if "requestBody" in op and "application/json" in op["requestBody"]["content"]
    }
    assert documented == set(CASES)  # a new route with a body needs cases here


@pytest.mark.parametrize(
    ("route", "label", "body", "schema_ok", "status"),
    [(route, *case) for route, cases in CASES.items() for case in cases],
    ids=[f"{route[0]} {route[1]}: {case[0]}" for route, cases in CASES.items() for case in cases],
)
def test_schema_matches_handler(client, ctx, route, label, body, schema_ok, status):
    spec = client.get("/openapi.json").json()
    payload = body(ctx) if callable(body) else body
    assert _schema_allows(spec, route, payload) is schema_ok, "schema disagrees with the table"
    method, path = route
    r = client.request(method.upper(), PATHS.get(path, path), json=payload)
    if status is None:
        assert r.status_code >= 400, r.text
    else:
        assert r.status_code == status, r.text
