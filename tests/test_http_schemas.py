"""OpenAPI request schemas vs. what the handlers really accept (datatoolkit-issues#251).

The schemas describe today's behaviour, quirks included (``"5"`` for an int,
any ``overwrite``, a body ``name`` that the path overrides). Each case below
gives a body, whether the route's schema allows it, and the handler's status;
a schema that drifts from its handler fails here. A schema-forbidden body is
only asserted to fail (status >= 400), the exact error is not part of the promise.
"""

from __future__ import annotations

import errno
import json
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
        ("formats not a list", lambda c: {"out_dir": c.fresh, "formats": 7}, False, 422),
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
        # The schema allows any string; int("abc") is rejected by the handler (422).
        ("head_rows non-numeric string", _ws_ref(head_rows="abc"), True, 422),
        ("head_rows null", _ws_ref(head_rows=None), False, 422),
        ("head_rows array", _ws_ref(head_rows=[2]), False, 422),
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
        ("offset non-numeric string", _ws_ref(offset="abc"), True, 422),
        ("offset null", _ws_ref(offset=None), False, 422),
        ("offset array", _ws_ref(offset=[]), False, 422),
        ("offset object", _ws_ref(offset={}), False, 422),
        ("limit non-numeric string", _ws_ref(limit="abc"), True, 422),
        ("limit null", _ws_ref(limit=None), False, 422),
        ("limit array", _ws_ref(limit=[]), False, 422),
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


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ("/api/workspace/preview", "head_rows"),
        ("/api/workspace/rows", "offset"),
        ("/api/workspace/rows", "limit"),
    ],
)
@pytest.mark.parametrize("value", ["abc", None, [], {}, "nan", "inf", "1e999", "x" * 500])
def test_integer_field_not_convertible_is_a_422_naming_the_field(client, path, field, value):
    r = client.post(path, json=_ws_ref(**{field: value})(None))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["type"] == "KeyParamsError"
    assert body["message"].startswith(f"{field}: must be an integer, got ")
    assert len(body["message"]) < 100


def test_integer_field_json_infinity_is_a_422(client):
    # httpx refuses to encode inf; the server's json parser accepts a bare 1e999 (= inf).
    ws = json.dumps(_ws())
    r = client.post(
        "/api/workspace/preview", content=f'{{"workspace": {ws}, "head_rows": 1e999}}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422, r.text
    assert r.json()["message"] == "head_rows: must be an integer, got inf"


# -- ui commands timeout (datatoolkit-issues#269) --


@pytest.fixture
def ui(client, monkeypatch):
    """The ui command route, with the bridge's ``send_command`` replaced by a recorder."""
    seen = []

    async def send_command(self, body, *, session, timeout):
        seen.append(timeout)
        return {"id": "c1", "ok": True}

    monkeypatch.setattr("dtk_engine.ui_bridge.UiBridge.send_command", send_command)
    token = client.app.state.ui_bridge.token
    local = TestClient(client.app, base_url="http://127.0.0.1:8765", raise_server_exceptions=False)

    def post(query="", body=None):
        return local.post(
            f"/api/ui/commands?{query}", json={"type": "x", **(body or {})},
            headers={"Authorization": f"Bearer {token}"},
        )

    post.seen = seen
    return post


@pytest.mark.parametrize("raw", ["nan", "-nan", "abc", "1,5"])
def test_ui_timeout_not_a_number_is_a_422_naming_the_field(ui, raw):
    r = ui(f"timeout={raw}")
    assert r.status_code == 422, r.text
    assert r.json()["type"] == "KeyParamsError"
    assert r.json()["message"] == f"timeout: must be a number, got {raw!r}"
    assert ui.seen == []


@pytest.mark.parametrize("value", [None, [], {}, "abc", "x" * 500])
def test_ui_timeout_bad_body_value_is_a_422(ui, value):
    r = ui(body={"timeout": value})
    assert r.status_code == 422, r.text
    assert r.json()["message"].startswith("timeout: must be a number, got ")
    assert len(r.json()["message"]) < 100


@pytest.mark.parametrize(
    ("query", "body", "expected"),
    [
        ("timeout=inf", None, 120.0),
        ("timeout=1e999", None, 120.0),
        ("timeout=-5", None, 0.0),
        ("timeout=-inf", None, 0.0),
        ("timeout=0", {"timeout": 7}, 0.0),  # the query string wins, "0" included
        ("", {"timeout": 0}, 0.0),
        ("timeout=2.5", None, 2.5),
        ("timeout=500", None, 120.0),
        ("", None, 30.0),
        ("", {"timeout": 9}, 9.0),
    ],
)
def test_ui_timeout_finite_values_keep_clamping(ui, query, body, expected):
    assert ui(query, body).status_code == 200
    assert ui.seen == [expected]


# -- export formats (datatoolkit-issues#270) --


@pytest.mark.parametrize(
    ("formats", "shown"),
    [(5, "5"), ("csv", "'csv'"), ({"a": 1}, "{'a': 1}"), ([1], "[1]"), (["csv", None], "['csv', None]")],
)
def test_export_formats_wrong_type_is_a_422_showing_the_value(client, ctx, formats, shown):
    r = client.post("/api/workspaces/w/export", json={"out_dir": ctx.fresh, "formats": formats})
    assert r.status_code == 422, r.text
    msg = r.json()["message"]
    assert f"got {shown}" in msg and "parquet" in msg
    assert not Path(ctx.fresh).exists()  # rejected before any staging dir is created


def test_export_formats_unknown_message_unchanged(client, ctx):
    r = client.post("/api/workspaces/w/export", json={"out_dir": ctx.fresh, "formats": ["csv", "x"]})
    assert r.status_code == 422
    assert r.json()["message"].endswith("got ['csv', 'x']")
    assert not Path(ctx.fresh).exists()


def test_export_formats_error_leaves_no_staging_dir_in_existing_out_dir(client, ctx):
    before = sorted(p.name for p in Path(ctx.out).iterdir())
    r = client.post(
        "/api/workspaces/w/export", json={"out_dir": ctx.out, "overwrite": True, "formats": 5}
    )
    assert r.status_code == 422
    assert sorted(p.name for p in Path(ctx.out).iterdir()) == before


def test_export_formats_duplicates_dropped_in_first_seen_order(client, ctx):
    r = client.post(
        "/api/workspaces/w/export", json={"out_dir": ctx.fresh, "formats": ["csv", "py", "csv", "py"]}
    )
    assert r.status_code == 200, r.text
    manifest = r.json()
    assert manifest["formats"] == ["csv", "py"]
    csv_outputs = [k for k in manifest["outputs"] if k.endswith("_csv")]
    assert len(csv_outputs) == len(set(csv_outputs)) == 2


# -- upload file names (datatoolkit-issues#271) --


@pytest.mark.parametrize(
    "name", ["a" * 300 + ".csv", "é" * 300 + ".csv", "日" * 200, "a" * 300, "a" * 10 + "." + "b" * 300],
)
def test_upload_long_name_is_truncated_to_a_stable_path(client, name):
    first = client.put(f"/api/uploads/{name}", content=b"abc")
    assert first.status_code == 200, first.text
    path = Path(first.json()["path"])
    assert len(path.name.encode()) + len(".." + "0" * 16 + ".tmp") <= 255  # temp name fits too
    assert path.read_bytes() == b"abc"
    again = client.put(f"/api/uploads/{name}", content=b"abc")
    assert again.json() == first.json()


def test_upload_long_name_keeps_the_extension(client):
    r = client.put("/api/uploads/" + "a" * 300 + ".csv", content=b"abc")
    assert Path(r.json()["path"]).suffix == ".csv"


def test_upload_normal_name_is_unchanged(client):
    r = client.put("/api/uploads/data.csv", content=b"abc")
    assert Path(r.json()["path"]).name == "data.csv"


def test_upload_dotdot_name_falls_back_to_upload(tmp_path):
    from dtk_engine import http

    for name in ("..", ""):
        sink = http._UploadSink(tmp_path)
        sink.write(b"x")
        assert sink.commit(name).name == "upload"


def _no_leak(r):
    assert r.status_code == 500, r.text
    assert r.json() == {"type": "UploadStoreError", "message": "could not store the upload"}
    for secret in ("/secret", "errno", "notes.txt", "No space"):
        assert secret not in r.text


@pytest.mark.parametrize(
    "error",
    [OSError(errno.ENOSPC, "No space left", "/secret/path/x"), PermissionError(13, "denied", "/secret/path/x")],
    ids=["ENOSPC", "PermissionError"],
)
def test_upload_write_failure_is_a_generic_500_without_path(client, monkeypatch, tmp_path, error):
    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr("dtk_engine.http.os.replace", boom)
    _no_leak(client.put("/api/uploads/notes.txt", content=b"abc"))


@pytest.mark.parametrize("stage", ["replace", "replace+unlink", "mkdir"])
def test_upload_failure_inside_the_write_leaves_no_temp_and_no_path(client, monkeypatch, tmp_path, stage):
    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "I/O error", "/secret/path/x")

    if stage == "mkdir":
        monkeypatch.setattr(Path, "mkdir", fail)
    else:
        monkeypatch.setattr("dtk_engine.http.os.replace", fail)
    if stage == "replace+unlink":  # the cleanup itself fails too: still the generic error
        monkeypatch.setattr(Path, "unlink", fail)
    _no_leak(client.put("/api/uploads/notes.txt", content=b"abc"))
    if stage == "replace":
        assert [p for p in (tmp_path / "uploads").rglob("*") if p.is_file()] == []


def test_upload_rewrite_of_a_corrupt_file_failing_is_a_generic_500(client, monkeypatch, tmp_path):
    first = client.put("/api/uploads/notes.txt", content=b"abc")
    Path(first.json()["path"]).write_bytes(b"xyz")  # same size, other bytes: will be rewritten

    def fail(*args, **kwargs):
        raise PermissionError(13, "denied", "/secret/path/x")

    monkeypatch.setattr("dtk_engine.http.os.replace", fail)
    _no_leak(client.put("/api/uploads/notes.txt", content=b"abc"))
