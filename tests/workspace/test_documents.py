"""Workspace documents: model, contract, HTTP routes, export (datatoolkit-issues#178)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dtk_engine import contract
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.http import create_app
from dtk_engine.workspace import documents
from dtk_engine.workspace.dataset import workspace_key
from dtk_engine.workspace.models import DOCUMENTS_MAX, Workspace


def make_pdf(*pages: str) -> bytes:
    """A minimal valid PDF, one Helvetica text line per page."""
    n = len(pages)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font = 3 + 2 * n
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "up"))
    (tmp_path / "up").mkdir()
    return tmp_path


@pytest.fixture
def upload(home):
    def put(name: str, data: bytes | str) -> str:
        folder = home / "up" / f"h{abs(hash(name)) % 10**6}"
        folder.mkdir(exist_ok=True)
        path = folder / name
        path.write_bytes(data.encode() if isinstance(data, str) else data)
        return str(path)

    return put


def _ws(**extra) -> dict:
    return {
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        **extra,
    }


# -- model ----------------------------------------------------------------------


def test_ids_filled_above_existing_and_duplicates_refused():
    docs = [{"name": "a", "path": "/x/a"}, {"id": "d4", "name": "b", "path": "/x/b"},
            {"name": "c", "path": "/x/c"}]
    ws = Workspace.model_validate(_ws(documents=docs))
    assert [d.id for d in ws.documents] == ["d5", "d4", "d6"]
    with pytest.raises(ValueError, match="duplicate document id"):
        Workspace.model_validate(_ws(documents=[{"id": "d1", "name": "a", "path": "/a"}] * 2))
    with pytest.raises(ValueError, match="at most"):
        Workspace.model_validate(
            _ws(documents=[{"name": "a", "path": "/a"}] * (DOCUMENTS_MAX + 1))
        )


def test_documents_are_not_data():
    a = Workspace.model_validate(_ws())
    b = Workspace.model_validate(_ws(documents=[{"name": "a", "path": "/a"}]))
    assert workspace_key(a, "x", "train", a.steps) == workspace_key(b, "x", "train", b.steps)


# -- contract -------------------------------------------------------------------


def test_add_read_remove_round_trip(upload):
    contract.save_workspace(_ws())
    md = upload("dictionary.md", "# Dictionary\nledd: mg/day, 0 = untreated\n")
    out = contract.add_document("demo", md, note="from the study team")
    doc = out["document"]
    assert doc["id"] == "d1" and doc["name"] == "dictionary.md" and doc["kind"] == "text"
    assert doc["mime"] == "text/markdown" and doc["note"] == "from the study team"
    assert doc["size"] > 0 and doc["added_at"].endswith("Z")
    assert out["workspace"]["documents"] == [doc] == contract.list_documents("demo")
    text = contract.document_text("demo", "d1", offset=2, max_chars=10)
    assert text["text"] == "Dictionary" and text["next_offset"] == 12
    assert contract.document_text("demo", "d1")["next_offset"] is None
    assert contract.remove_document("demo", "d1")["workspace"]["documents"] == []
    with pytest.raises(KeyParamsError, match="no document"):
        contract.remove_document("demo", "d1")
    assert Path(md).is_file()  # the upload stays


def test_kinds_and_unreadable(upload):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("visits.csv", "id,age\n1,2\n"))
    contract.add_document("demo", upload("blob.bin", b"\xff\xfe\x00\x80"))
    kinds = [d["kind"] for d in contract.list_documents("demo")]
    assert kinds == ["table", "other"]
    with pytest.raises(documents.DocumentNotReadable, match="table"):
        contract.document_text("demo", "d1")
    with pytest.raises(documents.DocumentNotReadable, match="other"):
        contract.document_text("demo", "d2")
    summary = contract.document_summaries("demo")[0]
    assert summary["source"]["kind"] == "csv" and "path" not in summary


def test_pdf_text_with_page_markers(upload):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("protocol.pdf", make_pdf("Visit one", "Visit two")))
    doc = contract.list_documents("demo")[0]
    assert doc["kind"] == "pdf" and doc["mime"] == "application/pdf"
    text = contract.document_text("demo", "d1")["text"]
    assert "--- page 1 ---" in text and "Visit one" in text
    assert text.index("--- page 2 ---") > text.index("Visit one") and "Visit two" in text


def test_pdf_without_the_extra_is_listed_not_read(upload, monkeypatch):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("protocol.pdf", make_pdf("x")))
    monkeypatch.setattr(documents, "pdf_available", lambda: False)
    with pytest.raises(documents.DocumentNotReadable, match="pdf"):
        contract.document_text("demo", "d1")


def test_broken_pdf_is_not_readable(upload):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("broken.pdf", b"not a pdf"))
    with pytest.raises(documents.DocumentNotReadable, match="PDF"):
        contract.document_text("demo", "d1")


def test_paths_outside_the_upload_dir_are_refused(home, upload):
    outside = home / "secret.txt"
    outside.write_text("secret")
    contract.save_workspace(_ws())
    with pytest.raises(documents.NotAnUploadError):
        contract.add_document("demo", str(outside))
    with pytest.raises(KeyParamsError, match="not an uploaded file"):
        contract.save_workspace(_ws(documents=[{"name": "s", "path": str(outside), "kind": "text"}]))
    # A stored path is read only while it resolves under the upload dir.
    md = upload("a.md", "hello")
    contract.add_document("demo", md)
    Path(md).unlink()
    Path(md).symlink_to(outside)
    with pytest.raises(documents.NotAnUploadError):
        contract.document_text("demo", "d1")


def test_known_paths_still_save_when_the_upload_dir_moves(home, upload, monkeypatch):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("a.md", "hello"))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(home / "elsewhere"))
    ws = contract.get_workspace("demo")
    assert contract.save_workspace({**ws, "steps": []})["documents"] == ws["documents"]


def test_duplicate_and_rename_keep_documents(upload):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("a.md", "hello"))
    copy = contract.duplicate_workspace("demo", "copy")
    assert copy["documents"] == contract.list_documents("demo")
    contract.delete_workspace("copy")
    assert contract.document_text("demo", "d1")["text"] == "hello"


def test_export_lists_documents(home, upload):
    contract.save_workspace(_ws())
    contract.add_document("demo", upload("a.md", "hello"), note="n")
    contract.add_document("demo", upload("gone.md", "x"))
    Path(contract.list_documents("demo")[1]["path"]).unlink()
    manifest = contract.export_workspace("demo", str(home / "out"))
    first, gone = manifest["documents"]
    assert first["id"] == "d1" and first["note"] == "n" and len(first["sha256"]) == 64
    assert gone["missing"] is True
    assert not (home / "out" / "documents").exists()  # listed, not copied


# -- HTTP -----------------------------------------------------------------------


def test_http_routes(home):
    client = TestClient(create_app())
    assert client.put("/api/workspaces/demo", json=_ws()).status_code == 200
    path = client.put("/api/uploads/dictionary.md", content=b"# Dict\nage: years").json()["path"]
    described = client.post("/api/documents/describe", json={"path": path, "name": "Dict"}).json()
    assert described["name"] == "Dict" and described["kind"] == "text" and "id" not in described
    added = client.post("/api/workspaces/demo/documents", json={"path": path})
    assert added.status_code == 200 and added.json()["document"]["id"] == "d1"
    assert [d["id"] for d in client.get("/api/workspaces/demo/documents").json()] == ["d1"]
    text = client.get("/api/workspaces/demo/documents/d1/text", params={"max_chars": 6}).json()
    assert text["text"] == "# Dict" and text["next_offset"] == 6
    raw = client.get("/api/workspaces/demo/documents/d1/file")
    assert raw.status_code == 200 and raw.content == b"# Dict\nage: years"
    assert raw.headers["content-type"].startswith("text/markdown")
    assert raw.headers["content-disposition"].startswith("inline")
    removed = client.delete("/api/workspaces/demo/documents/d1")
    assert removed.status_code == 200 and removed.json()["workspace"]["documents"] == []


def test_http_errors(home):
    client = TestClient(create_app())
    client.put("/api/workspaces/demo", json=_ws())
    outside = home / "secret.txt"
    outside.write_text("s")
    r = client.post("/api/workspaces/demo/documents", json={"path": str(outside)})
    assert r.status_code == 422 and r.json()["type"] == "NotAnUploadError"
    assert client.post("/api/workspaces/nope/documents", json={"path": "x"}).status_code == 422
    path = client.put("/api/uploads/t.csv", content=b"a,b\n1,2\n").json()["path"]
    client.post("/api/workspaces/demo/documents", json={"path": path})
    r = client.get("/api/workspaces/demo/documents/d1/text")
    assert r.status_code == 422 and r.json()["type"] == "DocumentNotReadable"
    assert client.get("/api/workspaces/demo/documents/d9/file").status_code == 422
    assert client.get("/api/workspaces/nope/documents").status_code == 404
    assert json.dumps(client.get("/api/workspaces/demo").json()["documents"])
