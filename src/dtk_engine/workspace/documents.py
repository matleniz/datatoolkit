"""Workspace documents: reference files kept with a workspace (datatoolkit-issues#178).

A document is an uploaded file (``PUT /api/uploads/{filename}``, stored
content-addressed under the upload dir) listed in ``Workspace.documents``; the
file is never copied, so duplicating or deleting a workspace leaves uploads
alone. Reads always re-resolve the path under the upload dir (symlinks
followed): a workspace file naming another path reads nothing. PDF text needs
the optional ``pdf`` extra (pypdf); without it a PDF is listed, not read.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from dtk_engine.errors import KeyParamsError

TEXT_LIMIT = 1_000_000  # bytes: larger (or non UTF-8) files are kind "other"
PDF_MAX_BYTES = 50_000_000
PDF_MAX_PAGES = 500
TABLE_EXTS = {".csv", ".tsv", ".parquet", ".xlsx", ".json", ".jsonl"}
_MIMES = {".md": "text/markdown", ".jsonl": "application/x-ndjson", ".parquet": "application/vnd.apache.parquet"}


class NotAnUploadError(KeyParamsError):
    """The path is not a file under the upload dir."""


class DocumentNotReadable(KeyParamsError):
    """The document has no text to give (table / other, or PDF without pypdf)."""


def upload_dir() -> Path:
    """``$DTK_UPLOAD_DIR``, else ``$DTK_HOME/uploads`` (``~/.datatoolkit/uploads``)."""
    if os.environ.get("DTK_UPLOAD_DIR"):
        return Path(os.environ["DTK_UPLOAD_DIR"]).expanduser()
    home = os.environ.get("DTK_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".datatoolkit") / "uploads"


def resolve_upload(path: str) -> Path:
    """Realpath of ``path`` when it is a file under the upload dir (symlinks followed)."""
    real = Path(os.path.realpath(Path(path).expanduser()))
    root = Path(os.path.realpath(upload_dir()))
    if not real.is_relative_to(root) or not real.is_file():
        raise NotAnUploadError(f"not an uploaded file: {path!r} (upload it first)")
    return real


def is_text(path: Path, limit: int = TEXT_LIMIT) -> bool:
    try:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            return False
        data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return True


def detect_kind(path: Path, limit: int = TEXT_LIMIT, *, pdf: bool = True) -> str:
    """``table`` / ``pdf`` (by extension), ``text`` (UTF-8 sniff up to ``limit``), else ``other``."""
    ext = path.suffix.lower()
    if ext in TABLE_EXTS:
        return "table"
    if pdf and ext == ".pdf":
        return "pdf"
    return "text" if is_text(path, limit) else "other"


def source_spec(path: Path) -> dict:
    """The ``source`` spec the usual tools read a table file with."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return {"kind": "parquet", "path": str(path)}
    if ext == ".xlsx":
        return {"kind": "excel", "path": str(path)}
    if ext in (".json", ".jsonl"):
        return {"kind": "json", "path": str(path), "lines": ext == ".jsonl"}
    return {"kind": "csv", "path": str(path), "sep": "\t" if ext == ".tsv" else "auto"}


def guess_mime(name: str) -> str:
    ext = Path(name).suffix.lower()
    return _MIMES.get(ext) or mimetypes.guess_type(name)[0] or "application/octet-stream"


def describe(path: str, name: str | None = None) -> dict:
    """A document entry (no ``id``) for an uploaded file: name, path, mime, size,
    kind, added_at (now). File I/O: run it off the event loop."""
    real = resolve_upload(path)
    label = (name or "").strip() or real.name
    return {
        "name": label[:255],
        "path": str(real),
        "mime": guess_mime(real.name),
        "size": real.stat().st_size,
        "kind": detect_kind(real),
        "added_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def pdf_available() -> bool:
    try:
        import pypdf  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=8)
def _pdf_text(path: str, mtime_ns: int, size: int) -> str:
    """Page texts joined with page markers; cached per file version (uploads
    are content-addressed, so a path rarely changes)."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    if size > PDF_MAX_BYTES:
        raise DocumentNotReadable(f"PDF over {PDF_MAX_BYTES // 1_000_000} MB: not extracted")
    try:
        reader = PdfReader(path)
        pages = [
            f"--- page {i} ---\n\n{(page.extract_text() or '').strip()}"
            for i, page in enumerate(reader.pages[:PDF_MAX_PAGES], start=1)
        ]
    except (PdfReadError, ValueError, OSError) as exc:
        raise DocumentNotReadable(f"PDF text extraction failed: {exc}") from exc
    if len(reader.pages) > PDF_MAX_PAGES:
        pages.append(f"--- {len(reader.pages) - PDF_MAX_PAGES} more pages not extracted ---")
    return "\n\n".join(pages)


def _bounded_text(real: Path) -> str:
    with real.open("rb") as handle:
        data = handle.read(TEXT_LIMIT + 1)
    if len(data) > TEXT_LIMIT:
        raise DocumentNotReadable("the file grew over the text limit since it was added")
    return data.decode("utf-8", errors="replace")


def full_text(doc: dict) -> str:
    """The whole text of a ``text`` or ``pdf`` document, else DocumentNotReadable."""
    kind = doc.get("kind")
    real = resolve_upload(doc["path"])  # re-checked at every read
    if kind == "text":
        return _bounded_text(real)
    if kind == "pdf":
        if not pdf_available():
            raise DocumentNotReadable(
                "PDF text needs the optional `pdf` extra (pypdf): pip install 'dtk-engine[pdf]'"
            )
        stat = real.stat()
        return _pdf_text(str(real), stat.st_mtime_ns, stat.st_size)
    hint = "read it with the usual tools on its source spec" if kind == "table" else "name only"
    raise DocumentNotReadable(f"document {doc.get('id')!r} is {kind}: no text ({hint})")


def text_slice(doc: dict, offset: int = 0, max_chars: int | None = None) -> dict:
    """``{id, name, kind, offset, text, total_chars, next_offset}``."""
    text = full_text(doc)
    start = max(int(offset or 0), 0)
    chunk = text[start:] if max_chars is None else text[start : start + max(int(max_chars), 1)]
    end = start + len(chunk)
    return {
        "id": doc.get("id"), "name": doc.get("name"), "kind": doc.get("kind"),
        "offset": start, "text": chunk, "total_chars": len(text),
        "next_offset": end if end < len(text) else None,
    }


def find(documents: list[dict], doc_id: str) -> dict:
    found = next((d for d in documents if d.get("id") == doc_id), None)
    if found is None:
        raise KeyParamsError(f"no document {doc_id!r} in this workspace")
    return found


def summary(doc: dict) -> dict[str, Any]:
    """What the agent lists: no path (read through read_document), table columns."""
    out = {k: doc.get(k) for k in ("id", "name", "kind", "mime", "size", "note")}
    if doc.get("kind") == "table":
        out["source"] = source_spec(Path(doc["path"]))
    return out
