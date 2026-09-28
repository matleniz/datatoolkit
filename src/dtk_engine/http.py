"""HTTP API over the JSON contract (optional extra ``api``).

Thin FastAPI routes under ``/api`` that call ``dtk_engine.contract``; no
per-key logic. Install with ``pip install 'dtk-engine[api]'`` (or
``uv sync --extra api``) then run ``dtk-api``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePath

# FastAPI is an optional dependency; keep the import inside this module so
# ``import dtk_engine`` still works without the ``api`` extra.
from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from dtk_engine import contract
from dtk_engine.errors import (
    KeyParamsError,
    SourceError,
    UnknownKeyError,
    UnknownTransformError,
    message_from_validation_details,
    validation_error_details,
)
from dtk_engine.workspace import WorkspaceNotFoundError


def _cors_origins() -> list[str]:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra = os.environ.get("DTK_CORS_ORIGINS", "")
    for part in extra.split(","):
        origin = part.strip()
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def _upload_dir() -> Path:
    """``$DTK_UPLOAD_DIR``, else ``$DTK_HOME/uploads`` (``~/.datatoolkit/uploads``)."""
    if os.environ.get("DTK_UPLOAD_DIR"):
        return Path(os.environ["DTK_UPLOAD_DIR"]).expanduser()
    home = os.environ.get("DTK_HOME") or "~/.datatoolkit"
    return Path(home).expanduser() / "uploads"


def _save_upload(name: str, data: bytes, root: Path | None = None) -> Path:
    """Write an uploaded file under ``root`` and return its absolute path.

    Stored as ``<root>/<content hash>/<file name>``: same-name uploads with
    other contents never overwrite a file a saved workspace still reads.
    """
    digest = hashlib.sha256(data).hexdigest()[:12]
    base = PurePath(name.replace("\\", "/")).name
    if base in ("", ".", ".."):
        base = "upload"
    path = (root or _upload_dir()) / digest / base
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path.resolve()


def _error_body(exc: BaseException) -> dict:
    """JSON error payload: ``{type, message}`` plus ``details`` for validation.

    When the failure is a pydantic ``ValidationError`` (as ``exc.details`` or
    ``exc.__cause__``), ``message`` is the concise joined inner issues — never
    the full pydantic dump — and ``details`` is ``[{loc, msg, type}, ...]``.
    """
    details = getattr(exc, "details", None)
    cause = exc.__cause__
    if details is None and isinstance(cause, ValidationError):
        details = validation_error_details(cause)
    if details is not None:
        return {
            "type": type(exc).__name__,
            "message": message_from_validation_details(details),
            "details": details,
        }
    return {"type": type(exc).__name__, "message": str(exc)}


def create_app() -> FastAPI:
    """Build the FastAPI app (CORS + contract routes under ``/api``)."""
    application = FastAPI(title="dtk-api", docs_url=None, redoc_url=None)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(UnknownKeyError)
    @application.exception_handler(UnknownTransformError)
    @application.exception_handler(WorkspaceNotFoundError)
    async def _not_found(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content=_error_body(exc))

    @application.exception_handler(KeyParamsError)
    @application.exception_handler(SourceError)
    async def _unprocessable(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content=_error_body(exc))

    @application.exception_handler(Exception)
    async def _internal(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=_error_body(exc))

    api = APIRouter(prefix="/api")
    # Read-only compute endpoints run the (sync, CPU-bound) engine call in the
    # threadpool: on the event loop, one slow key (feature_selection: tens of
    # seconds) would stall every other request, previews included (MAT-212).
    # Store writes stay on the loop, so they remain serialized.

    @api.get("/keys")
    def get_keys() -> list[dict]:
        return contract.list_keys()

    @api.get("/keys/{key_id}/schema")
    def get_key_schema(key_id: str) -> dict:
        return contract.key_schema(key_id)

    @api.post("/keys/{key_id}/run")
    async def post_run_key(key_id: str, request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "params" not in body:
            raise KeyParamsError("body must be {params: ...}")
        return await run_in_threadpool(contract.run_key, key_id, body["params"])

    @api.get("/transforms")
    def get_transforms() -> list[dict]:
        return contract.list_transforms()

    @api.get("/transforms/{op}/schema")
    def get_transform_schema(op: str) -> dict:
        return contract.transform_schema(op)

    @api.get("/workspaces")
    def get_workspaces() -> list[dict]:
        return contract.list_workspaces()

    @api.get("/workspaces/summaries")
    def get_workspace_summaries() -> list[dict]:
        # Registered before /workspaces/{name} so "summaries" is not a name.
        return contract.list_workspace_summaries()

    @api.get("/workspaces/{name}")
    def get_one_workspace(name: str) -> dict:
        return contract.get_workspace(name)

    @api.put("/workspaces/{name}")
    async def put_workspace(name: str, request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict):
            raise KeyParamsError("workspace body must be a JSON object")
        return contract.save_workspace({**body, "name": name})

    @api.delete("/workspaces/{name}", status_code=204)
    def delete_one_workspace(name: str) -> Response:
        contract.delete_workspace(name)
        return Response(status_code=204)

    @api.post("/workspaces/{name}/rename")
    async def post_rename(name: str, request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "new_name" not in body:
            raise KeyParamsError("body must be {new_name: ...}")
        return contract.rename_workspace(name, body["new_name"])

    @api.post("/workspaces/{name}/duplicate")
    async def post_duplicate(name: str, request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "new_name" not in body:
            raise KeyParamsError("body must be {new_name: ...}")
        return contract.duplicate_workspace(name, body["new_name"])

    @api.post("/workspaces/{name}/export")
    async def post_export(name: str, request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "out_dir" not in body:
            raise KeyParamsError("body must include out_dir")
        return contract.export_workspace(
            name, body["out_dir"], overwrite=bool(body.get("overwrite", False))
        )

    @api.post("/source/columns")
    async def post_source_columns(request: Request) -> list[dict]:
        body = await request.json()
        if not isinstance(body, dict) or "spec" not in body:
            raise KeyParamsError("body must be {spec: ...}")
        return await run_in_threadpool(contract.source_columns, body["spec"])

    @api.post("/workspace/preview")
    async def post_preview(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "workspace" not in body:
            raise KeyParamsError("body must include workspace")
        return await run_in_threadpool(
            contract.preview_workspace,
            body["workspace"],
            body.get("role", "train"),
            head_rows=int(body.get("head_rows", 5)),
        )

    @api.post("/workspace/rows")
    async def post_rows(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "workspace" not in body:
            raise KeyParamsError("body must include workspace")
        return await run_in_threadpool(
            contract.workspace_rows,
            body["workspace"],
            body.get("role", "train"),
            version=body.get("version"),
            offset=int(body.get("offset", 0)),
            limit=int(body.get("limit", 500)),
            columns=body.get("columns"),
        )

    @api.post("/workspace/profiles")
    async def post_profiles(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "workspace" not in body:
            raise KeyParamsError("body must include workspace")
        return await run_in_threadpool(
            contract.column_profiles,
            body["workspace"],
            body.get("role", "train"),
            version=body.get("version"),
            columns=body.get("columns"),
        )

    @api.post("/workspace/preview-step")
    async def post_preview_step(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "workspace" not in body or "step" not in body:
            raise KeyParamsError("body must include workspace and step")
        return await run_in_threadpool(
            contract.preview_step,
            body["workspace"], body["step"], body.get("role", "train")
        )

    @api.post("/workspace/align")
    async def post_align(request: Request) -> dict:
        body = await request.json()
        if not isinstance(body, dict) or "workspace" not in body:
            raise KeyParamsError("body must include workspace")
        return await run_in_threadpool(contract.align_report, body["workspace"])

    @api.put("/uploads/{filename}")
    async def put_upload(filename: str, request: Request) -> dict:
        data = await request.body()
        path = _save_upload(filename, data)
        return {"path": str(path)}

    application.include_router(api)
    return application


app = create_app()


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``dtk-api [--host HOST] [--port PORT]``."""
    parser = argparse.ArgumentParser(prog="dtk-api", description="datatoolkit HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
