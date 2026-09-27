"""Workspace export: processed parquet + a provenance manifest.

``export_workspace(name, out_dir)`` writes, under ``out_dir``::

    processed/train.parquet     train after every train / both step (labels joined)
    processed/test.parquet      test after every test / both step (if a test set)
    states/step_<i>_<op>.json   fitted states too big to inline (e.g. impute_knn)
    manifest.json               written last: its presence marks a complete export

The manifest records the sources (path, size, sha256, mtime), the export time,
every step with its fitted state (inline, or a side file with its sha256) and
the engine / pandas / sklearn / pyarrow / python versions. Raw inputs are only
read: an output path that is a source path is refused. Overwrite only deletes
relative paths under processed/ or states/ listed by a manifest of ours.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.sources.csv_pandas import resolve_path
from dtk_engine.sources.dataset import raw_workspace_frame
from dtk_engine.workspace.models import Workspace
from dtk_engine.workspace.replay import replay_fitted
from dtk_engine.workspace.store import JsonWorkspaceStore

MANIFEST_VERSION = 1
GENERATOR = "dtk_engine"
MANIFEST = "manifest.json"
PROCESSED_DIR = "processed"
STATES_DIR = "states"
# A fitted state whose JSON exceeds this goes to a side file (impute_knn and
# impute_iterative keep the whole train matrix in their state).
INLINE_STATE_BYTES = 64 * 1024
_CHUNK = 1 << 20


def _sha256_file(path: Path, digest=None) -> str:
    digest = digest or hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def file_provenance(path: str | Path) -> dict:
    """``{path, size, sha256, mtime}`` of a file, or of a directory (partitioned
    parquet: files hashed in sorted relative-path order, names included)."""
    p = Path(resolve_path(str(path))).resolve()
    if p.is_file():
        stat = p.stat()
        return {
            "path": str(p),
            "size": stat.st_size,
            "sha256": _sha256_file(p),
            "mtime": _iso(stat.st_mtime),
        }
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        digest = hashlib.sha256()
        for f in files:
            digest.update(str(f.relative_to(p)).encode() + b"\0")
            _sha256_file(f, digest)
        return {
            "path": str(p),
            "size": sum(f.stat().st_size for f in files),
            "sha256": digest.hexdigest(),
            "mtime": _iso(max((f.stat().st_mtime for f in files), default=0)),
            "n_files": len(files),
        }
    raise SourceError(f"source not found: {path}")


def _sources(ws: Workspace) -> list[dict]:
    out = []
    for role in ("train", "test"):
        dataset = getattr(ws.datasets, role)
        if dataset is None:
            continue
        for part in ("x", "y"):
            spec = getattr(dataset, part)
            if spec is None:
                continue
            out.append(
                {
                    "role": role,
                    "part": part,
                    "spec": spec.model_dump(mode="json"),
                    **file_provenance(spec.path),
                }
            )
    return out


def _merges(ws: Workspace) -> list[dict]:
    out = []
    for m in ws.merges:
        entry = m.model_dump(mode="json")
        entry.update(file_provenance(m.source.path))
        out.append(entry)
    return out



def versions() -> dict:
    """Versions of everything that shapes the output."""
    import pyarrow
    import sklearn

    try:
        engine = version("dtk-engine")
    except PackageNotFoundError:  # pragma: no cover  (source checkout, not installed)
        engine = "unknown"
    return {
        "dtk_engine": engine,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "python": platform.python_version(),
    }


def _write_parquet(df: pd.DataFrame, path: Path) -> dict:
    # Parquet needs string column names; the index is not kept (row order is).
    df = df.rename(columns=str)
    try:
        df.to_parquet(path, index=False)
    except (ValueError, TypeError, ImportError) as exc:
        raise SourceError(
            f"cannot write {path.name}: {exc} (a column mixing numbers and text? "
            "add a `cast` step)"
        ) from exc
    return {
        "path": str(path.relative_to(path.parent.parent)),
        "rows": len(df),
        "columns": [str(c) for c in df.columns],
        "sha256": _sha256_file(path),
    }


def _step_entry(i: int, step, fitted: dict, out_dir: Path, inline_limit: int) -> dict:
    entry = {
        "index": i,
        "op": step.op,
        "target": step.target,
        "params": step.params,
        "fitted_on": fitted["fitted_on"],
    }
    blob = json.dumps(fitted["state"])
    if len(blob.encode()) <= inline_limit:
        entry["state"] = fitted["state"]
        return entry
    rel = Path(STATES_DIR) / f"step_{i}_{step.op}.json"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(blob, encoding="utf-8")
    entry["state_file"] = rel.as_posix()
    entry["state_bytes"] = path.stat().st_size
    entry["state_sha256"] = _sha256_file(path)
    return entry


def _owned_paths(out_dir: Path) -> list[Path]:
    """Files a previous export of ours listed, validated; raises before any delete."""
    manifest = out_dir / MANIFEST
    try:
        old = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KeyParamsError(f"{manifest} is unreadable: {exc}") from exc
    if not isinstance(old, dict) or old.get("generator") != GENERATOR:
        raise KeyParamsError(
            f"{manifest} was not written by {GENERATOR}; refusing to overwrite"
        )
    try:
        listed = [o["path"] for o in old.get("outputs", {}).values()]
        listed += [s["state_file"] for s in old.get("steps", []) if "state_file" in s]
    except (TypeError, KeyError, AttributeError) as exc:
        raise KeyParamsError(f"{manifest} is malformed: {exc!r}") from exc
    owned = []
    for rel in listed:
        if not isinstance(rel, str) or Path(rel).is_absolute():
            raise KeyParamsError(f"{manifest} lists a non-relative path: {rel!r}")
        path = (out_dir / rel).resolve()
        if not any(
            path.is_relative_to(out_dir / d) for d in (PROCESSED_DIR, STATES_DIR)
        ):
            raise KeyParamsError(
                f"{manifest} lists {rel!r}, outside {PROCESSED_DIR}/ and {STATES_DIR}/"
            )
        owned.append(path)
    return owned


def _remove_previous(out_dir: Path) -> None:
    """Delete the files a previous export listed (and nothing else)."""
    owned = _owned_paths(out_dir)  # all validated before the first unlink
    for path in owned:
        path.unlink(missing_ok=True)
    (out_dir / MANIFEST).unlink()


def export_workspace(
    name: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    store=None,
    inline_state_bytes: int = INLINE_STATE_BYTES,
) -> dict:
    """Export the workspace's processed train / test and its manifest; returns it.

    An existing export in ``out_dir`` (a ``manifest.json``) is refused unless
    ``overwrite``, which first removes the files that export listed.
    """
    store = store if store is not None else JsonWorkspaceStore()
    ws = store.get(name)
    out = Path(out_dir).resolve()
    sources = _sources(ws)  # hashed before reading: provenance of what was read
    merges = _merges(ws)

    source_paths = {Path(s["path"]) for s in sources} | {
        Path(m["path"]) for m in merges
    }
    targets = [out / MANIFEST, out / PROCESSED_DIR, out / STATES_DIR]
    for src in source_paths:
        if any(src == t or t in src.parents for t in targets):
            raise KeyParamsError(
                f"export to {out} would overwrite the raw input {src}; "
                "choose another directory"
            )
    if (out / MANIFEST).exists():
        if not overwrite:
            raise KeyParamsError(
                f"{out / MANIFEST} exists (a previous export); pass overwrite=True"
            )
        _remove_previous(out)

    train = raw_workspace_frame(ws, "train", labeled=True)
    test = None
    if ws.datasets.test is not None:
        test = raw_workspace_frame(ws, "test", labeled=True)
    train, test, fitted = replay_fitted(ws.steps, train, test)

    (out / PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    outputs = {"train": _write_parquet(train, out / PROCESSED_DIR / "train.parquet")}
    if test is not None:
        outputs["test"] = _write_parquet(test, out / PROCESSED_DIR / "test.parquet")

    manifest = {
        "generator": GENERATOR,
        "manifest_version": MANIFEST_VERSION,
        "workspace": ws.name,
        "exported_at": datetime.now(UTC).isoformat(),
        "versions": versions(),
        "sources": sources,
        "label": ws.label.model_dump(mode="json"),
        "merges": merges,
        "variables": [v.model_dump(mode="json") for v in ws.variables],
        "steps": [
            _step_entry(i, step, f, out, inline_state_bytes)
            for i, (step, f) in enumerate(zip(ws.steps, fitted, strict=True))
        ],
        "outputs": outputs,
    }
    tmp = out / (MANIFEST + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(out / MANIFEST)
    return manifest


def load_state(out_dir: str | Path, step_entry: dict) -> dict:
    """A manifest step's fitted state, inline or read from its side file."""
    if "state" in step_entry:
        return step_entry["state"]
    path = Path(out_dir) / step_entry["state_file"]
    if _sha256_file(path) != step_entry["state_sha256"]:
        raise SourceError(f"state file {path} does not match its manifest sha256")
    return json.loads(path.read_text(encoding="utf-8"))
