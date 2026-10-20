"""Workspace export: processed data + code + a provenance manifest.

``export_workspace(name, out_dir)`` writes, under ``out_dir``::

    processed/train.parquet     train after every train / both step (labels joined)
    processed/test.parquet      test after every test / both step (if a test set)
    processed/{train,test}.csv  same frames as csv (format ``csv``)
    states/step_<i>_<op>.json   fitted states too big to inline (e.g. impute_knn)
    code/pipeline.ipynb         the pipeline as a notebook (format ``ipynb``)
    code/pipeline.py            the pipeline as a script (format ``py``)
    manifest.json               written last: its presence marks a complete export

``formats`` picks the data / code outputs (default parquet only):
``parquet``, ``csv``, ``ipynb`` and ``py``, the last two replaying the
pipeline through the notebook door (``workspace.notebook``). Parquet refuses
duplicate column names (csv writes them as they are).

The manifest records the sources (path, size, sha256, mtime), the export time,
the step groups, every step (with its ``group_id``) with its fitted state
(inline, or a side file with its sha256) and the engine / pandas / sklearn /
pyarrow / python versions. Raw inputs are only read: an output path that is a
source path is refused. Overwrite only deletes relative paths under
processed/, states/ or code/ listed by a manifest of ours.

The export is built in a ``.export-staging-*`` directory inside ``out_dir`` and
moved in (the previous one set aside in ``.export-old-*`` meanwhile) only once
every format is written: a failing export (replay or write time) leaves
``out_dir``, a previous export included, untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.sources.csv_pandas import resolve_path
from dtk_engine.workspace.dataset import raw_workspace_frame, replay_label
from dtk_engine.workspace.models import Workspace
from dtk_engine.workspace.notebook import notebook_json, script_text
from dtk_engine.workspace.replay import replay_fitted
from dtk_engine.workspace.store import JsonWorkspaceStore

MANIFEST_VERSION = 1
GENERATOR = "dtk_engine"
MANIFEST = "manifest.json"
PROCESSED_DIR = "processed"
STATES_DIR = "states"
CODE_DIR = "code"
FORMATS = ("parquet", "csv", "ipynb", "py")
# A fitted state whose JSON exceeds this goes to a side file (impute_knn and
# impute_iterative keep the whole train matrix in their state).
INLINE_STATE_BYTES = 64 * 1024
_CHUNK = 1 << 20
MAX_LISTED_COLUMNS = 10
STAGING_PREFIX = ".export-staging-"
ASIDE_PREFIX = ".export-old-"
STAGING_MARKER = ".dtk-staging"  # holds the pid of the exporting process
STAGING_MAX_AGE = 3600  # seconds before a staging dir of a live pid counts as stale
_IS_WINDOWS = sys.platform == "win32"  # monkeypatched in tests


class ExportRestoreError(OSError):
    """The swap failed and the previous export could not be fully put back."""


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


def _documents(ws: Workspace) -> list[dict]:
    """Workspace documents, listed (not copied); a missing file is flagged, not fatal."""
    out = []
    for doc in ws.documents:
        entry = doc.model_dump(mode="json", exclude={"kind", "added_at"})
        try:
            entry["sha256"] = file_provenance(doc.path)["sha256"]
        except (SourceError, OSError):
            entry["missing"] = True
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
    dups = list(dict.fromkeys(df.columns[df.columns.duplicated()]))
    if dups:
        shown = dups[:MAX_LISTED_COLUMNS]
        more = f" (and {len(dups) - len(shown)} more)" if len(dups) > len(shown) else ""
        raise SourceError(
            f"cannot write parquet: duplicate column names {shown}{more} in "
            f"{path.stem} (rename them with a rename step)"
        )
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


def _write_csv(df: pd.DataFrame, path: Path) -> dict:
    df = df.rename(columns=str)
    df.to_csv(path, index=False)
    return {
        "path": str(path.relative_to(path.parent.parent)),
        "rows": len(df),
        "columns": [str(c) for c in df.columns],
        "sha256": _sha256_file(path),
    }


def _write_text(text: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path.relative_to(path.parent.parent)), "sha256": _sha256_file(path)}


def _check_formats(formats: list[str] | None) -> list[str]:
    if formats is None:
        formats = ["parquet"]
    elif not isinstance(formats, (list, tuple)) or not all(isinstance(f, str) for f in formats):
        shown = repr(formats)
        shown = shown if len(shown) <= 40 else shown[:37] + "..."
        raise KeyParamsError(f"formats must be a list of {list(FORMATS)}, got {shown}")
    formats = list(dict.fromkeys(formats))  # duplicates dropped, first-seen order
    unknown = [f for f in formats if f not in FORMATS]
    if unknown or not formats:
        raise KeyParamsError(f"formats must be a non-empty subset of {list(FORMATS)}, got {formats}")
    return formats


def _write_outputs(
    ws: Workspace, out: Path, frames: dict, formats: list[str], exported_at: str
) -> dict:
    outputs: dict[str, dict] = {}
    if "parquet" in formats or "csv" in formats:
        (out / PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    for role, df in frames.items():
        if "parquet" in formats:
            outputs[role] = _write_parquet(df, out / PROCESSED_DIR / f"{role}.parquet")
        if "csv" in formats:
            outputs[f"{role}_csv"] = _write_csv(df, out / PROCESSED_DIR / f"{role}.csv")
    if "ipynb" in formats:
        text = notebook_json(ws, exported_at)
        outputs["notebook"] = _write_text(text, out / CODE_DIR / "pipeline.ipynb")
    if "py" in formats:
        outputs["script"] = _write_text(script_text(ws, exported_at), out / CODE_DIR / "pipeline.py")
    return outputs


def _groups(ws: Workspace) -> list[dict]:
    """Groups with their step positions, so a reader can map steps to groups."""
    pos = {s.id: i for i, s in enumerate(ws.steps)}
    return [
        {
            "id": g.id,
            "name": g.name,
            "intent": g.intent,
            "steps": [pos[sid] for sid in g.step_ids],
        }
        for g in ws.groups
    ]


def _step_entry(
    i: int, step, fitted: dict, out_dir: Path, inline_limit: int, group_id: str
) -> dict:
    entry = {
        "index": i,
        "group_id": group_id,
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
            path.is_relative_to(out_dir / d) for d in (PROCESSED_DIR, STATES_DIR, CODE_DIR)
        ):
            raise KeyParamsError(
                f"{manifest} lists {rel!r}, outside {PROCESSED_DIR}/, {STATES_DIR}/ "
                f"and {CODE_DIR}/"
            )
        owned.append(path)
    return owned


def _move_files(src: Path, dst: Path, files: list[Path] | None = None) -> list[tuple[Path, Path]]:
    """``os.replace`` each file under ``src`` (or the listed ones) to the same relative
    path under ``dst``; returns the (from, to) pairs done."""
    done: list[tuple[Path, Path]] = []
    for f in files if files is not None else sorted(x for x in src.rglob("*") if x.is_file()):
        dest = dst / f.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, dest)
        done.append((f, dest))
    return done


def _swap(new: Path, aside: Path, out: Path, owned: list[Path]) -> None:
    """Move the previous owned files to ``aside``, then the staged ones into ``out``
    (manifest last); on any error put the previous export back and drop the new files.
    Foreign files are never touched."""
    previous = [p for p in [*owned, out / MANIFEST] if p.exists()]
    moved: list[tuple[Path, Path]] = []
    installed: list[tuple[Path, Path]] = []
    staged = sorted((f for f in new.rglob("*") if f.is_file()), key=lambda f: f.name == MANIFEST)
    try:
        for path in previous:
            moved += _move_files(out, aside, [path])
        for f in staged:
            installed += _move_files(new, out, [f])
    except BaseException:
        for _, dest in installed:
            dest.unlink(missing_ok=True)
        failed = []
        for path, backup in moved:
            try:
                os.replace(backup, path)
            except OSError as exc:
                failed.append(exc)
        if failed:
            raise ExportRestoreError(
                f"export failed ({sys.exc_info()[1]!r}) and the previous export could not be "
                f"fully restored ({failed[0]}); previous export kept in {aside}"
            ) from sys.exc_info()[1]
        raise


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` may still be running (``True`` when unsure: never delete a live one).

    Never probes on Windows: there ``os.kill`` TERMINATES the process for any signal other
    than CTRL_C_EVENT/CTRL_BREAK_EVENT, so ``os.kill(pid, 0)`` would kill the live export
    (or an unrelated process owning a recycled pid). The caller then relies on the marker
    age alone. On POSIX, signal 0 only checks existence; EPERM means it exists, and any
    other OSError is treated as alive (the conservative choice). A recycled pid owned by an
    unrelated live process also reads as alive: the age rule reclaims it after an hour."""
    if _IS_WINDOWS:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM (the process exists) or anything unexpected: assume alive
        return True
    return True


def _staging_is_stale(staging: Path) -> bool:
    """Only a staging dir carrying our marker, of a dead pid or older than the max age."""
    marker = staging / STAGING_MARKER
    try:
        pid = int(marker.read_text(encoding="utf-8"))
        age = datetime.now(UTC).timestamp() - marker.stat().st_mtime
    except (OSError, ValueError):
        return False  # not ours (or unreadable): never delete
    return age > STAGING_MAX_AGE or not _pid_alive(pid)


def _recover_stale(out: Path) -> None:
    """Leftovers of a crashed export: staging dirs are dropped; an aside dir holds the
    only copy of a previous export, so it is moved back when ``out`` has no manifest and
    otherwise left untouched. Concurrent exports to one ``out_dir`` stay unsupported, but
    a live one's staging dir (marker with a live pid, recent) is never removed."""
    if not out.is_dir():
        return
    for stale in out.glob(STAGING_PREFIX + "*"):
        if _staging_is_stale(stale):
            shutil.rmtree(stale, ignore_errors=True)
    if (out / MANIFEST).exists():
        return
    for aside in out.glob(ASIDE_PREFIX + "*"):
        _move_files(aside, out)
        shutil.rmtree(aside, ignore_errors=True)


def export_workspace(
    name: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    store=None,
    inline_state_bytes: int = INLINE_STATE_BYTES,
    formats: list[str] | None = None,
) -> dict:
    """Export the workspace's processed train / test and its manifest; returns it.

    An existing export in ``out_dir`` (a ``manifest.json``) is refused unless
    ``overwrite``, which replaces the files that export listed once the new one
    is fully written.
    """
    formats = _check_formats(formats)
    store = store if store is not None else JsonWorkspaceStore()
    ws = store.get(name)
    out = Path(out_dir).resolve()
    sources = _sources(ws)  # hashed before reading: provenance of what was read
    merges = _merges(ws)

    source_paths = {Path(s["path"]) for s in sources} | {
        Path(m["path"]) for m in merges
    }
    targets = [out / MANIFEST, out / PROCESSED_DIR, out / STATES_DIR, out / CODE_DIR]
    for src in source_paths:
        if any(src == t or t in src.parents for t in targets):
            raise KeyParamsError(
                f"export to {out} would overwrite the raw input {src}; "
                "choose another directory"
            )
    _recover_stale(out)
    owned = None
    if (out / MANIFEST).exists():
        if not overwrite:
            raise KeyParamsError(
                f"{out / MANIFEST} exists (a previous export); pass overwrite=True"
            )
        owned = _owned_paths(out)  # validated before anything is built

    created = next((p for p in [*out.parents[::-1], out] if not p.exists()), None)
    out.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out))
    (staging / STAGING_MARKER).write_text(str(os.getpid()), encoding="utf-8")
    new = staging / "new"
    new.mkdir()
    aside = Path(tempfile.mkdtemp(prefix=ASIDE_PREFIX, dir=out))
    keep_aside = False
    try:
        manifest = _build_export(ws, new, formats, sources, merges, inline_state_bytes)
        _swap(new, aside, out, owned or [])
    except ExportRestoreError:
        keep_aside = True  # the only copy of the previous export
        raise
    except BaseException:
        if created is not None:
            shutil.rmtree(created, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not keep_aside:
            shutil.rmtree(aside, ignore_errors=True)
    return manifest


def _build_export(
    ws: Workspace, out: Path, formats: list[str], sources: list[dict], merges: list[dict],
    inline_state_bytes: int,
) -> dict:
    train = raw_workspace_frame(ws, "train", labeled=True)
    test = None
    if ws.datasets.test is not None:
        test = raw_workspace_frame(ws, "test", labeled=True)
    train, test, fitted = replay_fitted(ws.steps, train, test, replay_label(ws))

    exported_at = datetime.now(UTC).isoformat()
    frames = {"train": train} if test is None else {"train": train, "test": test}
    outputs = _write_outputs(ws, out, frames, formats, exported_at)

    group_of = {sid: g.id for g in ws.groups for sid in g.step_ids}
    manifest = {
        "generator": GENERATOR,
        "manifest_version": MANIFEST_VERSION,
        "workspace": ws.name,
        "exported_at": exported_at,
        "formats": formats,
        "versions": versions(),
        "sources": sources,
        "label": ws.label.model_dump(mode="json"),
        "merges": merges,
        "variables": [v.model_dump(mode="json") for v in ws.variables],
        "documents": _documents(ws),
        "groups": _groups(ws),
        "steps": [
            _step_entry(i, step, f, out, inline_state_bytes, group_of[step.id])
            for i, (step, f) in enumerate(zip(ws.steps, fitted, strict=True))
        ],
        "outputs": outputs,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_state(out_dir: str | Path, step_entry: dict) -> dict:
    """A manifest step's fitted state, inline or read from its side file."""
    if "state" in step_entry:
        return step_entry["state"]
    path = Path(out_dir) / step_entry["state_file"]
    if _sha256_file(path) != step_entry["state_sha256"]:
        raise SourceError(f"state file {path} does not match its manifest sha256")
    return json.loads(path.read_text(encoding="utf-8"))
