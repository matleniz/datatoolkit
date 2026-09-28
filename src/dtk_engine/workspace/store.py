"""WorkspaceStore: where workspaces live. JSON files today."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace.models import NAME_PATTERN, Workspace


class WorkspaceNotFoundError(KeyError):
    """No workspace with this name in the store."""


class WorkspaceStore(Protocol):
    def list(self) -> list[str]: ...

    def get(self, name: str) -> Workspace: ...

    def save(self, workspace: Workspace) -> None: ...

    def delete(self, name: str) -> None: ...

    def rename(self, name: str, new_name: str) -> Workspace: ...

    def duplicate(self, name: str, new_name: str) -> Workspace: ...


def default_root() -> Path:
    """``$DTK_HOME/workspaces``, ``~/.datatoolkit/workspaces`` when unset."""
    home = os.environ.get("DTK_HOME")
    base = Path(home) if home else Path.home() / ".datatoolkit"
    return base / "workspaces"


class JsonWorkspaceStore:
    """One ``<name>.json`` per workspace under ``root``.

    Mutations (save / delete / rename / duplicate) take an exclusive flock on
    ``root/.lock`` so a concurrent PUT cannot race a rename or delete and lose
    an update (MAT-171). Reads stay lock-free: writes are atomic (tmp + replace).
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _path(self, name: str) -> Path:
        if not re.match(NAME_PATTERN, name):
            raise KeyParamsError(f"invalid workspace name {name!r}")
        return self.root / f"{name}.json"

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / ".lock", "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def list(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def path_of(self, name: str) -> Path:
        """Validated path for ``name`` (for mtime / existence checks)."""
        return self._path(name)

    def get(self, name: str) -> Workspace:
        path = self._path(name)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise WorkspaceNotFoundError(name) from None
        try:
            return Workspace.model_validate_json(raw)
        except ValidationError as exc:
            raise KeyParamsError(f"invalid workspace file {path}: {exc}") from exc

    def _save_unlocked(self, workspace: Workspace) -> None:
        path = self._path(workspace.name)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(workspace.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def save(self, workspace: Workspace) -> None:
        with self._exclusive():
            self._save_unlocked(workspace)

    def delete(self, name: str) -> None:
        with self._exclusive():
            try:
                self._path(name).unlink()
            except FileNotFoundError:
                raise WorkspaceNotFoundError(name) from None

    def rename(self, name: str, new_name: str) -> Workspace:
        """Move ``name`` to ``new_name`` (same content, updated ``name`` field).

        Source paths stay as-is. Fails if ``new_name`` already exists. Same-name
        is a no-op. Under the store lock so a concurrent save cannot recreate
        the old name after the move (MAT-171).
        """
        if not re.match(NAME_PATTERN, new_name):
            raise KeyParamsError(f"invalid workspace name {new_name!r}")
        with self._exclusive():
            if name == new_name:
                return self.get(name)
            ws = self.get(name)
            new_path = self._path(new_name)
            if new_path.exists():
                raise KeyParamsError(f"workspace {new_name!r} already exists")
            renamed = ws.model_copy(update={"name": new_name})
            self._save_unlocked(renamed)
            self._path(name).unlink()
            return renamed

    def duplicate(self, name: str, new_name: str) -> Workspace:
        """Copy ``name`` to ``new_name``; content-addressed source refs stay shared.

        Uploaded files are not copied. Fails if ``new_name`` already exists.
        """
        if not re.match(NAME_PATTERN, new_name):
            raise KeyParamsError(f"invalid workspace name {new_name!r}")
        with self._exclusive():
            if name == new_name:
                raise KeyParamsError(
                    f"duplicate target must differ from source ({name!r})"
                )
            ws = self.get(name)
            new_path = self._path(new_name)
            if new_path.exists():
                raise KeyParamsError(f"workspace {new_name!r} already exists")
            copy = ws.model_copy(update={"name": new_name})
            self._save_unlocked(copy)
            return copy
