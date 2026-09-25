"""WorkspaceStore: where workspaces live. JSON files today."""

from __future__ import annotations

import json
import os
import re
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


def default_root() -> Path:
    """``$DTK_HOME/workspaces``, ``~/.datatoolkit/workspaces`` when unset."""
    home = os.environ.get("DTK_HOME")
    base = Path(home) if home else Path.home() / ".datatoolkit"
    return base / "workspaces"


class JsonWorkspaceStore:
    """One ``<name>.json`` per workspace under ``root``."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _path(self, name: str) -> Path:
        if not re.match(NAME_PATTERN, name):
            raise KeyParamsError(f"invalid workspace name {name!r}")
        return self.root / f"{name}.json"

    def list(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

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

    def save(self, workspace: Workspace) -> None:
        path = self._path(workspace.name)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(workspace.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def delete(self, name: str) -> None:
        try:
            self._path(name).unlink()
        except FileNotFoundError:
            raise WorkspaceNotFoundError(name) from None
