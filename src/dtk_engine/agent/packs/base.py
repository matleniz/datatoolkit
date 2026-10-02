"""Pack descriptor: how one agent CLI is wired to the ``dtk`` MCP server.

A pack only *describes* the wiring: the config files to write, the command the
user runs, what it turns off. External packs never spawn anything; the user
runs their own CLI next to Studio.

``ServerRef`` is where the CLI reaches the server: ``{"command": [...],
"env"?: {...}}`` (stdio, the default: the config survives ``dtk-api``
restarts and holds no token) or ``{"url": ..., "headers": {...}}`` (``--http``,
valid for one engine run only).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ServerRef = dict[str, Any]
SERVER_NAME = "dtk"


def is_http(server: ServerRef) -> bool:
    return "url" in server


@dataclass(frozen=True)
class Pack:
    id: str
    title: str
    panel: Literal["external", "chat", "terminal"]
    cli: str | None
    auth: str
    cost: str

    def detect(self, path: str | None = None) -> bool:
        """The pack's CLI is on ``path`` (default ``$PATH``)."""
        return self.cli is not None and shutil.which(self.cli, path=path) is not None

    def files(self, server: ServerRef) -> dict[str, dict]:
        """Relative path -> JSON content of each config file to write."""
        return {}

    def launch_command(self, directory: Path) -> list[str]:
        """What the user runs, from ``directory`` (where the files were written)."""
        return []

    def tool_policy(self) -> str:
        """One paragraph: what the config turns off, what it cannot."""
        return ""
