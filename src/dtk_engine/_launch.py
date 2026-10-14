"""Script entry points of ``dtk-api`` and ``dtk-mcp``.

Both scripts ship with the bare package, but their dependencies live in the
extras ``api`` / ``agent``. These wrappers import the real entry lazily and turn
a missing extra into a one-line message (exit 1) instead of a traceback.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

REPO = "git+https://github.com/matleniz/datatoolkit"
# Third-party modules each extra provides; anything else missing is a real bug.
_EXTRA_MODULES = {"api": {"fastapi", "uvicorn", "starlette"}, "agent": {"anyio", "mcp", "httpx"}}


def _launch(script: str, extra: str, target: str) -> None:
    module, _, attr = target.partition(":")
    try:
        entry: Callable[..., None] = getattr(importlib.import_module(module), attr)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in _EXTRA_MODULES[extra]:
            raise
        sys.exit(
            f'{script} needs the "{extra}" extra: '
            f'uv pip install "dtk-engine[{extra}] @ {REPO}"'
        )
    entry()


def api_main() -> None:
    _launch("dtk-api", "api", "dtk_engine.http:main")


def mcp_main() -> None:
    _launch("dtk-mcp", "agent", "dtk_engine.agent.cli:main")
