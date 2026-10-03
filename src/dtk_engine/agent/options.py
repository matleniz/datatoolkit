"""``GET /api/ui/agent/options``: the packs Studio may offer, as JSON.

Chat packs come from the hub's registry (``chat.Pack``); the terminal packs
(``claude-code``, ``gemini``, ``opencode``: panel ``terminal``, mode ``cli``)
are the external CLIs of ``packs/``, offered only when the terminal is on
(``DTK_AGENT_TERMINAL``). Model listers are shared by pack so that one
discovery serves ``agent-sdk`` and ``claude-code``.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from dtk_engine.agent.models import ModelList, gemini_models, opencode_models
from dtk_engine.agent.packs import PACKS as EXTERNAL_PACKS

TERMINAL_IDS = ("claude-code", "gemini", "opencode")
TERMINAL_OFF = "terminal off: set DTK_AGENT_TERMINAL=1"
_TRUE = ("1", "true", "yes", "on")

Lister = Callable[[], Awaitable[ModelList]]


def terminal_enabled() -> bool:
    return os.environ.get("DTK_AGENT_TERMINAL", "").strip().lower() in _TRUE


def terminal_lister(pack_id: str) -> Lister:
    """The model lister of a terminal pack."""
    if pack_id == "claude-code":
        from dtk_engine.agent.packs.agent_sdk import discover_models

        return discover_models
    return opencode_models if pack_id == "opencode" else gemini_models


def terminal_reason(pack_id: str) -> str | None:
    """Why a terminal pack cannot be opened (None = ready)."""
    if not terminal_enabled():
        return TERMINAL_OFF
    pack = EXTERNAL_PACKS[pack_id]
    return None if pack.detect() else f"{pack.cli} not found on PATH"


def entry(
    *, id: str, title: str, mode: str, panel: str, reason: str | None,
    provider: str | None, default_model: str | None, listing: ModelList,
) -> dict:
    return {
        "id": id,
        "title": title,
        "mode": mode,
        "panel": panel,
        "available": reason is None,
        "reason": reason,
        "provider": provider,
        "default_model": default_model,
        **listing.to_json(),
    }
