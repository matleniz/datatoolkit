"""Chat packs: ``stub`` (scripted, no network) and ``agent-sdk`` (Claude Agent SDK).

A chat pack (``chat.Pack``) runs the agent loop inside the engine for Studio's
agent panel; the registry in ``packs/__init__.py`` (``base.Pack``) describes
external CLIs the user runs. ``stub`` is both.

``DTK_AGENT_PACK`` picks one (unset / ``off`` = no agent); ``hub_from_env``
builds the app's ``AgentHub``. Shared here: ``tool_result_fields`` turns a dtk
tool answer (``{identity, data}`` or the ``{type, message}`` error body) into
the ``tool_result`` event fields.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from dtk_engine.agent.chat import AgentHub, Pack
from dtk_engine.agent.policy import AuditLog
from dtk_engine.ui_bridge import UiBridge

SUMMARY_CHARS = 200
TOOL_PREFIX = "mcp__dtk__"
OFF = ("", "0", "off", "none", "false", "no")


def _loaders() -> dict[str, Callable[[], Pack]]:
    from dtk_engine.agent.packs import agent_sdk, stub

    return {"stub": stub.chat_pack, "agent-sdk": agent_sdk.chat_pack}


def pack_names() -> list[str]:
    return list(_loaders())


def hub_from_env(bridge: UiBridge, audit: AuditLog | None = None) -> AgentHub:
    """The hub for ``DTK_AGENT_PACK`` (an unknown name is reported, not raised).

    The offered packs are every registered one except ``stub``, which is only
    offered when it is the default.
    """
    name = os.environ.get("DTK_AGENT_PACK", "").strip().lower()
    if name in OFF:
        reason = "agent off: set DTK_AGENT_PACK=agent-sdk (or run dtk-api --agent)"
        return AgentHub(bridge, None, audit=audit, unavailable=reason)
    loaders = _loaders()
    loader = loaders.get(name)
    if loader is None:
        reason = f"unknown DTK_AGENT_PACK {name!r} (known: {', '.join(pack_names())})"
        return AgentHub(bridge, None, audit=audit, unavailable=reason)
    default = loader()
    packs = {i: load() for i, load in loaders.items() if i != "stub"}
    packs[default.id] = default
    return AgentHub(bridge, default, audit=audit, packs=packs)


def bare_tool_name(name: str) -> str:
    return name.removeprefix(TOOL_PREFIX)


def _summary(data: Any) -> str:
    text = data if isinstance(data, str) else json.dumps(data, default=str)
    return text if len(text) <= SUMMARY_CHARS else text[: SUMMARY_CHARS - 1] + "…"


def tool_result_fields(payload: Any, *, is_error: bool) -> dict:
    """``{ok, summary?, error?, identity?, pending?, command?}`` of one tool answer."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": _summary(payload)} if is_error else {
            "ok": True, "summary": _summary(payload),
        }
    if is_error:
        return {"ok": False, "error": str(payload.get("message") or payload)}
    data = payload.get("data")
    out: dict[str, Any] = {"ok": True}
    if isinstance(data, dict) and "ok" in data and "id" in data:  # a UI command ack
        out["command"] = data["id"]
        if data.get("ok") is False:
            out.update(ok=False, error=str(data.get("error") or "failed"))
        if data.get("pending"):
            out["pending"] = data["pending"]
    else:
        out["summary"] = _summary(data)
    identity = payload.get("identity")
    if identity:
        out["identity"] = identity
    return out


def parse_tool_text(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text
