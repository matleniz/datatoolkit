"""Pack registry: one descriptor per agent CLI (``base.Pack``)."""

from __future__ import annotations

from dtk_engine.agent.packs import claude_code, gemini, opencode, stub
from dtk_engine.agent.packs.base import Pack, ServerRef
from dtk_engine.errors import KeyParamsError

PACKS: dict[str, Pack] = {
    p.id: p for p in (claude_code.PACK, gemini.PACK, opencode.PACK, stub.PACK)
}


def get_pack(pack_id: str) -> Pack:
    try:
        return PACKS[pack_id]
    except KeyError:
        raise KeyParamsError(
            f"unknown pack {pack_id!r}; known: {', '.join(PACKS)}"
        ) from None


__all__ = ["PACKS", "Pack", "ServerRef", "get_pack"]
