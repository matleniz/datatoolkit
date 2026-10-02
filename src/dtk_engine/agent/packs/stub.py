"""Stub pack: a scripted agent, no network, no CLI.

``run_script`` drives an MCP ``Server`` through the SDK's in-memory ``Client``
(``[{tool, args}]`` -> ``[{tool, is_error, result}]``). Used by the tests and the
phase-3 end-to-end check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp import Client

from dtk_engine.agent.packs.base import Pack


def _decode(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return text


@dataclass(frozen=True)
class StubPack(Pack):
    def detect(self, path: str | None = None) -> bool:
        return True

    async def run_script(self, server: Any, script: list[dict]) -> list[dict]:
        """Call each ``{tool, args?}`` in order on ``server``; the decoded text results."""
        out = []
        async with Client(server) as client:
            for step in script:
                result = await client.call_tool(step["tool"], step.get("args") or {})
                texts = [getattr(c, "text", "") for c in result.content]
                out.append({
                    "tool": step["tool"],
                    "is_error": bool(result.is_error),
                    "result": _decode(texts[0]) if len(texts) == 1 else texts,
                })
        return out


PACK = StubPack(
    id="stub",
    title="Scripted stub (tests)",
    panel="external",
    cli=None,
    auth="none",
    cost="free: no model, no network",
)
