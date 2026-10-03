"""Stub pack: a scripted agent, no network, no CLI. Two faces:

- ``PACK`` (registry, ``base.Pack``): ``run_script`` drives an MCP ``Server``
  through the SDK's in-memory ``Client`` (``[{tool, args}]`` ->
  ``[{tool, is_error, result}]``). Used by the tests and the phase-3
  end-to-end check.
- ``chat_pack()`` (in-Studio chat, ``chat.Pack``, ``DTK_AGENT_PACK=stub``):
  scripted turns over the real dtk tools, for unit / e2e tests:

  - text with ``permission`` -> ``permission_request`` first (deny -> failed
    ``tool_result``, allow -> as ``add a step``);
  - text with ``add a step`` -> ``propose_steps`` adding ``scale`` on ``age``
    through the real MCP server and UI bridge (workspace / base_identity filled
    from the Studio context), then a one-line reply;
  - text ``drop <column>`` -> the same with ``drop_columns [<column>]``, a
    destructive proposal Studio holds for review (``pending: "review"``);
  - anything else -> echoed back.

  Usage is fake but deterministic: 10 input + 5 output tokens per turn.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from mcp import Client

from dtk_engine.agent import chat as _chat
from dtk_engine.agent.packs.base import Pack
from dtk_engine.agent.packs.chat_packs import parse_tool_text, tool_result_fields

STEP = {"op": "scale", "target": "both", "params": {"columns": ["age"]}}
TURN_USAGE = (10, 5)
_DROP = re.compile(r"\bdrop (\w+)", re.IGNORECASE)


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
                    "result": parse_tool_text(texts[0]) if len(texts) == 1 else texts,
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


class StubAdapter:
    def __init__(self) -> None:
        self.chat: _chat.ChatSession | None = None
        self._calls = 0

    async def start(self, chat: _chat.ChatSession) -> None:
        self.chat = chat

    async def send(self, text: str) -> str | None:
        assert self.chat is not None
        chat = self.chat
        lowered = text.lower()
        reply = f"stub: {text}"
        drop = _DROP.search(text)
        if "permission" in lowered or "add a step" in lowered:
            reply = await self._add_step(chat, ask="permission" in lowered)
        elif drop:
            step = {"op": "drop_columns", "target": "both", "params": {"columns": [drop[1]]}}
            reply = await self._add_step(chat, ask=False, step=step)
        await asyncio.sleep(0)  # a real pack yields between events; so does the stub
        chat.emit("assistant_delta", text=reply)
        chat.set_turn_usage(*TURN_USAGE)
        return "end_turn"

    async def _add_step(
        self, chat: _chat.ChatSession, *, ask: bool, step: dict | None = None
    ) -> str:
        self._calls += 1
        tool_id = f"stub-{self._calls}"
        step = step or STEP
        args: dict[str, Any] = {"ops": [{"add": {"step": step}}], "session": chat.session}
        chat.emit("tool_call", id=tool_id, name="propose_steps", input=args)
        if ask and not await chat.ask("propose_steps", args, "Add a step: scale age"):
            chat.emit("tool_result", id=tool_id, ok=False, error="denied")
            return "stub: not added (permission denied)"
        async with Client(chat.mcp_server()) as client:
            result = await client.call_tool("propose_steps", args)
        text = "".join(getattr(c, "text", "") for c in result.content)
        fields = tool_result_fields(parse_tool_text(text), is_error=bool(result.is_error))
        chat.emit("tool_result", id=tool_id, **fields)
        if fields.get("pending"):
            return "stub: waiting for your review in Studio"
        if not fields["ok"]:
            return "stub: step not added"
        return "stub: added a scale step on age" if step is STEP else f"stub: added {step['op']}"

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        self.chat = None


def chat_pack() -> _chat.Pack:
    return _chat.Pack(
        id="stub",
        provider=lambda: "stub (no network)",
        model=lambda: "stub",
        detect=lambda: None,
        create=StubAdapter,
    )
