"""In-Studio agent chat: the adapter interface, the event protocol and the hub.

Protocol (``docs/agent-chat-protocol.md``): Studio posts to ``/api/ui/agent/*``
(``http.py``) and reads ``event: agent`` frames on its ``/api/ui/events``
stream; ``data`` is one event ``{type, turn, ...}``: ``user_message``,
``assistant_delta``, ``tool_call``, ``tool_result``, ``permission_request``,
``usage``, ``done``, ``error``.

A **pack** (``dtk_engine.agent.packs``) builds one ``Adapter`` per Studio
session. The adapter drives its agent (``start`` / ``send`` / ``cancel`` /
``close``) and reports through the ``ChatSession`` it is started with:
``emit`` (events), ``ask`` (``permission_request`` -> ``permission_reply``),
``set_turn_usage`` (tokens of the running turn, checked against
``DTK_AGENT_MAX_TOKENS``) and ``mcp_server()`` (the dtk MCP server bound to
this app's UI bridge: the agent's only tools). The hub owns turns: one at a
time per session, ``usage`` then ``done`` always closing a turn.

Idle sessions are reaped: ``IDLE_GRACE`` seconds after a Studio session loses
its last SSE listener (closed tab), the hub cancels its turn, closes its
adapter (e.g. the ``claude`` CLI of ``agent-sdk``) and drops its chat state
and attachments. A reconnect within the grace (a reload keeps the id) keeps it.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.server.lowlevel import Server

from dtk_engine.agent import attachments as _attachments
from dtk_engine.agent.models import FREE_TEXT, ModelList
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.agent.tools import build_tools
from dtk_engine.ui_bridge import UiBridge, review_timeout_from_env

EVENT = "agent"
CANCEL_GRACE = 5.0
IDLE_GRACE = 300.0  # seconds a session may have no listener before it is reaped


class NoAgentError(Exception):
    """No pack is configured or the configured one is not usable."""


class AgentBusyError(Exception):
    """A turn is already running for this Studio session."""


def max_tokens_from_env() -> int | None:
    """``DTK_AGENT_MAX_TOKENS`` (positive int: input + output tokens per session), else None."""
    try:
        value = int(os.environ.get("DTK_AGENT_MAX_TOKENS", ""))
    except ValueError:
        return None
    return value if value > 0 else None


class Adapter(Protocol):
    """One agent conversation, owned by one ``ChatSession``."""

    async def start(self, chat: ChatSession) -> None: ...

    async def send(self, text: str) -> str | None:
        """Run one turn; return its stop reason (None = ``end_turn``)."""
        ...

    async def cancel(self) -> None: ...

    async def close(self) -> None: ...


class ConfigError(Exception):
    """``POST /agent/config`` refused: ``code`` is the error ``type`` (a 422)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


async def _free_text() -> ModelList:
    return FREE_TEXT


def _no_refresh() -> None:
    return None


@dataclass(frozen=True)
class Pack:
    """A chat pack descriptor.

    ``detect()`` returns why it is unusable, or None when ready; ``model()`` is the
    env default model (None = the provider's own); ``models()`` discovers what
    the user may pick (never raises, see ``models.ModelList``); ``create(model)``
    builds one adapter for the model the session chose (None = the default).
    An adapter may offer ``async set_model(model)``: the conversation then goes
    on across a model change, otherwise the hub resets it.
    """

    id: str
    provider: Callable[[], str | None]
    model: Callable[[], str | None]
    detect: Callable[[], str | None]
    create: Callable[..., Adapter]
    title: str = ""
    mode: str = "cli"
    panel: str = "chat"
    models: Callable[[], Awaitable[ModelList]] = _free_text
    refresh: Callable[[], None] = _no_refresh  # drop cached detection (options?refresh=1)

    @property
    def default_model(self) -> str | None:
        return self.model()


class ChatSession:
    """The chat state of one Studio session; what an adapter talks to."""

    def __init__(self, hub: AgentHub, session: str) -> None:
        self.hub = hub
        self.session = session
        self.turn: str | None = None
        self.adapter: Adapter | None = None
        self.pack: Pack | None = None  # the session's choice; None = the hub default
        self.model: str | None = None
        self.task: asyncio.Task | None = None
        self.cancelled = False
        self.totals = {"input_tokens": 0, "output_tokens": 0}
        self.turn_usage = {"input_tokens": 0, "output_tokens": 0}
        self._turns = itertools.count(1)
        self._asks = itertools.count(1)
        self._permissions: dict[str, asyncio.Future] = {}

    # -- what adapters call ----------------------------------------------
    def emit(self, type_: str, **fields: Any) -> None:
        self.hub.bridge.emit(self.session, EVENT, {"type": type_, "turn": self.turn, **fields})

    def mcp_server(self) -> Server:
        return self.hub.mcp_server()

    @property
    def session_tools(self) -> frozenset[str]:
        """dtk tools taking a ``session`` argument (adapters pin it to this session)."""
        return self.hub.session_tools

    def set_turn_usage(self, input_tokens: int, output_tokens: int) -> bool:
        """Tokens of the running turn so far; True once the session cap is reached."""
        self.turn_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return self.over_cap()

    def over_cap(self) -> bool:
        cap = self.hub.max_tokens
        return cap is not None and self.used() >= cap

    def used(self) -> int:
        return sum(self.totals.values()) + sum(self.turn_usage.values())

    async def ask(
        self, tool: str, input: dict, summary: str, lines: list[str] | None = None
    ) -> bool:
        """``permission_request``; True when the user allows (denied on timeout / cancel)."""
        pid = f"p{next(self._asks)}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._permissions[pid] = future
        event = {"id": pid, "tool": tool, "input": input, "summary": summary}
        self.emit("permission_request", **event, **({"lines": lines} if lines else {}))
        try:
            return bool(await asyncio.wait_for(future, self.hub.review_timeout))
        except TimeoutError:
            return False
        finally:
            self._permissions.pop(pid, None)

    # -- what the hub calls ----------------------------------------------
    def reply(self, pid: str, allow: bool) -> bool:
        future = self._permissions.get(pid)
        if future is None or future.done():
            return False
        future.set_result(allow)
        return True

    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    def next_turn(self) -> str:
        return f"t{next(self._turns)}"

    def close_turn(self) -> dict:
        """Fold the turn's usage into the totals; the ``usage`` event fields."""
        for key, value in self.turn_usage.items():
            self.totals[key] += value
        event = {
            **self.turn_usage,
            "total_input_tokens": self.totals["input_tokens"],
            "total_output_tokens": self.totals["output_tokens"],
        }
        self.turn_usage = {"input_tokens": 0, "output_tokens": 0}
        return event

    def deny_pending(self) -> None:
        for future in self._permissions.values():
            if not future.done():
                future.set_result(False)


class AgentHub:
    """Chat sessions per Studio session over the offered packs (``pack`` = the default)."""

    def __init__(
        self,
        bridge: UiBridge,
        pack: Pack | None,
        *,
        audit: AuditLog | None = None,
        unavailable: str | None = None,
        max_tokens: int | None = None,
        packs: dict[str, Pack] | None = None,
    ) -> None:
        self.bridge = bridge
        self.pack = pack
        self.packs: dict[str, Pack] = packs if packs is not None else (
            {pack.id: pack} if pack else {}
        )
        self.audit = audit or AuditLog()
        self.max_tokens = max_tokens if max_tokens is not None else max_tokens_from_env()
        self.review_timeout = review_timeout_from_env()
        self._unavailable = unavailable
        self._chats: dict[str, ChatSession] = {}
        self._discovery: dict[Any, asyncio.Future[ModelList]] = {}
        self.idle_grace = IDLE_GRACE
        self._idle: dict[str, asyncio.TimerHandle] = {}
        self._reaping: set[asyncio.Task] = set()
        bridge.listener_hooks.append(self._on_listening)
        port = LocalUiPort(bridge)
        self.session_tools = frozenset(
            spec.name for spec in build_tools(port)
            if "session" in spec.input_schema.get("properties", {})
        )

    def mcp_server(self) -> Server:
        return build_server(LocalUiPort(self.bridge), self.audit)

    def _chat(self, session: str) -> ChatSession:
        chat = self._chats.get(session)
        if chat is None:
            chat = self._chats[session] = ChatSession(self, session)
        return chat

    def pack_of(self, session: str | None) -> Pack | None:
        """The session's pack: its own choice, else the default."""
        chat = self._chats.get(session or "")
        return (chat.pack if chat else None) or self.pack

    def unavailable(self, session: str | None = None) -> str | None:
        """Why no agent can run (None when the session's pack is ready)."""
        pack = self.pack_of(session)
        if pack is None:
            return self._unavailable or "no agent pack configured (DTK_AGENT_PACK)"
        return pack.detect()

    def status(self, session: str | None) -> dict:
        reason = self.unavailable(session)
        chat = self._chats.get(session or "")
        pack = self.pack_of(session)
        out: dict[str, Any] = {
            "available": reason is None,
            "pack": pack.id if pack else None,
            "running": bool(chat and chat.busy()),
            "usage": dict(chat.totals) if chat else {"input_tokens": 0, "output_tokens": 0},
            "max_tokens": self.max_tokens,
        }
        if pack is not None:
            out["provider"] = pack.provider()
            out["model"] = chat.model if chat and chat.pack else pack.model()
            out.update(mode=pack.mode, panel=pack.panel, title=pack.title or pack.id)
        if reason is not None:
            out["reason"] = reason
        return out

    # -- options and per-session selection --------------------------------
    async def models_of(self, lister: Callable[[], Awaitable[ModelList]]) -> ModelList:
        """A lister's answer, run once per run (concurrent callers share it)."""
        future = self._discovery.get(lister)
        if future is None:
            future = self._discovery[lister] = asyncio.ensure_future(lister())
        return await asyncio.shield(future)

    async def options(self, refresh: bool = False) -> dict:
        """The offered packs (chat packs, then terminal packs) and the default."""
        from dtk_engine.agent import options

        if refresh:
            self._discovery.clear()
            for pack in self.packs.values():
                pack.refresh()
        chat_entries = [self._chat_entry(p) for p in self.packs.values()]
        terminal_entries = [self._terminal_entry(i) for i in options.TERMINAL_IDS]
        packs = await asyncio.gather(*chat_entries, *terminal_entries)
        out: dict[str, Any] = {
            "default": {"pack": self.pack.id if self.pack else None, "model": None},
            "packs": list(packs),
        }
        if self.pack is None:
            out.update(available=False, reason=self.unavailable())
        return out

    async def _chat_entry(self, pack: Pack) -> dict:
        from dtk_engine.agent import options

        reason = await asyncio.to_thread(pack.detect)
        listing = FREE_TEXT if reason else await self.models_of(pack.models)
        return options.entry(
            id=pack.id, title=pack.title or pack.id, mode=pack.mode, panel=pack.panel,
            reason=reason, provider=await asyncio.to_thread(pack.provider),
            default_model=pack.default_model, listing=listing,
        )

    async def _terminal_entry(self, pack_id: str) -> dict:
        from dtk_engine.agent import options

        pack = options.EXTERNAL_PACKS[pack_id]
        reason = await asyncio.to_thread(options.terminal_reason, pack_id)
        listing = FREE_TEXT if reason else await self.models_of(options.terminal_lister(pack_id))
        return options.entry(
            id=pack_id, title=pack.title, mode="cli", panel="terminal", reason=reason,
            provider=pack.auth, default_model=None, listing=listing,
        )

    async def configure(self, session: str, pack_id: str, model: str | None) -> dict:
        """Pick the session's pack and model; returns its status (see the protocol doc)."""
        from dtk_engine.agent import options

        if not self.packs:
            raise NoAgentError(self.unavailable())
        chat = self._chat(session)
        if chat.busy():
            raise AgentBusyError("a turn is running: wait for it or cancel it")
        pack = self.packs.get(pack_id)
        if pack is None:
            if pack_id in options.TERMINAL_IDS:
                raise ConfigError("WrongPanel", f"{pack_id} is a terminal pack: open /api/ui/terminal")
            raise ConfigError("UnknownPack", f"unknown pack {pack_id!r} (known: {', '.join(self.packs)})")
        reason = await asyncio.to_thread(pack.detect)
        if reason is not None:
            raise ConfigError("PackUnavailable", reason)
        model = model or None
        if model is not None:
            listing = await self.models_of(pack.models)
            if not listing.free_text and model not in listing.ids():
                raise ConfigError("UnknownModel", f"unknown model {model!r} for {pack.id}")
        reset = await self._apply(chat, pack, model)
        chat.emit("config", pack=pack.id, model=model, reset=reset)
        return self.status(session)

    async def _apply(self, chat: ChatSession, pack: Pack, model: str | None) -> bool:
        """Switch the session; True when its conversation was reset."""
        same_pack = pack.id == (chat.pack or self.pack or pack).id
        adapter = chat.adapter
        switched = False
        if same_pack and adapter is not None and chat.model != model:
            set_model = getattr(adapter, "set_model", None)
            if set_model is not None:
                try:
                    await set_model(model)
                    switched = True
                except Exception:  # noqa: BLE001 - fall back to a fresh conversation
                    switched = False
        keep = adapter is None or (same_pack and (chat.model == model or switched))
        if not keep and adapter is not None:
            with contextlib.suppress(Exception):
                await adapter.close()
            chat.adapter = None
        chat.pack, chat.model = pack, model
        return not (same_pack and keep)

    def send(self, session: str, text: str, attachments: list[str] | None = None) -> str:
        """Start a turn; its events stream on the session's SSE. Returns the turn id.

        ``attachments``: ids of the session's attachments (``UnknownAttachmentError`` if any is
        unknown); the turn echoes them and the adapter's text starts with a note about them.
        """
        reason = self.unavailable(session)
        if reason is not None or self.pack_of(session) is None:
            raise NoAgentError(reason)
        picked = _attachments.registry(self.bridge).pick(session, attachments or [])
        chat = self._chat(session)
        if chat.busy():
            raise AgentBusyError("a turn is already running")
        chat.turn = chat.next_turn()
        chat.cancelled = False
        chat.task = asyncio.create_task(self._run(chat, text, picked))
        return chat.turn

    async def _start(self, chat: ChatSession) -> Adapter:
        if chat.adapter is None:
            pack = chat.pack or self.pack
            assert pack is not None
            adapter = pack.create(chat.model)
            await adapter.start(chat)
            chat.adapter = adapter
        return chat.adapter

    async def _run(self, chat: ChatSession, text: str, picked: list[dict]) -> None:
        extra = {"attachments": _attachments.echo(picked)} if picked else {}
        chat.emit("user_message", text=text, **extra)
        stop = "end_turn"
        try:
            if chat.over_cap():
                stop = "max_tokens"
            else:
                adapter = await self._start(chat)
                stop = await adapter.send(_attachments.turn_note(picked) + text) or "end_turn"
        except asyncio.CancelledError:
            stop = "cancelled"
        except Exception as exc:  # noqa: BLE001 - a pack failure ends the turn, not the server
            chat.emit("error", message=str(exc) or type(exc).__name__, code="pack_error")
            stop = "error"
        if chat.cancelled:
            stop = "cancelled"
        elif chat.over_cap():
            cap = self.max_tokens
            chat.emit("error", message=f"DTK_AGENT_MAX_TOKENS ({cap}) reached", code="max_tokens")
            stop = "max_tokens"
        chat.emit("usage", **chat.close_turn())
        chat.emit("done", stop_reason=stop)
        chat.turn = None

    async def cancel(self, session: str) -> bool:
        """Stop the running turn (the adapter first, then the task); False when idle."""
        chat = self._chats.get(session)
        if chat is None or not chat.busy() or chat.task is None:
            return False
        chat.cancelled = True
        chat.deny_pending()
        if chat.adapter is not None:
            with contextlib.suppress(Exception):
                await chat.adapter.cancel()
        task = chat.task
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), CANCEL_GRACE)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return True

    def reply(self, session: str, pid: str, allow: bool) -> bool:
        chat = self._chats.get(session)
        return chat is not None and chat.reply(pid, allow)

    # -- idle sessions -----------------------------------------------------
    def _on_listening(self, session: str, listening: bool) -> None:
        timer = self._idle.pop(session, None)
        if timer is not None:
            timer.cancel()
        if listening:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # no loop (sync caller): nothing to schedule on
            return
        self._idle[session] = loop.call_later(self.idle_grace, self._start_reap, session)

    def _start_reap(self, session: str) -> None:
        self._idle.pop(session, None)
        task = asyncio.ensure_future(self.reap(session))
        self._reaping.add(task)
        task.add_done_callback(self._reaping.discard)

    async def reap(self, session: str) -> bool:
        """Release an idle session (turn, adapter, chat state, attachments).

        False (nothing done) when the session listens again.
        """
        if self.bridge.has_listener(session):
            return False
        chat = self._chats.get(session)
        if chat is not None:
            await self.cancel(session)
            if self.bridge.has_listener(session):  # came back while cancelling
                return False
            self._chats.pop(session, None)
            adapter, chat.adapter = chat.adapter, None
            if adapter is not None:
                with contextlib.suppress(Exception):
                    await adapter.close()
        _attachments.registry(self.bridge).drop(session)
        return True

    async def close(self) -> None:
        for timer in self._idle.values():
            timer.cancel()
        self._idle.clear()
        reaping = list(self._reaping)
        for task in reaping:
            task.cancel()
        await asyncio.gather(*reaping, return_exceptions=True)
        with contextlib.suppress(ValueError):
            self.bridge.listener_hooks.remove(self._on_listening)
        for session, chat in list(self._chats.items()):
            await self.cancel(session)
            adapter = chat.adapter
            if adapter is not None:
                with contextlib.suppress(Exception):
                    await adapter.close()
        self._chats.clear()
