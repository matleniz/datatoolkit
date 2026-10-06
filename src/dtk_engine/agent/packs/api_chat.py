"""Direct API chat packs: ``api-anthropic`` and ``api-openai`` over ``httpx``.

No CLI, no vendor SDK: one streaming HTTP request per model response. A shared
tool loop (``ApiAdapter``) drives the in-process dtk MCP server exactly like
``agent-sdk`` (same system prompt, ``session`` pinned to the sending tab,
``DTK_AGENT_MAX_TURNS`` tool round trips, token cap, cancel); a small
``Provider`` per API builds the request and decodes the SSE stream.

History is bounded (``ApiAdapter._compact``, before every request): the latest
``KEEP_TURNS`` user turns stay whole, older tool results become a placeholder
(ids kept, so tool_use / tool_result pairs stay valid), and the oldest whole
turns are dropped while the history is over ``HISTORY_CHARS``. A provider
"context too long" answer drops the oldest half of the turns and retries once.

Credentials come from the environment only and are never put in an event, a
status, an option, a log or an exception message (``_scrub`` also masks them in
provider error text).

Env: ``ANTHROPIC_API_KEY`` (required), ``DTK_ANTHROPIC_BASE_URL`` (default
``https://api.anthropic.com``), ``DTK_ANTHROPIC_MODEL``; ``DTK_OPENAI_BASE_URL``
(required), ``DTK_OPENAI_API_KEY`` (optional), ``DTK_OPENAI_MODEL``.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx
from mcp import Client

from dtk_engine.agent.chat import ChatSession, Pack
from dtk_engine.agent.models import DISCOVERY_TIMEOUT, ModelList, failed
from dtk_engine.agent.packs.agent_sdk import SYSTEM_PROMPT, _max_turns
from dtk_engine.agent.packs.chat_packs import parse_tool_text, tool_result_fields

ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_OUTPUT = 8192
KEEP_TURNS = 4  # latest user turns sent whole
HISTORY_CHARS = 400_000  # JSON size of the history above which the oldest turns go
ELIDED = "[older tool result elided]"
_TOO_LONG = ("context_length_exceeded", "prompt is too long", "context length", "maximum context")
TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)


def make_transport() -> httpx.AsyncBaseTransport | None:
    """The HTTP transport (None = httpx's own); tests replace this with a mock."""
    return None


def _client(timeout: httpx.Timeout | float = TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=make_transport(), timeout=timeout)


class ApiError(Exception):
    """A provider failure ending the turn: ``code`` is the event's error code,
    ``detail`` the provider's finer code when it has one (OpenAI ``error.code``)."""

    def __init__(self, message: str, code: str, detail: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _host(base: str) -> str:
    parsed = urlparse(base)
    host = parsed.hostname or base
    return f"{host}:{parsed.port}" if parsed.port else host


# -- stream state -------------------------------------------------------------
@dataclass
class Round:
    """One streamed model response, filled by ``Provider.feed``."""

    text: str = ""
    calls: list[dict] = field(default_factory=list)  # {id, name, args (raw JSON text)}
    input_tokens: int = 0  # cache writes and reads included
    output_tokens: int = 0
    cache_write: int = 0
    cache_read: int = 0
    stop: str | None = None


def _json_args(raw: str) -> dict | None:
    """Tool arguments from the model's JSON text (None = not a JSON object)."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _error_of(status: int, body: bytes) -> ApiError:
    """The ``ApiError`` of an HTTP error answer (Anthropic and OpenAI error bodies)."""
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        err = None
    if not isinstance(err, dict):
        err = {}
    code = err.get("type") or err.get("code") or f"http_{status}"
    return ApiError(str(err.get("message") or f"HTTP {status}"), str(code), str(err.get("code") or ""))


# -- providers ----------------------------------------------------------------
class Provider:
    """What differs between the APIs: URLs, headers, request and stream decoding."""

    id = ""
    title = ""
    label = ""
    default_model_env = ""

    def base(self) -> str:
        raise NotImplementedError

    def key(self) -> str:
        raise NotImplementedError

    def detect(self) -> str | None:
        raise NotImplementedError

    def headers(self) -> dict[str, str]:
        raise NotImplementedError

    def chat_url(self) -> str:
        raise NotImplementedError

    def models_url(self) -> str:
        raise NotImplementedError

    def request(self, model: str, history: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError

    def feed(self, round_: Round, event: dict) -> str:
        """Fold one SSE ``data`` object into ``round_``; the text delta (or "")."""
        raise NotImplementedError

    def assistant(self, round_: Round) -> dict:
        raise NotImplementedError

    def tool_results(self, results: list[tuple[str, str, bool]]) -> list[dict]:
        """History entries for ``[(call id, text, is_error)]``."""
        raise NotImplementedError

    def user(self, text: str) -> dict:
        return {"role": "user", "content": text}

    def pick_default(self, ids: list[str]) -> str:
        return ids[0]

    def env_model(self) -> str | None:
        return _env(self.default_model_env) or None

    def provider_label(self) -> str | None:
        base = self.base()
        return f"{self.label} ({_host(base)})" if base else None

    def parse_models(self, payload: dict) -> list[dict]:
        return [
            {"id": item["id"], "label": item.get("display_name") or item["id"]}
            for item in payload.get("data") or []
            if isinstance(item, dict) and item.get("id")
        ]


class AnthropicProvider(Provider):
    id = "api-anthropic"
    title = "Claude (Anthropic API, ANTHROPIC_API_KEY)"
    label = "Anthropic API"
    default_model_env = "DTK_ANTHROPIC_MODEL"

    def base(self) -> str:
        return (_env("DTK_ANTHROPIC_BASE_URL") or ANTHROPIC_DEFAULT_BASE).rstrip("/")

    def key(self) -> str:
        return _env("ANTHROPIC_API_KEY")

    def detect(self) -> str | None:
        return None if self.key() else "ANTHROPIC_API_KEY not set"

    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.key(), "anthropic-version": ANTHROPIC_VERSION}

    def chat_url(self) -> str:
        return f"{self.base()}/v1/messages"

    def models_url(self) -> str:
        return f"{self.base()}/v1/models?limit=1000"

    def pick_default(self, ids: list[str]) -> str:
        return next((i for i in ids if "sonnet" in i), ids[0])

    def request(self, model: str, history: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": ANTHROPIC_MAX_OUTPUT,
            "stream": True,
            "system": SYSTEM_PROMPT,
            "messages": history,
        }
        if tools:
            body["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
                for t in tools
            ]
        return body

    def feed(self, round_: Round, event: dict) -> str:
        kind = event.get("type")
        if kind == "error":
            err = event.get("error") or {}
            raise ApiError(str(err.get("message") or "stream error"), str(err.get("type") or "error"))
        if kind == "message_start":
            self._usage(round_, (event.get("message") or {}).get("usage"))
        elif kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                round_.calls.append({"id": block.get("id", ""), "name": block.get("name", ""), "args": ""})
        elif kind == "content_block_delta":
            return self._delta(round_, event.get("delta") or {})
        elif kind == "message_delta":
            self._usage(round_, event.get("usage"))
            round_.stop = (event.get("delta") or {}).get("stop_reason") or round_.stop
        return ""

    @staticmethod
    def _delta(round_: Round, delta: dict) -> str:
        if delta.get("type") == "text_delta":
            text = delta.get("text", "")
            round_.text += text
            return text
        if delta.get("type") == "input_json_delta" and round_.calls:
            round_.calls[-1]["args"] += delta.get("partial_json", "")
        return ""

    @staticmethod
    def _usage(round_: Round, usage: dict | None) -> None:
        """Input counts cache reads and writes (as agent-sdk); output is cumulative."""
        if not usage:
            return
        keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        if any(k in usage for k in keys):
            round_.input_tokens = sum(int(usage.get(k) or 0) for k in keys)
            round_.cache_write = int(usage.get("cache_creation_input_tokens") or 0)
            round_.cache_read = int(usage.get("cache_read_input_tokens") or 0)
        if "output_tokens" in usage:
            round_.output_tokens = int(usage.get("output_tokens") or 0)

    def assistant(self, round_: Round) -> dict:
        content: list[dict] = []
        if round_.text:
            content.append({"type": "text", "text": round_.text})
        for call in round_.calls:
            args = _json_args(call["args"]) or {}
            content.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": args})
        return {"role": "assistant", "content": content}

    def tool_results(self, results: list[tuple[str, str, bool]]) -> list[dict]:
        blocks = [
            {"type": "tool_result", "tool_use_id": cid, "content": text, "is_error": err}
            for cid, text, err in results
        ]
        return [{"role": "user", "content": blocks}]


class OpenAIProvider(Provider):
    id = "api-openai"
    title = "OpenAI-compatible API (DTK_OPENAI_BASE_URL)"
    label = "OpenAI-compatible"
    default_model_env = "DTK_OPENAI_MODEL"

    def base(self) -> str:
        return _env("DTK_OPENAI_BASE_URL").rstrip("/")

    def key(self) -> str:
        return _env("DTK_OPENAI_API_KEY")

    def detect(self) -> str | None:
        return None if self.base() else "DTK_OPENAI_BASE_URL not set"

    def headers(self) -> dict[str, str]:
        key = self.key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def chat_url(self) -> str:
        return f"{self.base()}/chat/completions"

    def models_url(self) -> str:
        return f"{self.base()}/models"

    def request(self, model: str, history: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history],
        }
        if tools:
            body["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t["description"], "parameters": t["schema"],
                }}
                for t in tools
            ]
        return body

    def feed(self, round_: Round, event: dict) -> str:
        if isinstance(event.get("error"), dict):
            err = event["error"]
            raise ApiError(str(err.get("message") or "stream error"), str(err.get("type") or "error"))
        usage = event.get("usage")
        if usage:  # a server ignoring include_usage never sends it: zeros
            round_.input_tokens = int(usage.get("prompt_tokens") or 0)
            round_.output_tokens = int(usage.get("completion_tokens") or 0)
            details = usage.get("prompt_tokens_details") or {}
            round_.cache_read = int(details.get("cached_tokens") or 0)
        text = ""
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text += delta["content"]
            for call in delta.get("tool_calls") or []:
                self._call_delta(round_, call)
            round_.stop = choice.get("finish_reason") or round_.stop
        round_.text += text
        return text

    @staticmethod
    def _call_delta(round_: Round, delta: dict) -> None:
        """Assemble streamed ``tool_calls`` by ``index`` (id / name arrive once, args in pieces)."""
        index = int(delta.get("index") or 0)
        while len(round_.calls) <= index:
            round_.calls.append({"id": "", "name": "", "args": ""})
        call = round_.calls[index]
        fn = delta.get("function") or {}
        call["id"] = delta.get("id") or call["id"]
        call["name"] = fn.get("name") or call["name"]
        call["args"] += fn.get("arguments") or ""

    def assistant(self, round_: Round) -> dict:
        message: dict[str, Any] = {"role": "assistant", "content": round_.text or None}
        if round_.calls:
            message["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for c in round_.calls
            ]
        return message

    def tool_results(self, results: list[tuple[str, str, bool]]) -> list[dict]:
        return [{"role": "tool", "tool_call_id": cid, "content": text} for cid, text, _ in results]


# -- history bounds -----------------------------------------------------------


def _is_turn_start(message: dict) -> bool:
    """A user's own message (not an Anthropic tool_result carrier)."""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return not (
        isinstance(content, list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _turn_starts(history: list[dict]) -> list[int]:
    return [i for i, m in enumerate(history) if _is_turn_start(m)]


def _elide(message: dict) -> dict:
    """``message`` with its tool result text replaced by ``ELIDED`` (ids kept)."""
    if message.get("role") == "tool":  # OpenAI
        return {**message, "content": ELIDED}
    content = message.get("content")
    if message.get("role") == "user" and isinstance(content, list):  # Anthropic
        return {**message, "content": [
            {**b, "content": ELIDED} if b.get("type") == "tool_result" else b for b in content
        ]}
    return message


def _too_long(exc: ApiError) -> bool:
    text = f"{exc.code} {exc.detail} {exc}".lower()
    return any(marker in text for marker in _TOO_LONG)


# -- SSE ----------------------------------------------------------------------
async def _sse_events(response: httpx.Response) -> AsyncIterator[dict]:
    """The JSON ``data:`` objects of an SSE response (``[DONE]`` and junk skipped)."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict):
            yield event


# -- model discovery ----------------------------------------------------------
async def discover(provider: Provider) -> ModelList:
    """``GET`` the provider's model list; a failure is free text with an error."""
    try:
        async with _client(DISCOVERY_TIMEOUT) as client:
            response = await client.get(provider.models_url(), headers=provider.headers())
    except httpx.HTTPError as exc:
        return failed(_scrub(provider, f"model discovery failed: {type(exc).__name__}"))
    if response.status_code >= 400:
        err = _error_of(response.status_code, response.content)
        return failed(_scrub(provider, f"model discovery failed: {err.code}: {err}"))
    try:
        entries = provider.parse_models(response.json())
    except (ValueError, AttributeError):
        entries = []
    return ModelList(entries) if entries else failed("the endpoint listed no models")


def _scrub(provider: Provider, text: str) -> str:
    key = provider.key()
    return text.replace(key, "***") if key else text


# -- the adapter --------------------------------------------------------------
class ApiAdapter:
    """One conversation (history kept here) against one provider."""

    def __init__(self, provider: Provider, model: str | None = None) -> None:
        self.provider = provider
        self.model = model or provider.env_model()
        self.chat: ChatSession | None = None
        self.history: list[dict] = []
        self._tools: list[dict] | None = None
        self._work: asyncio.Task | None = None
        self._cancelled = False
        self._mark = 0  # where the running turn starts in ``history``

    async def start(self, chat: ChatSession) -> None:
        self.chat = chat

    async def set_model(self, model: str | None) -> None:
        self.model = model or self.provider.env_model()

    async def cancel(self) -> None:
        if self._work is not None and not self._work.done():
            self._cancelled = True
            self._work.cancel()

    async def close(self) -> None:
        await self.cancel()
        self.chat = None

    async def send(self, text: str) -> str | None:
        self._cancelled = False
        self._mark = len(self.history)  # moved back when old turns are dropped
        self._work = asyncio.ensure_future(self._turn(text))
        try:
            stop = await self._work
        except asyncio.CancelledError:
            if not self._cancelled:
                raise
            stop = "cancelled"
        if stop in ("error", "cancelled"):
            del self.history[self._mark:]  # keep the history well-formed for the next turn
        return stop

    async def _turn(self, text: str) -> str:
        assert self.chat is not None
        chat = self.chat
        async with Client(chat.mcp_server()) as mcp:
            try:  # inside the client: its task group would wrap an escaping ApiError
                await self._load_tools(mcp)
                return await self._loop(chat, mcp, text)
            except ApiError as exc:
                chat.emit("error", message=_scrub(self.provider, str(exc)), code=exc.code)
                return "error"

    async def _load_tools(self, mcp: Client) -> None:
        if self._tools is None:
            listed = await mcp.list_tools()
            self._tools = [
                {"name": t.name, "description": t.description or "", "schema": t.input_schema}
                for t in listed.tools
            ]

    async def _loop(self, chat: ChatSession, mcp: Client, text: str) -> str:
        model = await self._resolve_model()
        self.history.append(self.provider.user(text))
        used_in = used_out = write = read = rounds = 0
        while True:
            round_ = await self._respond_bounded(chat, model)
            used_in += round_.input_tokens
            used_out += round_.output_tokens
            write += round_.cache_write
            read += round_.cache_read
            capped = chat.set_turn_usage(
                used_in, used_out, cache_write=write, cache_read=read,
                context=round_.input_tokens,
            )
            if not round_.calls:
                self.history.append(self.provider.assistant(round_))
                return _stop_reason(round_.stop)
            if capped:
                return "max_tokens"
            if rounds >= _max_turns():
                return "max_turns"
            self.history.append(self.provider.assistant(round_))
            results = [await self._run_tool(chat, mcp, call) for call in round_.calls]
            self.history.extend(self.provider.tool_results(results))
            rounds += 1

    async def _resolve_model(self) -> str:
        if self.model:
            return self.model
        listing = await discover(self.provider)
        if not listing.models:
            raise ApiError(listing.error or "no model available", "no_model")
        self.model = self.provider.pick_default(listing.ids())
        return self.model

    def _drop_oldest_turns(self, count: int) -> bool:
        """Drop the ``count`` oldest whole turns (never the newest); False if none."""
        starts = _turn_starts(self.history)
        cut = starts[min(count, len(starts) - 1)] if starts else 0
        if cut <= 0:
            return False
        del self.history[:cut]
        self._mark = max(0, self._mark - cut)
        return True

    def _compact(self) -> None:
        starts = _turn_starts(self.history)
        if len(starts) > KEEP_TURNS:
            cut = starts[-KEEP_TURNS]
            self.history[:cut] = [_elide(m) for m in self.history[:cut]]
        while len(json.dumps(self.history)) > HISTORY_CHARS and self._drop_oldest_turns(1):
            pass

    async def _respond_bounded(self, chat: ChatSession, model: str) -> Round:
        """``_respond`` on the compacted history; on "context too long", drop the
        oldest half of the turns and retry once."""
        self._compact()
        try:
            return await self._respond(chat, model)
        except ApiError as exc:
            half = max(1, len(_turn_starts(self.history)) // 2)
            if not _too_long(exc) or not self._drop_oldest_turns(half):
                raise
        return await self._respond(chat, model)

    async def _respond(self, chat: ChatSession, model: str) -> Round:
        round_ = Round()
        body = self.provider.request(model, self.history, self._tools or [])
        try:
            async with _client() as client, client.stream(
                "POST", self.provider.chat_url(), json=body, headers=self.provider.headers(),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _error_of(response.status_code, response.content)
                async for event in _sse_events(response):
                    delta = self.provider.feed(round_, event)
                    if delta:
                        chat.emit("assistant_delta", text=delta)
        except httpx.HTTPError as exc:
            host = _host(self.provider.base())
            raise ApiError(f"cannot reach {host}: {type(exc).__name__}", "network") from None
        return round_

    async def _run_tool(self, chat: ChatSession, mcp: Client, call: dict) -> tuple[str, str, bool]:
        """Run one tool call through the dtk server: ``(call id, text for the model, is_error)``."""
        args = _json_args(call["args"])
        chat.emit("tool_call", id=call["id"], name=call["name"], input=args or {})
        if args is None:
            text, is_error = "tool arguments are not a JSON object", True
            chat.emit("tool_result", id=call["id"], ok=False, error=text)
            return call["id"], text, True
        if call["name"] in chat.session_tools:
            args = {**args, "session": chat.session}
        try:
            result = await mcp.call_tool(call["name"], args)
            text = "".join(getattr(c, "text", "") for c in result.content)
            is_error = bool(result.is_error)
        except Exception as exc:  # noqa: BLE001 - a failing tool is the model's to read
            text, is_error = str(exc) or type(exc).__name__, True
        fields = tool_result_fields(parse_tool_text(text), is_error=is_error)
        chat.emit("tool_result", id=call["id"], **fields)
        return call["id"], text, is_error


def _stop_reason(stop: str | None) -> str:
    return {"stop": "end_turn", "length": "max_tokens", None: "end_turn"}.get(stop, stop or "end_turn")


# -- packs --------------------------------------------------------------------
def _pack(provider: Provider) -> Pack:
    return Pack(
        id=provider.id,
        provider=provider.provider_label,
        model=provider.env_model,
        detect=provider.detect,
        create=lambda model=None: ApiAdapter(provider, model),
        title=provider.title,
        mode="api",
        panel="chat",
        models=lambda: discover(provider),
    )


def anthropic_pack() -> Pack:
    return _pack(AnthropicProvider())


def openai_pack() -> Pack:
    return _pack(OpenAIProvider())
