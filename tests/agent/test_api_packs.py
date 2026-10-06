"""Direct API packs (api-anthropic, api-openai) over httpx.MockTransport: no network."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from test_chat import Studio

from dtk_engine.agent.packs import api_chat
from dtk_engine.agent.packs.chat_packs import hub_from_env
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio
KEY = "sk-secret-key-123"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DTK_UI_TOKEN", "t")
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    for name in (
        "DTK_AGENT_PACK", "DTK_AGENT_MAX_TOKENS", "DTK_AGENT_MAX_TURNS", "ANTHROPIC_API_KEY",
        "DTK_ANTHROPIC_BASE_URL", "DTK_ANTHROPIC_MODEL", "DTK_OPENAI_BASE_URL",
        "DTK_OPENAI_API_KEY", "DTK_OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def sse(*events: dict | str) -> httpx.Response:
    body = "".join(
        f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events
    )
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


class Server:
    """A scripted provider: ``chats`` answers POSTs in order; ``models`` answers the GET."""

    def __init__(self, chats, models=None):
        self.chats, self.models = list(chats), models
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            return self.models if self.models is not None else httpx.Response(404)
        item = self.chats.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


@pytest.fixture
def serve(monkeypatch):
    def install(server: Server) -> Server:
        monkeypatch.setattr(api_chat, "make_transport", lambda: httpx.MockTransport(server))
        return server

    return install


def anthropic_text(text: str, out: int = 5) -> httpx.Response:
    return sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "cache_read_input_tokens": 4, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:2]}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[2:]}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": out}},
        {"type": "message_stop"},
    )


def anthropic_tool(name="list_keys", args=None, tid="toolu_1") -> httpx.Response:
    return sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Looking."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": tid, "name": name, "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": json.dumps(args or {})}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    )


def openai_text(text: str) -> httpx.Response:
    return sse(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text[:2]}}]},
        {"choices": [{"index": 0, "delta": {"content": text[2:]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                                  "prompt_tokens_details": {"cached_tokens": 8}}},
        "[DONE]",
    )


def openai_tool(name="list_keys", args=None, tid="call_1") -> httpx.Response:
    raw = json.dumps(args or {})
    return sse(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": tid, "type": "function", "function": {"name": name, "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[:1]}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[1:]}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 9}},
        "[DONE]",
    )


ANTHROPIC_MODELS = httpx.Response(200, json={"data": [
    {"id": "claude-opus-5-5", "display_name": "Opus 5.5"},
    {"id": "claude-sonnet-5-5", "display_name": "Sonnet 5.5"},
]})
OPENAI_MODELS = httpx.Response(200, json={"data": [{"id": "llama3"}, {"id": "qwen"}]})


def setup_pack(monkeypatch, pack_id: str, *, model: str | None = "m1"):
    if pack_id == "api-anthropic":
        monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
        if model:
            monkeypatch.setenv("DTK_ANTHROPIC_MODEL", model)
    else:
        monkeypatch.setenv("DTK_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("DTK_OPENAI_API_KEY", KEY)
        if model:
            monkeypatch.setenv("DTK_OPENAI_MODEL", model)
    monkeypatch.setenv("DTK_AGENT_PACK", pack_id)


async def run(pack_id: str, text="hi", *, hub_setup=None):
    bridge = UiBridge("t")
    hub = hub_from_env(bridge)
    studio = Studio(bridge)
    hub.send("s1", text)
    events = await studio.until_done()
    return hub, studio, events


def types(events):
    return [e["type"] for e in events]


@pytest.mark.parametrize("pack_id", ["api-anthropic", "api-openai"])
async def test_streaming_text(monkeypatch, serve, pack_id):
    setup_pack(monkeypatch, pack_id)
    reply = anthropic_text("hello") if pack_id == "api-anthropic" else openai_text("hello")
    server = serve(Server([reply]))
    _, _, events = await run(pack_id)
    deltas = [e["text"] for e in events if e["type"] == "assistant_delta"]
    assert "".join(deltas) == "hello" and len(deltas) == 2
    usage = next(e for e in events if e["type"] == "usage")
    assert (usage["input_tokens"], usage["output_tokens"]) == (
        (14, 5) if pack_id == "api-anthropic" else (12, 3)
    )
    # #151: the split behind input_tokens, and the per-call context size
    split = ("uncached_input_tokens", "cache_read_input_tokens", "context_tokens")
    assert tuple(usage[k] for k in split) == (
        (10, 4, 14) if pack_id == "api-anthropic" else (4, 8, 12)
    )
    assert events[-1] == {**events[-1], "type": "done", "stop_reason": "end_turn"}
    req = server.requests[0]
    body = server.bodies()[0]
    assert body["stream"] is True and body["model"] == "m1"
    if pack_id == "api-anthropic":
        assert str(req.url) == "https://api.anthropic.com/v1/messages"
        assert req.headers["x-api-key"] == KEY and req.headers["anthropic-version"] == "2023-06-01"
        assert body["system"] and body["tools"][0]["input_schema"]
    else:
        assert str(req.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert req.headers["authorization"] == f"Bearer {KEY}"
        assert body["stream_options"] == {"include_usage": True}
        assert body["messages"][0]["role"] == "system"
        assert body["tools"][0]["type"] == "function"


@pytest.mark.parametrize("pack_id", ["api-anthropic", "api-openai"])
async def test_tool_round_trip_through_dtk_server(monkeypatch, serve, pack_id):
    setup_pack(monkeypatch, pack_id)
    first, last = (
        (anthropic_tool(), anthropic_text("done")) if pack_id == "api-anthropic"
        else (openai_tool(), openai_text("done"))
    )
    server = serve(Server([first, last]))
    _, _, events = await run(pack_id)
    call = next(e for e in events if e["type"] == "tool_call")
    result = next(e for e in events if e["type"] == "tool_result")
    assert call["name"] == "list_keys" and result["id"] == call["id"] and result["ok"] is True
    assert result["summary"]
    second = server.bodies()[1]["messages"]
    if pack_id == "api-anthropic":
        assert second[-1]["content"][0]["type"] == "tool_result"
        assert second[-2]["content"][-1]["type"] == "tool_use"
    else:
        assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "call_1"
        assert second[-2]["tool_calls"][0]["function"]["name"] == "list_keys"
    usage = next(e for e in events if e["type"] == "usage")
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0


async def test_session_is_pinned_for_session_tools(monkeypatch, serve):
    setup_pack(monkeypatch, "api-anthropic")
    serve(Server([anthropic_tool("get_ui_context", {"session": "evil"}), anthropic_text("ok")]))
    seen = []
    real = api_chat.Client.call_tool

    async def spy(self, name, arguments=None, **kwargs):
        seen.append((name, arguments))
        return await real(self, name, arguments, **kwargs)

    monkeypatch.setattr(api_chat.Client, "call_tool", spy)
    hub, _, events = await run("api-anthropic")
    assert "get_ui_context" in hub.session_tools
    assert seen == [("get_ui_context", {"session": "s1"})]
    assert next(e for e in events if e["type"] == "tool_call")["input"] == {"session": "evil"}


async def test_usage_cap_stops_the_loop(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    monkeypatch.setenv("DTK_AGENT_MAX_TOKENS", "20")
    server = serve(Server([openai_tool(), openai_text("never")]))
    _, _, events = await run("api-openai")
    assert len(server.bodies()) == 1
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "max_tokens"
    assert events[-1]["stop_reason"] == "max_tokens"


async def test_max_turns(monkeypatch, serve):
    setup_pack(monkeypatch, "api-anthropic")
    monkeypatch.setenv("DTK_AGENT_MAX_TURNS", "2")
    server = serve(Server([anthropic_tool(tid=f"t{i}") for i in range(5)]))
    _, _, events = await run("api-anthropic")
    assert len(server.bodies()) == 3  # 2 round trips, the 3rd response's call is refused
    assert types(events).count("tool_result") == 2
    assert events[-1]["stop_reason"] == "max_turns"


@pytest.mark.parametrize(
    ("pack_id", "body", "code"),
    [
        ("api-anthropic", {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}, "overloaded_error"),
        ("api-openai", {"error": {"message": "bad key", "type": "invalid_request_error", "code": "x"}}, "invalid_request_error"),
        ("api-openai", {"nope": 1}, "http_500"),
    ],
)
async def test_http_error(monkeypatch, serve, pack_id, body, code):
    setup_pack(monkeypatch, pack_id)
    serve(Server([httpx.Response(500, json=body)]))
    _, _, events = await run(pack_id)
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == code and err["message"]
    assert events[-1]["stop_reason"] == "error"


async def test_error_mentioning_the_key_is_scrubbed(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    serve(Server([httpx.Response(401, json={"error": {"message": f"bad {KEY}", "type": "auth"}})]))
    _, _, events = await run("api-openai")
    assert KEY not in json.dumps(events)


async def test_network_error(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    serve(Server([httpx.ConnectError("refused")]))
    _, _, events = await run("api-openai")
    err = next(e for e in events if e["type"] == "error")
    assert err["code"] == "network" and "127.0.0.1:11434" in err["message"]
    assert events[-1]["stop_reason"] == "error"


async def test_mid_stream_error_event(monkeypatch, serve):
    setup_pack(monkeypatch, "api-anthropic")
    serve(Server([sse({"type": "error", "error": {"type": "overloaded_error", "message": "later"}})]))
    _, _, events = await run("api-anthropic")
    assert next(e for e in events if e["type"] == "error")["code"] == "overloaded_error"


async def test_cancel_in_flight(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    started = asyncio.Event()

    async def slow(request):
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(api_chat, "make_transport", lambda: httpx.MockTransport(slow))
    bridge = UiBridge("t")
    studio = Studio(bridge)
    hub = hub_from_env(bridge)
    hub.send("s1", "hi")
    await asyncio.wait_for(started.wait(), 5)
    assert await hub.cancel("s1") is True
    events = await studio.until_done()
    assert events[-1]["stop_reason"] == "cancelled"


async def test_history_kept_across_turns_and_model_switch(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    server = serve(Server([openai_text("one"), openai_text("two")]))
    bridge = UiBridge("t")
    studio = Studio(bridge)
    hub = hub_from_env(bridge)
    hub.send("s1", "first")
    await studio.until_done()
    await hub.configure("s1", "api-openai", None)  # same pack, same model: kept
    adapter = hub._chats["s1"].adapter
    await adapter.set_model("m2")
    hub.send("s1", "second")
    await studio.until_done()
    second = server.bodies()[1]
    assert second["model"] == "m2"
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]


async def test_failed_turn_rolls_history_back(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    server = serve(Server([httpx.Response(500, json={}), openai_text("ok")]))
    bridge = UiBridge("t")
    studio = Studio(bridge)
    hub = hub_from_env(bridge)
    hub.send("s1", "first")
    await studio.until_done()
    hub.send("s1", "again")
    await studio.until_done()
    assert [m["role"] for m in server.bodies()[1]["messages"]] == ["system", "user"]


async def test_discovery_and_default_selection(monkeypatch, serve):
    setup_pack(monkeypatch, "api-anthropic", model=None)
    server = serve(Server([anthropic_text("hi")], models=ANTHROPIC_MODELS))
    listing = await api_chat.discover(api_chat.AnthropicProvider())
    assert listing.ids() == ["claude-opus-5-5", "claude-sonnet-5-5"]
    assert listing.models[1]["label"] == "Sonnet 5.5"
    get = server.requests[0]
    assert str(get.url).startswith("https://api.anthropic.com/v1/models") and get.headers["x-api-key"] == KEY
    await run("api-anthropic")
    assert server.bodies()[0]["model"] == "claude-sonnet-5-5"

    setup_pack(monkeypatch, "api-openai", model=None)
    server = serve(Server([openai_text("hi")], models=OPENAI_MODELS))
    await run("api-openai")
    assert server.bodies()[0]["model"] == "llama3"
    assert str(server.requests[0].url) == "http://127.0.0.1:11434/v1/models"


async def test_discovery_failure_and_no_default(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai", model=None)
    serve(Server([], models=httpx.Response(401, json={"error": {"message": "no", "type": "auth"}})))
    listing = await api_chat.discover(api_chat.OpenAIProvider())
    assert listing.free_text and "auth" in listing.error
    _, _, events = await run("api-openai")
    assert next(e for e in events if e["type"] == "error")["code"] == "no_model"


async def test_options_availability_by_env(monkeypatch, serve):
    monkeypatch.setenv("DTK_AGENT_PACK", "api-openai")
    monkeypatch.setenv("DTK_OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    serve(Server([], models=OPENAI_MODELS))
    options = await hub_from_env(UiBridge("t")).options()
    by_id = {p["id"]: p for p in options["packs"]}
    assert by_id["api-openai"]["available"] is True
    assert by_id["api-openai"]["mode"] == "api" and by_id["api-openai"]["panel"] == "chat"
    assert by_id["api-openai"]["provider"] == "OpenAI-compatible (127.0.0.1:11434)"
    assert [m["id"] for m in by_id["api-openai"]["models"]] == ["llama3", "qwen"]
    assert by_id["api-anthropic"]["available"] is False
    assert by_id["api-anthropic"]["reason"] == "ANTHROPIC_API_KEY not set"
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.delenv("DTK_OPENAI_BASE_URL")
    monkeypatch.setenv("DTK_AGENT_PACK", "api-anthropic")
    serve(Server([], models=ANTHROPIC_MODELS))
    options = await hub_from_env(UiBridge("t")).options()
    by_id = {p["id"]: p for p in options["packs"]}
    assert by_id["api-anthropic"]["available"] is True
    assert by_id["api-anthropic"]["provider"] == "Anthropic API (api.anthropic.com)"
    assert by_id["api-openai"]["reason"] == "DTK_OPENAI_BASE_URL not set"


async def test_key_never_leaks(monkeypatch, serve):
    setup_pack(monkeypatch, "api-anthropic")
    serve(Server([anthropic_tool(), anthropic_text("done")], models=ANTHROPIC_MODELS))
    hub, _, events = await run("api-anthropic")
    blob = json.dumps([events, hub.status("s1"), await hub.options()])
    assert KEY not in blob
    pack = hub.packs["api-anthropic"]
    assert KEY not in repr(pack) and KEY not in repr(hub._chats["s1"].adapter.__dict__)


async def chat_turns(texts):
    bridge = UiBridge("t")
    studio = Studio(bridge)
    hub = hub_from_env(bridge)
    events = []
    for text in texts:
        hub.send("s1", text)
        events.append(await studio.until_done())
    return hub, events


def tool_ids(messages, pack_id):
    """(tool call ids, tool result ids) of a request's messages."""
    if pack_id == "api-anthropic":
        blocks = [b for m in messages if isinstance(m["content"], list) for b in m["content"]]
        return (
            [b["id"] for b in blocks if b["type"] == "tool_use"],
            [b["tool_use_id"] for b in blocks if b["type"] == "tool_result"],
        )
    return (
        [c["id"] for m in messages for c in m.get("tool_calls") or []],
        [m["tool_call_id"] for m in messages if m["role"] == "tool"],
    )


def results_text(messages, pack_id):
    if pack_id == "api-anthropic":
        return [
            b["content"] for m in messages if isinstance(m["content"], list)
            for b in m["content"] if b["type"] == "tool_result"
        ]
    return [m["content"] for m in messages if m["role"] == "tool"]


@pytest.mark.parametrize("pack_id", ["api-anthropic", "api-openai"])
async def test_old_tool_results_elided_ids_still_pair(monkeypatch, serve, pack_id):
    setup_pack(monkeypatch, pack_id)
    turns = api_chat.KEEP_TURNS + 3
    tool, text = (anthropic_tool, anthropic_text) if pack_id == "api-anthropic" else (openai_tool, openai_text)
    chats = [r for i in range(turns) for r in (tool(tid=f"id{i}"), text(f"done {i}"))]
    server = serve(Server(chats))
    await chat_turns([f"turn {i}" for i in range(turns)])
    last = server.bodies()[-1]["messages"]
    calls, results = tool_ids(last, pack_id)
    assert calls == results == [f"id{i}" for i in range(turns)]
    texts = results_text(last, pack_id)
    old = turns - api_chat.KEEP_TURNS
    assert texts[:old] == [api_chat.ELIDED] * old
    assert all(t != api_chat.ELIDED for t in texts[old:])  # newest turns intact


async def test_history_size_stays_bounded(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    monkeypatch.setattr(api_chat, "HISTORY_CHARS", 3000)
    server = serve(Server([openai_text("ok") for _ in range(20)]))
    await chat_turns([f"{i} " + "x" * 500 for i in range(20)])
    sizes = [len(json.dumps(b["messages"][1:])) for b in server.bodies()]
    assert max(sizes) <= 3000
    newest = server.bodies()[-1]["messages"]
    assert newest[1]["role"] == "user"  # starts on a whole turn
    assert newest[-1] == {"role": "user", "content": "19 " + "x" * 500}


@pytest.mark.parametrize(
    ("pack_id", "error"),
    [
        ("api-anthropic", {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 300000 tokens > 200000 maximum"}}),
        ("api-openai", {"error": {"message": "too many tokens", "type": "invalid_request_error", "code": "context_length_exceeded"}}),
    ],
)
async def test_context_too_long_drops_old_turns_and_retries_once(monkeypatch, serve, pack_id, error):
    setup_pack(monkeypatch, pack_id)
    text = anthropic_text if pack_id == "api-anthropic" else openai_text
    chats = [text(f"a{i}") for i in range(4)] + [httpx.Response(400, json=error), text("fits")]
    server = serve(Server(chats))
    _, events = await chat_turns([f"t{i}" for i in range(5)])
    assert events[-1][-1]["stop_reason"] == "end_turn"
    failed, retried = server.bodies()[-2:]
    assert failed["messages"] != retried["messages"]
    users = [m["content"] for m in retried["messages"] if m["role"] == "user"]
    assert users == ["t2", "t3", "t4"]  # the oldest half (2) of the 5 turns is gone


async def test_context_too_long_with_one_turn_is_an_error(monkeypatch, serve):
    setup_pack(monkeypatch, "api-openai")
    error = {"error": {"message": "x", "type": "invalid_request_error", "code": "context_length_exceeded"}}
    server = serve(Server([httpx.Response(400, json=error)]))
    _, _, events = await run("api-openai")
    assert next(e for e in events if e["type"] == "error")["code"] == "invalid_request_error"
    assert len(server.bodies()) == 1
