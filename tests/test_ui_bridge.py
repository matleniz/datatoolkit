"""UI bridge: context store, guard, and the SSE command/ack round trip (real socket)."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from dtk_engine.http import create_app
from dtk_engine.ui_bridge import UiBridge

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
READ_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.delenv("DTK_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("DTK_UI_ALLOWED_HOSTS", raising=False)


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://localhost") as c:
        yield c


class Server:
    def __init__(self, ping_interval: float = 15.0) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.app = create_app(ui_ping_interval=ping_interval)
        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error",
            timeout_graceful_shutdown=1,  # SSE streams outlive the server otherwise
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        assert self.server.started
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "server did not shut down with an open SSE stream"


def _open_stream(srv: Server, session: str = "s1", headers: dict | None = None):
    client = httpx.Client(timeout=READ_TIMEOUT)
    cm = client.stream(
        "GET", f"{srv.url}/api/ui/events", params={"session": session},
        headers=headers if headers is not None else AUTH,
    )
    return client, cm


def _read_until(lines, predicate):
    for line in lines:
        if predicate(line):
            return line
    raise AssertionError("stream ended")


def _wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.02)
    assert cond()


def _post_command(srv, body, results, **params):
    results.append(httpx.post(
        f"{srv.url}/api/ui/commands", json=body, headers=AUTH, params=params, timeout=10
    ))


# -- context ---------------------------------------------------------------


def test_context_round_trip_and_latest(client):
    assert client.get("/api/ui/context", headers=AUTH).status_code == 404
    body = client.get("/api/ui/context", headers=AUTH).json()
    assert set(body) == {"type", "message", "details"}

    ctx = {"session": "a", "screen": "bench", "version": 4, "extra": {"k": 1}}
    assert client.put("/api/ui/context", json=ctx, headers=AUTH).status_code == 204
    assert client.get("/api/ui/context", headers=AUTH).json() == ctx
    client.put("/api/ui/context", json={"session": "a", "screen": "home"}, headers=AUTH)
    assert client.get("/api/ui/context?session=a", headers=AUTH).json()["screen"] == "home"
    client.put("/api/ui/context", json={"session": "b", "screen": "bench"}, headers=AUTH)
    assert client.get("/api/ui/context", headers=AUTH).json()["session"] == "b"
    assert client.get("/api/ui/context?session=a", headers=AUTH).json()["session"] == "a"
    client.put("/api/ui/context", json={"session": "a", "screen": "x"}, headers=AUTH)
    assert client.get("/api/ui/context", headers=AUTH).json()["session"] == "a"
    assert client.get("/api/ui/context?session=zz", headers=AUTH).status_code == 404


def test_sessions_lists_contexts_and_listeners(client):
    assert client.get("/api/ui/sessions", headers=AUTH).json() == []
    ctx = {"session": "a", "workspace": "w", "identity": "w|x|1"}
    client.put("/api/ui/context", json=ctx, headers=AUTH)
    assert client.get("/api/ui/sessions", headers=AUTH).json() == [
        {"session": "a", "listening": False, "workspace": "w", "identity": "w|x|1"}
    ]


def test_sessions_python_api_order_and_listening():
    bridge = UiBridge("t")
    bridge.add_listener("solo")
    bridge.put_context({"session": "b", "workspace": "wb"})
    queue = bridge.add_listener("b")
    bridge.put_context({"session": "a"})
    assert [(s["session"], s["listening"]) for s in bridge.sessions()] == [
        ("b", True), ("a", False), ("solo", True)
    ]
    bridge.remove_listener("b", queue)
    assert bridge.sessions()[0] == {
        "session": "b", "listening": False, "workspace": "wb", "identity": None
    }


def test_context_requires_session(client):
    assert client.put("/api/ui/context", json={"screen": "x"}, headers=AUTH).status_code == 422


# -- commands over a real socket -------------------------------------------


def test_command_sse_ack_round_trip():
    with Server() as srv:
        client, cm = _open_stream(srv)
        with client, cm as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers["cache-control"] == "no-cache"
            assert resp.headers["x-accel-buffering"] == "no"
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            results: list = []
            t = threading.Thread(
                target=_post_command,
                args=(srv, {"type": "open_window", "tool": "dist"}, results),
            )
            t.start()
            assert _read_until(lines, lambda line: line.startswith("event:")) == "event: command"
            cmd = json.loads(next(lines).removeprefix("data: "))
            assert cmd["type"] == "open_window" and cmd["tool"] == "dist" and cmd["id"]
            ack = httpx.post(
                f"{srv.url}/api/ui/ack",
                json={"id": cmd["id"], "ok": True, "identity": "ws|train|v5|ab"},
                headers=AUTH, timeout=5,
            )
            assert ack.status_code == 200
            t.join(timeout=10)
            assert results[0].json() == {"id": cmd["id"], "ok": True, "identity": "ws|train|v5|ab"}


def test_stale_ack_passed_through_and_session_targeting():
    with Server() as srv:
        c1, cm1 = _open_stream(srv, "s1")
        c2, cm2 = _open_stream(srv, "s2")
        with c1, c2, cm1 as r1, cm2 as r2:
            l1, l2 = r1.iter_lines(), r2.iter_lines()
            _read_until(l1, lambda line: line.startswith(": connected"))
            _read_until(l2, lambda line: line.startswith(": connected"))
            results: list = []
            t = threading.Thread(
                target=_post_command, args=(srv, {"type": "x"}, results), kwargs={"session": "s2"}
            )
            t.start()
            cid = json.loads(
                _read_until(l2, lambda line: line.startswith("data:")).removeprefix("data: ")
            )["id"]
            stale = [{"id": "s1", "reason": "removed"}]  # #153: kept through the bridge
            httpx.post(
                f"{srv.url}/api/ui/ack",
                json={"id": cid, "ok": False, "error": "stale: step s1 removed", "stale": stale},
                headers=AUTH, timeout=5,
            )
            t.join(timeout=10)
            assert results[0].json() == {
                "id": cid, "ok": False, "error": "stale: step s1 removed", "stale": stale,
            }


def test_no_studio_when_no_listener():
    with Server() as srv:
        r = httpx.post(
            f"{srv.url}/api/ui/commands", json={"type": "x"}, headers=AUTH, timeout=5
        )
        assert r.json()["ok"] is False and r.json()["error"] == "no_studio"


def test_timeout_cleans_pending():
    with Server() as srv:
        client, cm = _open_stream(srv)
        with client, cm as resp:
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            r = httpx.post(
                f"{srv.url}/api/ui/commands", json={"type": "x"}, headers=AUTH,
                params={"timeout": "0.2"}, timeout=5,
            )
            assert r.json()["error"] == "timeout"
            assert srv.app.state.ui_bridge._pending == {}
            late = httpx.post(
                f"{srv.url}/api/ui/ack", json={"id": r.json()["id"], "ok": True},
                headers=AUTH, timeout=5,
            )
            assert late.status_code == 404


def test_unknown_ack_id_404(client):
    r = client.post("/api/ui/ack", json={"id": "nope", "ok": True}, headers=AUTH)
    assert r.status_code == 404
    assert set(r.json()) == {"type", "message", "details"}


def test_listener_disconnect_resolves_no_studio():
    with Server() as srv:
        bridge = srv.app.state.ui_bridge
        client, cm = _open_stream(srv)
        resp = cm.__enter__()
        lines = resp.iter_lines()
        _read_until(lines, lambda line: line.startswith(": connected"))
        results: list = []
        t = threading.Thread(
            target=_post_command, args=(srv, {"type": "x"}, results), kwargs={"timeout": "20"}
        )
        t.start()
        _wait(lambda: bridge._pending)
        cm.__exit__(None, None, None)
        client.close()
        t.join(timeout=10)
        assert results[0].json()["error"] == "no_studio"
        _wait(lambda: not bridge._listeners)


def test_heartbeat():
    with Server(ping_interval=0.05) as srv:
        client, cm = _open_stream(srv)
        with client, cm as resp:
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            assert _read_until(lines, lambda line: line.startswith(": ping")) == ": ping"


# -- guard -----------------------------------------------------------------


def test_token_required(client):
    assert client.get("/api/ui/context").status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert client.get("/api/ui/context", headers=bad).status_code == 401
    assert client.get("/api/ui/context?token=nope").status_code == 401
    assert client.get("/api/ui/context", headers={"Authorization": TOKEN}).status_code == 401
    # right token: header and query forms both pass the guard (then 404 = no context)
    assert client.get("/api/ui/context", headers=AUTH).status_code == 404
    assert client.get(f"/api/ui/context?token={TOKEN}").status_code == 404
    body = client.get("/api/ui/context").json()
    assert set(body) == {"type", "message", "details"}


def test_every_ui_route_is_guarded(client):
    calls = [
        client.put("/api/ui/context", json={"session": "a"}),
        client.get("/api/ui/context"),
        client.post("/api/ui/ack", json={"id": "x", "ok": True}),
        client.post("/api/ui/commands", json={"type": "x"}),
        client.get("/api/ui/events?session=a"),
        client.get("/api/ui/sessions"),
    ]
    assert [c.status_code for c in calls] == [401] * 6


def test_token_env_override_and_generated(monkeypatch):
    assert create_app().state.ui_bridge.token == TOKEN
    monkeypatch.delenv("DTK_UI_TOKEN")
    one, two = create_app().state.ui_bridge.token, create_app().state.ui_bridge.token
    assert len(one) >= 32 and one != two
    assert UiBridge().token != UiBridge().token


def test_origin_guard(client, monkeypatch):
    def get(origin):
        return client.get("/api/ui/context", headers={**AUTH, "Origin": origin}).status_code

    assert get("https://evil.example") == 403
    assert get("http://localhost:5173") == 404  # Vite dev origin: allowed
    assert get("http://localhost") == 404  # same origin as the request
    monkeypatch.setenv("DTK_CORS_ORIGINS", "https://studio.example")
    assert get("https://studio.example") == 404


def test_host_guard(monkeypatch):
    app = create_app()

    def get(host):
        with TestClient(app, base_url=f"http://{host}") as c:
            return c.get("/api/ui/context", headers=AUTH).status_code

    assert get("evil.example") == 403
    assert get("127.0.0.1:8765") == 404
    assert get("[::1]:8765") == 404
    monkeypatch.setenv("DTK_UI_ALLOWED_HOSTS", "studio.internal, other")
    assert get("studio.internal") == 404


def test_contract_routes_need_no_token(client):
    assert client.get("/api/keys").status_code == 200


def test_send_command_python_api_no_studio():
    result = asyncio.run(UiBridge("t").send_command({"type": "x", "id": "mine"}, timeout=0.1))
    assert result == {"id": "mine", "ok": False, "error": "no_studio"}


def test_server_shutdown_with_open_stream():
    with Server() as srv:
        client, cm = _open_stream(srv)
        with client, cm as resp:
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            srv.server.should_exit = True
            srv.thread.join(timeout=10)
            assert not srv.thread.is_alive()


# -- reviews: interim ack + command status (datatoolkit-issues#95) ---------


async def _send_and_get(bridge: UiBridge, queue: asyncio.Queue, **kwargs):
    task = asyncio.create_task(bridge.send_command({"type": "propose_steps"}, **kwargs))
    cmd = await asyncio.wait_for(queue.get(), 1)
    return task, cmd["id"]


def test_interim_ack_returns_pending_then_final_ack_is_kept():
    async def scenario():
        bridge = UiBridge("t")
        queue = bridge.add_listener("s")
        task, cid = await _send_and_get(bridge, queue, timeout=5)
        assert bridge.ack({"id": cid, "pending": "review"})
        assert await asyncio.wait_for(task, 1) == {"id": cid, "ok": None, "pending": "review"}
        assert bridge.command_status(cid) == {"id": cid, "ok": None, "pending": "review"}
        assert bridge.ack({"id": cid, "pending": "review"})  # repeated interim: harmless
        assert bridge.ack({"id": cid, "ok": True, "identity": "ws|train|v3|ab", "error": None})
        assert bridge.command_status(cid) == {"id": cid, "ok": True, "identity": "ws|train|v3|ab"}
        assert not bridge.ack({"id": cid, "ok": True})  # already final

    asyncio.run(scenario())


def test_review_deadline_reads_as_timeout():
    async def scenario():
        bridge = UiBridge("t", review_timeout=0.05)
        queue = bridge.add_listener("s")
        task, cid = await _send_and_get(bridge, queue, timeout=5)
        bridge.ack({"id": cid, "pending": "review"})
        await task
        await asyncio.sleep(0.1)
        assert bridge.command_status(cid) == {"id": cid, "ok": False, "error": "timeout"}
        assert not bridge.ack({"id": cid, "ok": True})

    asyncio.run(scenario())


def test_listener_drop_during_review_is_no_studio():
    async def scenario():
        bridge = UiBridge("t")
        queue = bridge.add_listener("s")
        task, cid = await _send_and_get(bridge, queue, timeout=5)
        bridge.ack({"id": cid, "pending": "review"})
        await task
        bridge.remove_listener("s", queue)
        assert bridge.command_status(cid) == {"id": cid, "ok": False, "error": "no_studio"}

    asyncio.run(scenario())


def test_status_of_plain_commands_and_unknown():
    async def scenario():
        bridge = UiBridge("t")
        assert bridge.command_status("nope") is None
        queue = bridge.add_listener("s")
        task, cid = await _send_and_get(bridge, queue, timeout=0.05)
        assert (await task)["error"] == "timeout"
        assert bridge.command_status(cid) == {"id": cid, "ok": False, "error": "timeout"}
        task, cid = await _send_and_get(bridge, queue, timeout=5)
        bridge.ack({"id": cid, "ok": False, "error": "stale"})
        await task
        assert bridge.command_status(cid) == {"id": cid, "ok": False, "error": "stale"}

    asyncio.run(scenario())


def test_review_id_cannot_be_reused():
    async def scenario():
        bridge = UiBridge("t")
        queue = bridge.add_listener("s")
        task = asyncio.create_task(bridge.send_command({"type": "x", "id": "r1"}, timeout=5))
        await asyncio.wait_for(queue.get(), 1)
        bridge.ack({"id": "r1", "pending": "review"})
        await task
        again = await bridge.send_command({"type": "x", "id": "r1"}, timeout=0.05)
        assert again == {"id": "r1", "ok": False, "error": "duplicate_id"}

    asyncio.run(scenario())


def test_ack_needs_exactly_one_of_ok_pending(client):
    for body in ({"id": "x"}, {"id": "x", "ok": True, "pending": "review"},
                 {"id": "x", "pending": "later"}):
        assert client.post("/api/ui/ack", json=body, headers=AUTH).status_code == 422


def test_command_status_route_unknown_404_and_guarded(client):
    r = client.get("/api/ui/commands/nope", headers=AUTH)
    assert r.status_code == 404
    assert r.json()["type"] == "UnknownCommand"
    assert client.get("/api/ui/commands/nope").status_code == 401


def test_review_over_http_round_trip():
    with Server() as srv:
        client, cm = _open_stream(srv)
        with client, cm as resp:
            lines = resp.iter_lines()
            _read_until(lines, lambda line: line.startswith(": connected"))
            results: list = []
            t = threading.Thread(
                target=_post_command, args=(srv, {"type": "propose_steps"}, results),
                kwargs={"timeout": "20"},
            )
            t.start()
            cid = json.loads(
                _read_until(lines, lambda line: line.startswith("data:")).removeprefix("data: ")
            )["id"]
            interim = httpx.post(
                f"{srv.url}/api/ui/ack", json={"id": cid, "pending": "review"},
                headers=AUTH, timeout=5,
            )
            assert interim.status_code == 200
            t.join(timeout=10)
            assert results[0].json() == {"id": cid, "ok": None, "pending": "review"}
            final = httpx.post(
                f"{srv.url}/api/ui/ack", json={"id": cid, "ok": True, "identity": "i2"},
                headers=AUTH, timeout=5,
            )
            assert final.status_code == 200
            status = httpx.get(f"{srv.url}/api/ui/commands/{cid}", headers=AUTH, timeout=5)
            assert status.json() == {"id": cid, "ok": True, "identity": "i2"}
