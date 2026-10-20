"""The 413 of the upload cap reaches the client through a proxy that streams the body.

A real uvicorn server plus a tiny asyncio forwarding proxy (no buffering, like a Vite or
nginx ``proxy_request_buffering off`` hop): it pipes the whole request to the engine and only
then reads the answer, and says 502 when the engine resets the socket mid-body.
"""

from __future__ import annotations

import asyncio
import socket
import threading

import pytest
import uvicorn

from dtk_engine import http

CAP_MB = 1
BODY = 8 * 2**20  # well past the cap and past the socket buffers


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DTK_MAX_UPLOAD_MB", str(CAP_MB))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(http.create_app(), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        thread.join(0.01)
    yield sock.getsockname()[1]
    server.should_exit = True
    thread.join(10)


async def _proxy_request(up_port: int, head: bytes, body_len: int, chunked: bool) -> bytes:
    """Act as the proxy: stream the request upstream, then return the upstream answer (or 502)."""
    reader, writer = await asyncio.open_connection("127.0.0.1", up_port)
    try:
        writer.write(head)
        sent, chunk = 0, b"x" * 65536
        while sent < body_len:
            piece = chunk[: body_len - sent]
            writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n" if chunked else piece)
            await writer.drain()
            sent += len(piece)
        if chunked:
            writer.write(b"0\r\n\r\n")
            await writer.drain()
    except (ConnectionError, OSError):
        return b"HTTP/1.1 502 Bad Gateway\r\n\r\n"
    try:
        return await reader.read()
    except (ConnectionError, OSError):
        return b"HTTP/1.1 502 Bad Gateway\r\n\r\n"
    finally:
        writer.close()


@pytest.mark.parametrize("chunked", [False, True])
def test_413_survives_a_streaming_proxy(engine, tmp_path, chunked):
    transfer = "Transfer-Encoding: chunked" if chunked else f"Content-Length: {BODY}"
    head = f"PUT /api/uploads/big.bin HTTP/1.1\r\nHost: x\r\n{transfer}\r\nConnection: close\r\n\r\n"
    for _ in range(5):  # the reset is a race: a broken build fails most rounds, not one
        answer = asyncio.run(_proxy_request(engine, head.encode(), BODY, chunked))
        assert answer.startswith(b"HTTP/1.1 413"), answer[:80]
        assert b"DTK_MAX_UPLOAD_MB" in answer
    assert not list((tmp_path / "uploads").rglob("*")) if (tmp_path / "uploads").exists() else True
