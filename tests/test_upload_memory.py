"""A large upload is streamed to disk: peak memory stays far below the body size.

Needs a real server: the Starlette TestClient hands the whole body over as one message.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import tracemalloc

import pytest
import uvicorn

from dtk_engine import http

TOTAL = 64 * 2**20
PIECE = b"q" * 2**16


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DTK_MAX_UPLOAD_MB", "128")
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


async def _put(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    head = f"PUT /api/uploads/big.bin HTTP/1.1\r\nHost: x\r\nContent-Length: {TOTAL}\r\nConnection: close\r\n\r\n"
    writer.write(head.encode())
    for _ in range(TOTAL // len(PIECE)):
        writer.write(PIECE)
        await writer.drain()
    answer = await reader.read()
    writer.close()
    return answer


def test_peak_memory_is_far_below_the_body_size(engine, tmp_path):
    tracemalloc.start()
    try:
        answer = asyncio.run(_put(engine))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert answer.startswith(b"HTTP/1.1 200"), answer[:80]
    stored = [p for p in (tmp_path / "uploads").rglob("*") if p.is_file()]
    assert [p.name for p in stored] == ["big.bin"]
    assert stored[0].stat().st_size == TOTAL
    assert peak < TOTAL // 8, peak
