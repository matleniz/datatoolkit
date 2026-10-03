"""Terminal packs: an agent CLI in a PTY, bridged to Studio over a WebSocket.

Protocol (``docs/agent-chat-protocol.md``, "v2 / Terminal"): Studio opens
``WS /api/ui/terminal?session=&pack=&model=&cols=&rows=&token=`` (guarded in
``http.py``); binary frames are raw PTY bytes both ways, text frames are JSON
control (client ``resize``; engine ``started``, ``exit``, ``error``).

The CLI is one of the external packs (``claude-code``, ``gemini``,
``opencode``) with the config ``dtk-mcp config <pack>`` would write (stdio
``dtk`` server only, built-in tools off where the CLI allows), written to
``$DTK_HOME/agent/terminal/<pack>``. It runs without a shell (argv exec) in
its own session with the PTY as controlling terminal, so Ctrl-C and resizes
reach it as on a real terminal. Closing the socket or stopping the engine
sends SIGHUP to its process group, then SIGTERM, then SIGKILL after a grace.

Opt-in (weaker guarantee than the chat packs: a CLI only turns off what it
lets us): ``DTK_AGENT_TERMINAL`` truthy, or ``dtk-api --terminal``. POSIX only
(stdlib ``pty``); no Windows console support.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import termios
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dtk_engine.agent.configure import stdio_server, write_files
from dtk_engine.agent.options import TERMINAL_IDS, terminal_unavailable
from dtk_engine.agent.packs import PACKS, Pack
from dtk_engine.ui_bridge import dtk_home

MODEL_FLAGS = {"claude-code": "--model", "gemini": "-m", "opencode": "-m"}
DEFAULT_SIZE = (120, 32)
MAX_SIZE = 1000
HUP_GRACE = 1.0
TERM_GRACE = 3.0
READ_CHUNK = 65536
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@\[\]-]{0,199}$")

# WebSocket close codes (the contract's; 1000 = the CLI exited, 1011 = spawn failure).
CLOSE_TOKEN, CLOSE_FORBIDDEN, CLOSE_UNKNOWN, CLOSE_BUSY, CLOSE_INVALID = (
    4401, 4403, 4404, 4409, 4422,
)


class TerminalError(Exception):
    """A refused terminal request; ``code`` is the WebSocket close code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def terminal_pack(pack_id: str) -> Pack:
    if pack_id not in TERMINAL_IDS:
        raise TerminalError(
            CLOSE_UNKNOWN, f"unknown terminal pack {pack_id!r}; known: {', '.join(TERMINAL_IDS)}"
        )
    return PACKS[pack_id]


def check_model(model: str | None) -> str | None:
    """The model passed as one argv item; refused when it could read as a flag."""
    if not model:
        return None
    if not _MODEL.match(model):
        raise TerminalError(CLOSE_INVALID, f"invalid model {model!r}")
    return model


def check_size(cols: str | None, rows: str | None) -> tuple[int, int]:
    try:
        size = (int(cols or DEFAULT_SIZE[0]), int(rows or DEFAULT_SIZE[1]))
    except ValueError:
        raise TerminalError(CLOSE_INVALID, "cols and rows must be integers") from None
    if not all(1 <= n <= MAX_SIZE for n in size):
        raise TerminalError(CLOSE_INVALID, f"cols and rows must be within 1..{MAX_SIZE}")
    return size


def prepare(pack: Pack) -> Path:
    """Write the pack's dtk-only config into its terminal directory; return it."""
    directory = dtk_home() / "agent" / "terminal" / pack.id
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_files(pack.files(stdio_server()), directory, force=True, secret=False)
    return directory


def build_argv(pack: Pack, directory: Path, model: str | None) -> list[str]:
    argv = pack.launch_command(directory)
    if model:
        argv += [MODEL_FLAGS[pack.id], model]
    return argv


# Exec shim: new session, the PTY (stdin) as controlling terminal, then exec the
# CLI (same pid). A separate interpreter, since ``preexec_fn`` is unsafe with threads.
_EXEC = (
    "import fcntl, os, sys, termios; os.setsid(); "
    "fcntl.ioctl(0, termios.TIOCSCTTY, 0); os.execvp(sys.argv[1], sys.argv[1:])"
)


def set_size(fd: int, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


class Terminal:
    """One CLI process on one PTY."""

    def __init__(self, argv: list[str], cwd: Path, size: tuple[int, int]) -> None:
        self.argv = argv
        if shutil.which(argv[0]) is None:
            raise FileNotFoundError(f"{argv[0]} not found on PATH")
        master, slave = os.openpty()
        set_size(master, *size)
        env = {**os.environ, "TERM": "xterm-256color", "COLORTERM": "truecolor"}
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-I", "-c", _EXEC, *argv],
                stdin=slave, stdout=slave, stderr=slave, cwd=cwd, env=env, close_fds=True,
            )
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        self.fd: int | None = master

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        while view and self.fd is not None:
            try:
                view = view[os.write(self.fd, view):]
            except BlockingIOError:
                continue
            except OSError:
                return

    def resize(self, cols: int, rows: int) -> None:
        if self.fd is not None:
            set_size(self.fd, cols, rows)

    def read(self) -> bytes | None:
        """Available output; None = nothing yet, b"" = end of stream (PTY closed)."""
        if self.fd is None:
            return b""
        try:
            return os.read(self.fd, READ_CHUNK)
        except BlockingIOError:
            return None
        except OSError:  # EIO: the CLI side of the PTY is closed
            return b""

    def _signal(self, sig: int) -> None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.proc.pid, sig)

    async def wait_exit(self, timeout: float) -> bool:
        try:
            await asyncio.to_thread(self.proc.wait, timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    async def stop(self) -> int | None:
        """SIGHUP, then SIGTERM, then SIGKILL (process group); reap; close the PTY."""
        if self.proc.poll() is None:
            for sig, grace in ((signal.SIGHUP, HUP_GRACE), (signal.SIGTERM, TERM_GRACE)):
                self._signal(sig)
                if await self.wait_exit(grace):
                    break
            else:
                self._signal(signal.SIGKILL)
                await self.wait_exit(TERM_GRACE)
        self._signal(signal.SIGKILL)  # leftovers in the group (the CLI's own children)
        self.close_fd()
        return self.proc.returncode

    def close_fd(self) -> None:
        if self.fd is not None:
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = None


Socket = Any  # a Starlette ``WebSocket`` (send_bytes / send_text / receive / close)


class Terminals:
    """The app's live terminals, one per (Studio session, pack)."""

    def __init__(self, argv_for: Callable[[Pack, Path, str | None], list[str]] = build_argv):
        self.argv_for = argv_for
        self._live: dict[tuple[str, str], Terminal] = {}

    def open(self, session: str, pack_id: str, model: str | None, size: tuple[int, int]) -> Terminal:
        """Spawn the pack's CLI (``TerminalError`` when refused, ``OSError`` on spawn)."""
        reason = terminal_unavailable()
        if reason is not None:
            raise TerminalError(CLOSE_FORBIDDEN, reason)
        pack = terminal_pack(pack_id)
        model = check_model(model)
        if (session, pack_id) in self._live:
            raise TerminalError(CLOSE_BUSY, f"session already has a {pack_id} terminal")
        directory = prepare(pack)
        terminal = Terminal(self.argv_for(pack, directory, model), directory, size)
        self._live[(session, pack_id)] = terminal
        return terminal

    async def release(self, session: str, pack_id: str) -> int | None:
        terminal = self._live.pop((session, pack_id), None)
        return None if terminal is None else await terminal.stop()

    async def close(self) -> None:
        """Engine shutdown: stop every CLI."""
        live, self._live = list(self._live.values()), {}
        await asyncio.gather(*(t.stop() for t in live), return_exceptions=True)

    def __len__(self) -> int:
        return len(self._live)


async def _pump_out(terminal: Terminal, ws: Socket) -> None:
    """PTY output -> binary frames, until the CLI closes the PTY."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def readable() -> None:
        data = terminal.read()
        if data is None:
            return
        if not data:
            loop.remove_reader(fd)
        queue.put_nowait(data)

    fd = terminal.fd
    assert fd is not None
    loop.add_reader(fd, readable)
    try:
        while data := await queue.get():
            await ws.send_bytes(data)
    finally:
        with contextlib.suppress(ValueError, OSError):
            loop.remove_reader(fd)


def _control(terminal: Terminal, text: str) -> None:
    """A client JSON control frame (only ``resize``; anything else is ignored)."""
    try:
        message = json.loads(text)
        if message.get("type") == "resize":
            terminal.resize(*check_size(message.get("cols"), message.get("rows")))
    except (ValueError, AttributeError, TypeError, TerminalError):
        return


async def _pump_in(terminal: Terminal, ws: Socket) -> None:
    """Binary frames -> PTY input, text frames -> control, until the client leaves."""
    while True:
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            return
        if message.get("bytes") is not None:
            terminal.write(message["bytes"])
        elif message.get("text") is not None:
            _control(terminal, message["text"])


async def serve(ws: Socket, terminals: Terminals, params: dict[str, str]) -> None:
    """Run one accepted terminal socket to its end (the guard already passed)."""
    session, pack_id = params.get("session") or "", params.get("pack") or ""
    try:
        if not session:
            raise TerminalError(CLOSE_INVALID, "session is required")
        size = check_size(params.get("cols"), params.get("rows"))
        terminal = terminals.open(session, pack_id, params.get("model") or None, size)
    except TerminalError as exc:
        await ws.close(exc.code, str(exc))
        return
    except OSError as exc:  # the CLI is missing or cannot start
        await ws.send_text(json.dumps({"type": "error", "message": f"cannot start: {exc}"}))
        await ws.close(1011)
        return
    model = params.get("model") or None
    started = {"type": "started", "pack": pack_id, "model": model, "command": terminal.argv}
    await ws.send_text(json.dumps(started))
    out = asyncio.create_task(_pump_out(terminal, ws))
    inp = asyncio.create_task(_pump_in(terminal, ws))
    try:
        done, _ = await asyncio.wait({out, inp}, return_when=asyncio.FIRST_COMPLETED)
        if out in done:  # the CLI closed the PTY: let it exit on its own first
            await terminal.wait_exit(HUP_GRACE)
    finally:
        for task in (out, inp):
            task.cancel()
        code = await terminals.release(session, pack_id)
    if out in done and not out.cancelled() and out.exception() is None:
        with contextlib.suppress(Exception):  # the client may be gone already
            await ws.send_text(json.dumps({"type": "exit", "code": code}))
            await ws.close(1000)
