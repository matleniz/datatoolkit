"""In-memory UI bridge between Studio (browser) and a future agent layer.

Pure asyncio + stdlib: it imports nothing from ops / keys / sources /
workspace / contract. Only ``http.py`` imports it (routes under ``/api/ui``).

It holds, in memory only (never on disk, never in the workspace JSON):

- the latest UI context per Studio session (``put_context`` / ``get_context``;
  "most recent session" = last ``put_context``);
- the SSE listeners per session (one ``asyncio.Queue`` each);
- the pending commands (id -> future) resolved by Studio's ack.

``emit(session, event, data)`` pushes any other named SSE event to a
session's listeners (``event: agent`` = the chat protocol of
``dtk_engine.agent.chat``); fire-and-forget, dropped when nobody listens.

``send_command`` is the Python API for the MCP layer: it relays a command to
the target session's listeners and returns the ack dict
``{id, ok, error?, identity?}``. Bridge-made failures: ``no_studio`` (no live
listener, or the listener went away before acking) and ``timeout``. A
``stale`` ack is just an ack Studio sends; it is passed through.

Reconnect grace (``DTK_UI_RECONNECT_GRACE`` seconds, default 3, 0 = off): an
``EventSource`` that is reconnecting has no listener for a moment. With no
listener, ``send_command`` waits up to the grace (bounded by its timeout) for
one to attach before answering ``no_studio``; when a session's last listener
drops, its pending commands and reviews are failed only after the grace, and
commands that were queued but never read are redelivered to the new listener.

Reviews (a command Studio holds for the user, e.g. a destructive
``propose_steps``): Studio first sends an **interim** ack ``{id, pending:
"review"}``; ``send_command`` then returns at once ``{id, ok: None, pending:
"review"}`` and the command waits in the reviews table until Studio's final
ack (accepted however late) or the review deadline (``DTK_UI_REVIEW_TIMEOUT``
seconds, default 900 -> ``timeout``; listener gone -> ``no_studio``).
``command_status(id)`` answers for any recent id (last 200 final results):
the final ack, the ``pending`` dict while under review, or None if unknown.

Access guard (applied by ``http.py`` on every ``/api/ui/*`` route):

- per-run token: ``DTK_UI_TOKEN`` if set, else ``secrets.token_urlsafe(32)``
  generated once per app and readable as ``app.state.ui_bridge.token``
  (``Authorization: Bearer <token>`` or ``?token=`` for ``EventSource``);
  never logged or persisted;
- ``Origin`` (when present) must be a CORS origin or the request's own origin;
- ``Host`` hostname must be ``localhost`` / ``127.0.0.1`` / ``[::1]`` or listed
  in ``DTK_UI_ALLOWED_HOSTS`` (comma separated hostnames; needed behind a
  proxy that rewrites ``Host``, see the README's "Agent in the Docker image").

Runtime file: while ``dtk-api`` runs, ``write_runtime`` publishes
``{url, token, pid, started}`` at ``$DTK_HOME/agent/runtime.json`` (owner-only
``0o600``, directory ``0o700``; removed on shutdown by ``clear_runtime``), the
way Jupyter publishes its server info. Readers (``read_runtime``): the Vite dev
plugin (token for a plain ``npm run dev``), ``dtk-mcp`` (stdio proxy to the
running ``/mcp``) and ``dtk-mcp doctor``. ``DTK_UI_RUNTIME_FILE=0`` skips it.
The token is still never stored in a workspace, localStorage or a log.

Log redaction: ``EventSource`` / browser WebSockets pass the token as
``?token=``, which uvicorn prints in its access and WebSocket lines.
``RedactTokenFilter`` (installed by ``dtk-api`` on uvicorn's handlers, see
``dtk_engine.http.uvicorn_log_config``) rewrites it to ``token=***``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import secrets
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
DEFAULT_REVIEW_TIMEOUT = 900.0
DEFAULT_RECONNECT_GRACE = 3.0
MAX_RESULTS = 200
_IS_WINDOWS = sys.platform == "win32"  # monkeypatched in tests
_TOKEN_IN_QUERY = re.compile(r"([?&]token=)[^&\s\"']*", re.IGNORECASE)


def review_timeout_from_env() -> float:
    """``DTK_UI_REVIEW_TIMEOUT`` seconds (positive number), else 900."""
    try:
        value = float(os.environ.get("DTK_UI_REVIEW_TIMEOUT", ""))
    except ValueError:
        return DEFAULT_REVIEW_TIMEOUT
    return value if value > 0 else DEFAULT_REVIEW_TIMEOUT


def reconnect_grace_from_env() -> float:
    """``DTK_UI_RECONNECT_GRACE`` seconds (>= 0; 0 = fail at once), else 3."""
    try:
        value = float(os.environ.get("DTK_UI_RECONNECT_GRACE", ""))
    except ValueError:
        return DEFAULT_RECONNECT_GRACE
    return value if value >= 0 else DEFAULT_RECONNECT_GRACE


def allowed_hosts() -> set[str]:
    """Loopback names plus the ``DTK_UI_ALLOWED_HOSTS`` comma list (lowercase)."""
    extra = os.environ.get("DTK_UI_ALLOWED_HOSTS", "")
    return {*LOOPBACK_HOSTS, *(p.strip().lower() for p in extra.split(",") if p.strip())}


def host_name(host_header: str) -> str:
    """Hostname of a ``Host`` header value (port stripped, ``[::1]`` kept bracketed)."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[: host.find("]") + 1] if "]" in host else host
    return host.partition(":")[0]


class UiBridge:
    """Context store, SSE listener registry and command / ack relay."""

    def __init__(
        self,
        token: str | None = None,
        *,
        review_timeout: float | None = None,
        reconnect_grace: float | None = None,
    ) -> None:
        self.token: str = token or os.environ.get("DTK_UI_TOKEN") or secrets.token_urlsafe(32)
        self.review_timeout = review_timeout or review_timeout_from_env()
        self.reconnect_grace = (
            reconnect_grace_from_env() if reconnect_grace is None else max(reconnect_grace, 0.0)
        )
        self._grace_timers: dict[str, asyncio.TimerHandle] = {}
        self._undelivered: dict[str, list[dict]] = {}  # queued, never read, last listener gone
        self._attach_waiters: list[asyncio.Future] = []
        self._contexts: dict[str, dict] = {}  # insertion order = recency (last PUT last)
        self._listeners: dict[str, list[asyncio.Queue]] = {}
        self._pending: dict[str, tuple[asyncio.Future, str]] = {}
        self._reviews: dict[str, tuple[str, float]] = {}  # id -> (session, deadline)
        self._results: OrderedDict[str, dict] = OrderedDict()  # final, most recent last
        self._ids = itertools.count(1)
        self.attachments: Any = None  # AttachmentRegistry (agent/attachments.py), built on first use
        # Called with (session, listening) when a session gets its first listener
        # (True) or loses its last one (False); the agent hub reaps idle chats with it.
        self.listener_hooks: list[Callable[[str, bool], None]] = []
        self._closed = False

    # -- context ---------------------------------------------------------
    def put_context(self, context: dict) -> None:
        session = context["session"]
        self._contexts.pop(session, None)
        self._contexts[session] = context

    def get_context(self, session: str | None = None) -> dict | None:
        if session is not None:
            return self._contexts.get(session)
        return next(reversed(self._contexts.values()), None)

    # -- listeners -------------------------------------------------------
    def add_listener(self, session: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        if self._closed:
            queue.put_nowait(None)
        first = not self._listeners.get(session)
        self._listeners.setdefault(session, []).append(queue)
        timer = self._grace_timers.pop(session, None)
        if timer is not None:
            timer.cancel()
        for cmd in self._undelivered.pop(session, []):
            if cmd["id"] in self._pending:  # not timed out meanwhile
                queue.put_nowait(cmd)
        for waiter in self._attach_waiters:
            if not waiter.done():
                waiter.set_result(None)
        if first:
            self._notify(session, True)
        return queue

    def remove_listener(self, session: str, queue: asyncio.Queue) -> None:
        queues = self._listeners.get(session, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            had = self._listeners.pop(session, None) is not None
            self._fail_after_grace(session, queue)
            if had:
                self._notify(session, False)

    def _fail_after_grace(self, session: str, queue: asyncio.Queue) -> None:
        """Last listener of ``session`` is gone: fail its commands now, or after the grace."""
        waiting = any(s == session for _, s in self._pending.values()) or any(
            s == session for s, _ in self._reviews.values()
        )
        if not waiting:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self.reconnect_grace <= 0 or self._closed or loop is None:
            self._fail_pending(session, "no_studio")
            return
        while not queue.empty():
            item = queue.get_nowait()
            if isinstance(item, dict) and item.get("id") in self._pending:
                self._undelivered.setdefault(session, []).append(item)
        previous = self._grace_timers.pop(session, None)
        if previous is not None:
            previous.cancel()
        self._grace_timers[session] = loop.call_later(
            self.reconnect_grace, self._grace_expired, session
        )

    def _grace_expired(self, session: str) -> None:
        self._grace_timers.pop(session, None)
        if not self.has_listener(session):
            self._undelivered.pop(session, None)
            self._fail_pending(session, "no_studio")

    async def _wait_for_listener(self, session: str | None, wait: float) -> str | None:
        """Target session once one (re)attaches within ``wait`` seconds, else None."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            waiter: asyncio.Future = loop.create_future()
            self._attach_waiters.append(waiter)
            try:
                await asyncio.wait_for(waiter, remaining)
            except TimeoutError:
                return None
            finally:
                self._attach_waiters.remove(waiter)
                waiter.cancel()
            if self._closed:
                return None
            target = self.target_session(session)
            if target is not None:
                return target

    def _notify(self, session: str, listening: bool) -> None:
        for hook in self.listener_hooks:
            hook(session, listening)

    def has_listener(self, session: str) -> bool:
        return bool(self._listeners.get(session))

    def emit(self, session: str, event: str, data: dict) -> bool:
        """Queue SSE ``event`` with ``data`` for ``session``; False when nobody listens."""
        queues = self._listeners.get(session, [])
        for queue in queues:
            queue.put_nowait((event, data))
        return bool(queues)

    def close(self) -> None:
        """Server shutdown: wake every listener (``None`` sentinel) and fail pending."""
        self._closed = True
        for queues in self._listeners.values():
            for queue in queues:
                queue.put_nowait(None)
        for timer in self._grace_timers.values():
            timer.cancel()
        self._grace_timers.clear()
        self._undelivered.clear()
        for waiter in self._attach_waiters:
            if not waiter.done():
                waiter.set_result(None)
        sessions = {s for _, s in self._pending.values()} | {s for s, _ in self._reviews.values()}
        for session in sessions:
            self._fail_pending(session, "no_studio")

    def sessions(self) -> list[dict]:
        """Known sessions ``{session, listening, workspace, identity}``, most recent last.

        Sessions with a context come first in publish order, then listener-only ones.
        """
        names = [*self._contexts, *(s for s in self._listeners if s not in self._contexts)]
        out = []
        for name in names:
            context = self._contexts.get(name, {})
            out.append({
                "session": name,
                "listening": self.has_listener(name),
                "workspace": context.get("workspace"),
                "identity": context.get("identity"),
            })
        return out

    def target_session(self, session: str | None = None) -> str | None:
        """Explicit session if it listens, else most recent context with a listener, else any."""
        if session is not None:
            return session if self.has_listener(session) else None
        for candidate in reversed(self._contexts):
            if self.has_listener(candidate):
                return candidate
        return next(reversed(self._listeners), None)

    # -- commands --------------------------------------------------------
    def _fail_pending(self, session: str, error: str) -> None:
        for cid, (future, target) in list(self._pending.items()):
            if target == session and not future.done():
                future.set_result({"id": cid, "ok": False, "error": error})
        for cid, (target, _) in list(self._reviews.items()):
            if target == session:
                del self._reviews[cid]
                self._record({"id": cid, "ok": False, "error": error})

    def _record(self, result: dict) -> None:
        self._results.pop(result["id"], None)
        self._results[result["id"]] = result
        while len(self._results) > MAX_RESULTS:
            self._results.popitem(last=False)

    def ack(self, ack: dict) -> bool:
        """Resolve command ``ack["id"]`` (interim ``pending`` or final); False when unknown."""
        cid = ack["id"]
        final = {k: v for k, v in ack.items() if v is not None and k != "pending"}
        if cid in self._reviews:
            if ack.get("pending") is None:
                del self._reviews[cid]
                self._record(final)
            return True
        entry = self._pending.get(cid)
        if entry is None or entry[0].done():
            return False
        future, session = entry
        if ack.get("pending") is not None:
            self._reviews[cid] = (session, time.monotonic() + self.review_timeout)
            future.set_result({"id": cid, "ok": None, "pending": ack["pending"]})
        else:
            future.set_result(final)
        return True

    def command_status(self, cid: str) -> dict | None:
        """Final result of a recent command, ``pending`` while under review, else None."""
        review = self._reviews.get(cid)
        if review is not None:
            if time.monotonic() < review[1]:
                return {"id": cid, "ok": None, "pending": "review"}
            del self._reviews[cid]
            self._record({"id": cid, "ok": False, "error": "timeout"})
        return self._results.get(cid)

    async def send_command(
        self, cmd: dict, *, session: str | None = None, timeout: float = 30.0
    ) -> dict:
        """Relay ``cmd`` to Studio and wait for its ack (see module docstring)."""
        cid = str(cmd.get("id") or f"c{next(self._ids)}")
        target = self.target_session(session)
        if target is None and self.reconnect_grace > 0 and not self._closed:
            target = await self._wait_for_listener(session, min(self.reconnect_grace, timeout))
        if target is None:
            return {"id": cid, "ok": False, "error": "no_studio"}
        if cid in self._pending or cid in self._reviews:
            return {"id": cid, "ok": False, "error": "duplicate_id"}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = (future, target)
        try:
            for queue in self._listeners[target]:
                queue.put_nowait({**cmd, "id": cid})
            result = await asyncio.wait_for(future, timeout)
        except TimeoutError:
            result = {"id": cid, "ok": False, "error": "timeout"}
        finally:
            self._pending.pop(cid, None)
            future.cancel()
        if cid not in self._reviews:
            self._record(result)
        return result


# -- runtime file ($DTK_HOME/agent/runtime.json) ---------------------------


def dtk_home() -> Path:
    """``$DTK_HOME`` (``~`` expanded), ``~/.datatoolkit`` when unset: the one resolver
    for http and agent (stdlib-only here so both may import it)."""
    home = os.environ.get("DTK_HOME")
    return Path(home).expanduser() if home else Path.home() / ".datatoolkit"


def runtime_path() -> Path:
    """``$DTK_HOME/agent/runtime.json``."""
    return dtk_home() / "agent" / "runtime.json"


def runtime_file_enabled() -> bool:
    return os.environ.get("DTK_UI_RUNTIME_FILE", "1").strip().lower() not in ("0", "false", "no")


def write_runtime(url: str, token: str) -> Path | None:
    """Publish ``{url, token, pid, started}``; owner-only, written atomically.

    Returns the path, or None when ``DTK_UI_RUNTIME_FILE=0``.
    """
    if not runtime_file_enabled():
        return None
    path = runtime_path()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "url": url,
        "token": token,
        "pid": os.getpid(),
        "started": datetime.now(UTC).isoformat(),
    }
    # mkstemp creates the file 0o600 from the start: never readable by others.
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".runtime-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is running. Never probes on Windows, where ``os.kill`` TERMINATES
    the process for any signal but CTRL_C_EVENT/CTRL_BREAK_EVENT: the runtime file is then
    trusted (``True``) and a stale one fails at connect time. Same guard as the export
    staging check; duplicated because this module imports nothing from the package."""
    if _IS_WINDOWS:
        return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def read_runtime() -> dict | None:
    """The running engine's runtime info, or None (missing, garbage, dead pid)."""
    try:
        data = json.loads(runtime_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pid"), int):
        return None
    return data if _pid_alive(data["pid"]) else None


def clear_runtime(pid: int | None = None) -> None:
    """Remove the runtime file if it belongs to ``pid`` (default: this process)."""
    path = runtime_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and data.get("pid") == (os.getpid() if pid is None else pid):
        path.unlink(missing_ok=True)


# -- log redaction ------------------------------------------------------------


def redact_token(text: str) -> str:
    """``...?token=<value>`` -> ``...?token=***`` (any query position)."""
    return _TOKEN_IN_QUERY.sub(r"\1***", text)


def _redact_arg(value: Any) -> Any:
    return redact_token(value) if isinstance(value, str) else value


class RedactTokenFilter(logging.Filter):
    """Redacts ``token=`` query values in a record's message and args.

    Args are redacted in place (not merged into ``msg``) because uvicorn's
    ``AccessFormatter`` unpacks ``record.args``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_token(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_arg(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _redact_arg(v) for k, v in record.args.items()}
        return True
