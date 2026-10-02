"""In-memory UI bridge between Studio (browser) and a future agent layer.

Pure asyncio + stdlib: it imports nothing from ops / keys / sources /
workspace / contract. Only ``http.py`` imports it (routes under ``/api/ui``).

It holds, in memory only (never on disk, never in the workspace JSON):

- the latest UI context per Studio session (``put_context`` / ``get_context``;
  "most recent session" = last ``put_context``);
- the SSE listeners per session (one ``asyncio.Queue`` each);
- the pending commands (id -> future) resolved by Studio's ack.

``send_command`` is the Python API for the MCP layer: it relays a command to
the target session's listeners and returns the ack dict
``{id, ok, error?, identity?}``. Bridge-made failures: ``no_studio`` (no live
listener, or the listener went away before acking) and ``timeout``. A
``stale`` ack is just an ack Studio sends; it is passed through.

Access guard (applied by ``http.py`` on every ``/api/ui/*`` route):

- per-run token: ``DTK_UI_TOKEN`` if set, else ``secrets.token_urlsafe(32)``
  generated once per app and readable as ``app.state.ui_bridge.token``
  (``Authorization: Bearer <token>`` or ``?token=`` for ``EventSource``);
  never logged or persisted;
- ``Origin`` (when present) must be a CORS origin or the request's own origin;
- ``Host`` hostname must be ``localhost`` / ``127.0.0.1`` / ``[::1]`` or listed
  in ``DTK_UI_ALLOWED_HOSTS`` (comma separated hostnames; a Docker / nginx
  setup would need it, the bridge is not in the image in phase 1).

Runtime file: while ``dtk-api`` runs, ``write_runtime`` publishes
``{url, token, pid, started}`` at ``$DTK_HOME/agent/runtime.json`` (owner-only
``0o600``, directory ``0o700``; removed on shutdown by ``clear_runtime``), the
way Jupyter publishes its server info. Readers (``read_runtime``): the Vite dev
plugin (token for a plain ``npm run dev``), ``dtk-mcp`` (stdio proxy to the
running ``/mcp``) and ``dtk-mcp doctor``. ``DTK_UI_RUNTIME_FILE=0`` skips it.
The token is still never stored in a workspace, localStorage or a log.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import secrets
import tempfile
from datetime import UTC, datetime
from pathlib import Path

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")


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

    def __init__(self, token: str | None = None) -> None:
        self.token: str = token or os.environ.get("DTK_UI_TOKEN") or secrets.token_urlsafe(32)
        self._contexts: dict[str, dict] = {}  # insertion order = recency (last PUT last)
        self._listeners: dict[str, list[asyncio.Queue]] = {}
        self._pending: dict[str, tuple[asyncio.Future, str]] = {}
        self._ids = itertools.count(1)
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
        self._listeners.setdefault(session, []).append(queue)
        return queue

    def remove_listener(self, session: str, queue: asyncio.Queue) -> None:
        queues = self._listeners.get(session, [])
        if queue in queues:
            queues.remove(queue)
        if not queues:
            self._listeners.pop(session, None)
            self._fail_pending(session, "no_studio")

    def has_listener(self, session: str) -> bool:
        return bool(self._listeners.get(session))

    def close(self) -> None:
        """Server shutdown: wake every listener (``None`` sentinel) and fail pending."""
        self._closed = True
        for queues in self._listeners.values():
            for queue in queues:
                queue.put_nowait(None)
        for session in {s for _, s in self._pending.values()}:
            self._fail_pending(session, "no_studio")

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

    def ack(self, ack: dict) -> bool:
        """Resolve the pending command ``ack["id"]``; False when the id is unknown."""
        entry = self._pending.get(ack["id"])
        if entry is None or entry[0].done():
            return False
        entry[0].set_result({k: v for k, v in ack.items() if v is not None})
        return True

    async def send_command(
        self, cmd: dict, *, session: str | None = None, timeout: float = 30.0
    ) -> dict:
        """Relay ``cmd`` to Studio and wait for its ack (see module docstring)."""
        cid = str(cmd.get("id") or f"c{next(self._ids)}")
        target = self.target_session(session)
        if target is None:
            return {"id": cid, "ok": False, "error": "no_studio"}
        if cid in self._pending:
            return {"id": cid, "ok": False, "error": "duplicate_id"}
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cid] = (future, target)
        try:
            for queue in self._listeners[target]:
                queue.put_nowait({**cmd, "id": cid})
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            return {"id": cid, "ok": False, "error": "timeout"}
        finally:
            self._pending.pop(cid, None)
            future.cancel()


# -- runtime file ($DTK_HOME/agent/runtime.json) ---------------------------


def runtime_path() -> Path:
    """``$DTK_HOME/agent/runtime.json`` (``~/.datatoolkit`` when unset)."""
    home = os.environ.get("DTK_HOME")
    base = Path(home) if home else Path.home() / ".datatoolkit"
    return base / "agent" / "runtime.json"


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
