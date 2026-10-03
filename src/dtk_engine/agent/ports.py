"""How a tool reaches Studio: in-process (``/mcp`` mount) or over HTTP (stdio ``dtk-mcp``).

Both ports answer the same calls, so every tool is written once.
``RemoteUiPort`` uses stdlib ``urllib`` in a worker thread (no HTTP dependency).
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from dtk_engine.ui_bridge import UiBridge, read_runtime

NO_STUDIO = "no_studio"


class UiPort(Protocol):
    async def get_context(self, session: str | None) -> dict | None: ...

    async def send_command(self, cmd: dict, session: str | None, timeout: float) -> dict: ...

    async def command_status(self, cid: str) -> dict | None: ...

    async def list_attachments(self, session: str) -> list[dict]: ...


class LocalUiPort:
    """The bridge of the app this server is mounted in."""

    def __init__(self, bridge: UiBridge) -> None:
        self.bridge = bridge

    async def get_context(self, session: str | None) -> dict | None:
        return self.bridge.get_context(session)

    async def send_command(self, cmd: dict, session: str | None, timeout: float) -> dict:
        return await self.bridge.send_command(cmd, session=session, timeout=timeout)

    async def command_status(self, cid: str) -> dict | None:
        return self.bridge.command_status(cid)

    async def list_attachments(self, session: str) -> list[dict]:
        from dtk_engine.agent.attachments import registry

        return registry(self.bridge).list(session)


class RemoteUiPort:
    """The running ``dtk-api``, found through the runtime file on every call."""

    async def get_context(self, session: str | None) -> dict | None:
        query = {"session": session} if session else {}
        found = await asyncio.to_thread(self._request, "GET", "/api/ui/context", query)
        return found if isinstance(found, dict) and "session" in found else None

    async def send_command(self, cmd: dict, session: str | None, timeout: float) -> dict:
        query: dict[str, Any] = {"timeout": timeout}
        if session:
            query["session"] = session
        found = await asyncio.to_thread(
            self._request, "POST", "/api/ui/commands", query, cmd, timeout + 10
        )
        if isinstance(found, dict) and "ok" in found:
            return found
        return {"id": cmd.get("id"), "ok": False, "error": NO_STUDIO}

    async def command_status(self, cid: str) -> dict | None:
        path = f"/api/ui/commands/{urllib.parse.quote(cid, safe='')}"
        found = await asyncio.to_thread(self._request, "GET", path, {})
        return found if isinstance(found, dict) and "id" in found else None

    async def list_attachments(self, session: str) -> list[dict]:
        found = await asyncio.to_thread(
            self._request, "GET", "/api/ui/agent/attachments", {"session": session}
        )
        return found if isinstance(found, list) else []

    @staticmethod
    def _request(
        method: str, path: str, query: dict, body: dict | None = None, timeout: float = 10.0
    ) -> Any:
        """JSON answer, or None when no engine is running / it answered an error."""
        runtime = read_runtime()
        if runtime is None or not runtime.get("url"):
            return None
        url = runtime["url"].rstrip("/") + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        headers = {"Authorization": f"Bearer {runtime.get('token', '')}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None
