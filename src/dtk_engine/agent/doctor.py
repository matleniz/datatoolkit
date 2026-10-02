"""``dtk-mcp doctor``: is an engine running, is its token valid, is Studio connected,
which pack CLIs are on ``PATH``.

Token check: ``GET /api/ui/context`` answers 200 / 404 with a valid token, 401
without. "Studio connected" = ``GET /api/ui/sessions`` lists a listening
session. ``ok`` (exit 0) = engine reachable with a valid token.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

from dtk_engine.agent.packs import PACKS, Pack
from dtk_engine.ui_bridge import read_runtime, runtime_path

HTTP_TIMEOUT = 5.0
VERSION_TIMEOUT = 5.0


def _get(url: str, token: str) -> tuple[int | None, Any]:
    """``(status, json body)``; ``(None, None)`` when the engine cannot be reached."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return None, None


def _engine(runtime: dict | None) -> dict:
    report: dict[str, Any] = {"reachable": False, "token_valid": None, "studio_connected": False,
                              "sessions": []}
    if runtime is None or not runtime.get("url"):
        return report
    base, token = runtime["url"].rstrip("/"), str(runtime.get("token", ""))
    status, _ = _get(base + "/api/ui/context", token)
    if status is None:
        return report
    report["reachable"] = True
    report["token_valid"] = status in (200, 404)
    if report["token_valid"]:
        status, sessions = _get(base + "/api/ui/sessions", token)
        report["sessions"] = sessions if status == 200 and isinstance(sessions, list) else []
        report["studio_connected"] = any(s.get("listening") for s in report["sessions"])
    return report


def _version(executable: str) -> str | None:
    try:
        done = subprocess.run(
            [executable, "--version"], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=VERSION_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (done.stdout or done.stderr).strip().splitlines()
    return lines[0].strip() if lines else None


def _pack(pack: Pack, path: str | None) -> dict:
    executable = shutil.which(pack.cli, path=path) if pack.cli else None
    return {
        "id": pack.id,
        "cli": pack.cli,
        "detected": pack.detect(path),
        "path": executable,
        "version": _version(executable) if executable else None,
    }


def diagnose(path: str | None = None) -> dict:
    """The full report (``--json`` prints it as is)."""
    runtime = read_runtime()
    engine = _engine(runtime)
    return {
        "ok": bool(engine["reachable"] and engine["token_valid"]),
        "runtime": {
            "path": str(runtime_path()),
            "found": runtime is not None,
            "url": runtime.get("url") if runtime else None,
            "pid": runtime.get("pid") if runtime else None,
        },
        "engine": engine,
        "packs": [_pack(p, path) for p in PACKS.values() if p.cli],
    }


def _mark(flag: bool | None) -> str:
    return {True: "ok", False: "NO", None: "--"}[flag]


def render(report: dict) -> str:
    runtime, engine = report["runtime"], report["engine"]
    found = f"{runtime['url']} (pid {runtime['pid']})" if runtime["found"] else "none"
    lines = [
        f"runtime file     {runtime['path']}: {found}",
        f"engine reachable {_mark(engine['reachable'] if runtime['found'] else None)}",
        f"token valid      {_mark(engine['token_valid'])}",
        f"studio connected {_mark(engine['studio_connected'] if engine['token_valid'] else None)}",
    ]
    for pack in report["packs"]:
        where = f"{pack['path']} ({pack['version'] or 'version unknown'})"
        lines.append(f"pack {pack['id']:<11} {where if pack['detected'] else 'not on PATH'}")
    if not runtime["found"]:
        lines.append("no running engine: start dtk-api")
    return "\n".join(lines) + "\n"
