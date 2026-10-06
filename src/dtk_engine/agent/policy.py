"""Agent policy: what an agent may read, how much it gets back, what it may write.

Pure Python (no ``mcp`` import). Every agent tool runs its arguments through
``check_args`` and its output through ``frame`` before answering.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dtk_engine import contract
from dtk_engine.agent import commands as _commands
from dtk_engine.errors import KeyParamsError
from dtk_engine.ui_bridge import dtk_home

DEFAULT_ROWS = 50
MAX_ROWS = 500
# Default; env DTK_AGENT_MAX_CHARS overrides. Dense JSON runs ~3 characters per
# token: 50k stays well under the 25k-token MCP tool output limit of the CLIs.
MAX_RESPONSE_CHARS = 50_000
PREVIEW_ITEMS = 20  # changed cells / removed rids / state list items, compact preview
PREVIEW_DETAIL_ITEMS = 200  # same, ``detail: true``
PREVIEW_COLUMNS = 200  # column name lists in a preview
_SHRINK_LIMITS = (50, 20, 10, 5, 2, 1)
_SHRINK_STRINGS = (500, 100)
UI_COMMANDS = frozenset(_commands.UI_COMMANDS)

_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_SECRET_KEYS = ("token", "secret", "password", "authorization", "api_key")
_TRUTHY = {"1", "true", "yes"}


class PolicyError(KeyParamsError):
    """The agent asked for something the policy refuses."""


def allowed_roots() -> list[Path]:
    """Realpaths of ``$DTK_HOME`` and, when set, ``$DTK_UPLOAD_DIR``."""
    roots = [Path(os.path.realpath(dtk_home()))]
    upload = os.environ.get("DTK_UPLOAD_DIR")
    if upload:
        roots.append(Path(os.path.realpath(Path(upload).expanduser())))
    return roots


def control_dir() -> Path:
    """Realpath of ``$DTK_HOME/agent``: the engine's own files (UI token, audit
    log, terminal configs), never readable by the agent."""
    return Path(os.path.realpath(dtk_home() / "agent"))


def _map_windows(path: str) -> str:
    """Same Windows-drive -> ``/mnt/<d>/`` mapping as the readers (Linux only)."""
    match = _WINDOWS_DRIVE.match(path)
    if not match or os.name != "posix" or os.path.exists(path):
        return path
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def _real(path: str) -> str:
    return os.path.realpath(_map_windows(path))


def _paths_in(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "path" and isinstance(value, str):
                yield value
            else:
                yield from _paths_in(value)
    elif isinstance(node, list):
        for item in node:
            yield from _paths_in(item)


def workspace_paths() -> set[str]:
    """Every ``path`` string in every stored workspace, as given and as realpath."""
    found: set[str] = set()
    for ws in contract.list_workspaces():
        for path in _paths_in(ws):
            found.add(path)
            found.add(_real(path))
    return found


def _has_sql(node: Any) -> bool:
    if isinstance(node, dict):
        return node.get("kind") == "sql" or any(_has_sql(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_sql(v) for v in node)
    return False


def check_args(args: Any) -> None:
    """Refuse ``sql`` sources and any file path outside the allowed scope."""
    if _has_sql(args):
        raise PolicyError("sql sources are not available to the agent")
    paths = list(_paths_in(args))
    if not paths:
        return
    roots = allowed_roots()
    known = workspace_paths()
    control = control_dir()
    for path in paths:
        real = _real(path)
        if Path(real).is_relative_to(control):
            raise PolicyError(
                f"path not allowed for the agent: {path} "
                "(engine control files under $DTK_HOME/agent)"
            )
        if any(Path(real).is_relative_to(root) for root in roots):
            continue
        if path in known or real in known:
            continue
        raise PolicyError(
            f"path not allowed for the agent: {path} "
            "(files under $DTK_HOME or used by a workspace only)"
        )


def row_limit(limit: int | None) -> int:
    if limit is None:
        return DEFAULT_ROWS
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_ROWS
    ):
        raise PolicyError(f"limit must be between 1 and {MAX_ROWS}")
    return limit


def compact_result(result: dict, include_figures: bool = False) -> dict:
    """Drop the plotly JSON from each figure unless ``include_figures``."""
    if include_figures:
        return result
    figures = [
        {"title": f.get("title"), "group": f.get("group"), "main": f.get("main")}
        for f in result.get("figures") or []
    ]
    return {**result, "figures": figures}


def max_response_chars() -> int:
    try:
        return int(os.environ["DTK_AGENT_MAX_CHARS"])
    except (KeyError, ValueError):
        return MAX_RESPONSE_CHARS


def _size(data: Any) -> int:
    return len(json.dumps(data, default=str))


def _row_lists(data: Any) -> list[list]:
    """Mutable row lists we may trim: ``rows`` / ``records`` / ``tables[].records``."""
    if not isinstance(data, dict):
        return []
    lists = [data[k] for k in ("rows", "records") if isinstance(data.get(k), list)]
    lists.extend(
        table["records"]
        for table in data.get("tables") or []
        if isinstance(table, dict) and isinstance(table.get("records"), list)
    )
    return lists


def _trim_rows(data: Any, cap: int) -> tuple[Any, str] | None:
    lists = _row_lists(data)
    if not lists:
        return None
    # Shallow copies so the caller's data is untouched.
    data = {**data}
    if isinstance(data.get("tables"), list):
        data["tables"] = [dict(t) if isinstance(t, dict) else t for t in data["tables"]]
    for key in ("rows", "records"):
        if isinstance(data.get(key), list):
            data[key] = list(data[key])
    lists = _row_lists(data)
    total = sum(len(rows) for rows in lists)
    while lists and _size(data) > cap:
        longest = max(lists, key=len)
        if not longest:
            break
        longest.pop()
    kept = sum(len(rows) for rows in lists)
    if _size(data) > cap:
        return None
    return (
        data,
        f"response truncated to fit {cap} characters: kept {kept} of {total} rows",
    )


def cap_lists(node: Any, limit: int, elided: dict[str, int], path: str = "") -> Any:
    """Copy of ``node`` with every list / dict cut to its first ``limit`` items;
    ``elided[path]`` records the original length of each cut one."""
    if isinstance(node, list):
        if len(node) > limit:
            elided[path] = max(elided.get(path, 0), len(node))
        return [cap_lists(v, limit, elided, f"{path}[]") for v in node[:limit]]
    if isinstance(node, dict):
        items = list(node.items())
        if len(items) > limit:
            elided[path] = max(elided.get(path, 0), len(items))
        return {
            k: cap_lists(v, limit, elided, f"{path}.{k}" if path else str(k))
            for k, v in items[:limit]
        }
    return node


def _cut_strings(node: Any, limit: int) -> Any:
    if isinstance(node, str):
        return node[:limit]
    if isinstance(node, list):
        return [_cut_strings(v, limit) for v in node]
    if isinstance(node, dict):
        return {k: _cut_strings(v, limit) for k, v in node.items()}
    return node


def _shrink(data: Any, cap: int) -> tuple[Any, str]:
    """Cut lists / dicts (then long strings) until ``data`` fits ``cap``; the
    result stays a JSON value, never a JSON string inside JSON."""
    out: Any = data
    elided: dict[str, int] = {}
    for limit in _SHRINK_LIMITS:
        elided = {}
        out = cap_lists(data, limit, elided)
        if _size(out) <= cap:
            break
    for limit in _SHRINK_STRINGS:
        if _size(out) <= cap:
            break
        out = _cut_strings(out, limit)
    if _size(out) > cap:
        return None, f"response over {cap} characters even when shrunk; narrow the request"
    cut = ", ".join(f"{p or '<top>'} ({n})" for p, n in sorted(elided.items()))
    return out, f"response shrunk to fit {cap} characters; lists cut (original length): {cut}"


def frame(data: Any, *, identity: str | None = None) -> dict:
    """``{identity, data}``; over the size cap, ``data`` is shrunk and flagged."""
    cap = max_response_chars()
    if _size(data) <= cap:
        return {"identity": identity, "data": data}
    trimmed = _trim_rows(data, cap)
    if trimmed is None:
        trimmed = _shrink(data, cap)
    out, note = trimmed
    return {"identity": identity, "data": out, "truncated": True, "note": note}


def compact_preview(data: dict, detail: bool = False) -> dict:
    """A ``preview_step`` result sized for an agent: ``changed``, ``removed_rids``
    and the fitted ``state`` cut to a sample (``changed_total`` keeps the
    count), column lists capped; ``elided`` maps each cut path to its length."""
    limit = PREVIEW_DETAIL_ITEMS if detail else PREVIEW_ITEMS
    elided: dict[str, int] = {}
    out = dict(data)
    for key in ("changed", "removed_rids", "state"):
        if key in out:
            out[key] = cap_lists(out[key], limit, elided, key)
    for key in ("columns", "added_columns", "removed_columns"):
        if key in out:
            out[key] = cap_lists(out[key], PREVIEW_COLUMNS, elided, key)
    if elided:
        out["elided"] = elided
    return out


_PROFILE_LISTS = {"top_values": 3, "sentinel_candidates": 3}
_DETAIL_HINT = "compact form; pass detail: true (or columns) for the full result"


def compact_profiles(data: dict) -> dict:
    """``column_profiles`` without histograms / bounds / suggestions: per column
    the non-null scalar fields, the 3 top values and the 3 sentinel candidates."""
    columns = []
    for col in data.get("columns") or []:
        slim = {
            k: v for k, v in col.items() if v is not None and not isinstance(v, dict | list)
        }
        for key, cap in _PROFILE_LISTS.items():
            if col.get(key):
                slim[key] = col[key][:cap]
        columns.append(slim)
    return {**data, "columns": columns, "note": _DETAIL_HINT}


def compact_align(data: dict) -> dict:
    """``align_report`` with the matching columns reduced to their names."""
    columns = data.get("columns") or []
    matched = [c["train"]["name"] for c in columns if c.get("status") == "match"]
    others = [c for c in columns if c.get("status") != "match"]
    return {**data, "columns": others, "matched": matched, "note": _DETAIL_HINT}


def compact_catalog(items: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """``list_keys`` / ``list_transforms`` reduced to ``keys`` (true flags only)."""
    return [{k: item[k] for k in keys if item.get(k) not in (None, False)} for item in items]


def check_command(cmd_type: str) -> None:
    if cmd_type not in UI_COMMANDS:
        raise PolicyError(f"command not allowed for the agent: {cmd_type}")


def _summarize_dict(d: dict) -> Any:
    picked = {
        k: d.get(k) for k in ("kind", "path", "workspace") if d.get(k) is not None
    }
    return picked or f"<dict {len(d)} keys>"


def _summarize(value: Any) -> Any:
    if isinstance(value, str):
        return value[:120]
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, dict):
        return _summarize_dict(value)
    if isinstance(value, list | tuple):
        return f"<list {len(value)}>"
    return f"<{type(value).__name__}>"


def _summarize_args(args: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in args.items():
        if any(s in str(key).lower() for s in _SECRET_KEYS):
            continue
        if key == "params" and isinstance(value, dict):
            out[key] = {
                k: _summarize(v)
                for k, v in value.items()
                if not any(s in str(k).lower() for s in _SECRET_KEYS)
            }
        else:
            out[key] = _summarize(value)
    return out


class AuditLog:
    """In-memory ring of the last 500 tool calls (summaries only, never data)."""

    def __init__(self) -> None:
        self._entries: deque[dict] = deque(maxlen=500)

    def record(
        self,
        tool: str,
        args: dict,
        *,
        status: str,
        identity: str | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> dict:
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "tool": tool,
            "args": _summarize_args(args),
            "status": status,
            "identity": identity,
            "error": error,
            "duration_ms": duration_ms,
        }
        self._entries.append(entry)
        self._write(entry)
        return entry

    def entries(self) -> list[dict]:
        return list(self._entries)

    @staticmethod
    def _write(entry: dict) -> None:
        if os.environ.get("DTK_AGENT_LOG", "").lower() not in _TRUTHY:
            return
        try:
            folder = dtk_home() / "agent"
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            with (folder / "log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass


__all__ = [
    "DEFAULT_ROWS",
    "MAX_RESPONSE_CHARS",
    "MAX_ROWS",
    "UI_COMMANDS",
    "AuditLog",
    "PolicyError",
    "allowed_roots",
    "cap_lists",
    "check_args",
    "check_command",
    "compact_align",
    "compact_catalog",
    "compact_preview",
    "compact_profiles",
    "compact_result",
    "control_dir",
    "frame",
    "row_limit",
    "workspace_paths",
]
