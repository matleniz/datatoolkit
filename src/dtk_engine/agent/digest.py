"""What the agent last saw of a workspace, and what changed since.

``StepTracker`` keeps, per (Studio session, workspace), the step list the agent
last observed (ids + content). Each tool call diffs the stored steps against
it, so the agent learns about user-side edits (``workspace_changes``) and
``propose_steps`` can send ``base_steps`` (datatoolkit-issues#153).

``frame_identity`` computes Studio's data identity engine-side for frames not
on screen (``src/bench/dataIdentity.ts``: FNV-1a over a JSON of sources, label,
merges and ``steps[0:version]``); best effort, Studio's own identity wins when
it describes the same frame.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from decimal import Decimal
from typing import Any

_MAX_KEYS = 64  # (session, workspace) pairs kept; the oldest is dropped
_STEP_KEYS = ("op", "target", "params")

Key = tuple[str, str]


def _content(step: dict) -> dict:
    return {k: step.get(k) for k in _STEP_KEYS}


class StepTracker:
    def __init__(self) -> None:
        self._seen: OrderedDict[Key, list[dict]] = OrderedDict()
        self._pending: dict[Key, set[str]] = {}

    def clear(self) -> None:
        self._seen.clear()
        self._pending.clear()

    def observe(self, key: Key, steps: list[dict]) -> None:
        self._seen[key] = [{"id": s.get("id"), **_content(s)} for s in steps]
        self._seen.move_to_end(key)
        while len(self._seen) > _MAX_KEYS:
            old, _ = self._seen.popitem(last=False)
            self._pending.pop(old, None)

    def seen(self, key: Key) -> dict[str, dict] | None:
        """``{id: {op, target, params}}`` last observed, None if never."""
        steps = self._seen.get(key)
        return None if steps is None else {s["id"]: _content(s) for s in steps}

    def changes(self, key: Key, steps: list[dict]) -> dict | None:
        """Steps added / removed / changed since the last observation (None on
        the first one or when nothing changed); ``steps`` becomes observed."""
        before = self.seen(key)
        self.observe(key, steps)
        if before is None:
            return None
        now = {s["id"]: _content(s) for s in steps}
        out = {
            "added": [{"id": i, "op": c["op"]} for i, c in now.items() if i not in before],
            "removed": [{"id": i, "op": c["op"]} for i, c in before.items() if i not in now],
            "changed": [
                {"id": i, "op": c["op"]}
                for i, c in now.items()
                if i in before and not _same(before[i], c)
            ],
        }
        return {k: v for k, v in out.items() if v} or None

    def add_pending(self, key: Key, cid: str) -> None:
        self._pending.setdefault(key, set()).add(cid)

    def pending(self, key: Key) -> set[str]:
        return set(self._pending.get(key, ()))

    def settle(self, key: Key, cid: str) -> None:
        self._pending.get(key, set()).discard(cid)


def _same(a: Any, b: Any) -> bool:
    """JSON value equality: Python ``==`` ignores key order and has ``1 == 1.0``."""
    return a == b


TRACKER = StepTracker()


# -- per-turn workspace note (datatoolkit-issues#151) -------------------------

_NOTE_STEPS = 80  # steps listed in a turn note
_STEP_LINE = 90  # characters per step summary
_LIST_ITEMS = 3


def _param_text(value: Any) -> str | None:
    if isinstance(value, list) and all(isinstance(v, str | int | float) for v in value):
        more = f"+{len(value) - _LIST_ITEMS}" if len(value) > _LIST_ITEMS else ""
        return ",".join(str(v) for v in value[:_LIST_ITEMS]) + more
    if isinstance(value, str | int | float | bool):
        return str(value)
    return None  # nested params: see get_workspace


def step_line(step: dict) -> str:
    """``s3 impute columns=age strategy=median`` (cut to a line)."""
    parts = [f"{step.get('id')} {step.get('op')}"]
    if step.get("note"):
        parts.append("(note)")
    if step.get("target") not in (None, "both"):
        parts.append(f"[{step['target']}]")
    for key, value in (step.get("params") or {}).items():
        text = _param_text(value)
        if text is not None:
            parts.append(f"{key}={text}")
    line = " ".join(parts)
    return line if len(line) <= _STEP_LINE else line[: _STEP_LINE - 1] + "…"


def _changes_text(changes: dict, steps: list[dict]) -> str:
    """Added / changed steps as full lines, removed ones by id."""
    by_id = {s.get("id"): s for s in steps}
    out = [
        f"{kind} {step_line(by_id[c['id']])}"
        for kind in ("added", "changed")
        for c in changes.get(kind) or []
        if c["id"] in by_id
    ]
    out.extend(f"removed {c['id']} ({c['op']})" for c in changes.get("removed") or [])
    return "; ".join(out)


def turn_note(ctx: dict, steps: list[dict], changes: dict | None, first: bool) -> str:
    """The short Studio state prepended to a user turn: what the user sees, the
    whole step list on the first turn, then only what changed since the agent
    last looked (one line per added / changed step)."""
    view = (
        f'[Studio: workspace "{ctx.get("workspace")}", role {ctx.get("role") or "train"}, '
        f"viewing version {ctx.get('version')} of {len(steps)}, identity {ctx.get('identity')}."
    )
    lines = [view]
    if first:
        shown = steps[:_NOTE_STEPS]
        more = len(steps) - len(shown)
        listed = "; ".join(step_line(s) for s in shown)
        tail = f"; … {more} more (get_workspace)" if more else ""
        lines.append(f"Steps (id op params): {listed}{tail}" if steps else "No steps yet.")
    elif changes:
        lines.append(
            f"Steps changed since you last looked ({len(steps)} now): "
            f"{_changes_text(changes, steps)}."
        )
    else:
        lines.append(f"Steps unchanged since you last looked ({len(steps)}).")
    return "\n".join(lines) + "]\n\n"


# -- engine-side data identity (mirrors Studio's dataIdentity.ts) -------------


def _js_number(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return "null"
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    text = repr(x)
    if "e" in text and 1e-6 <= abs(x) < 1e21:  # JS: fixed notation there
        return format(Decimal(text), "f")
    return text.replace("e-0", "e-")


def js_json(value: Any) -> str:
    """``JSON.stringify`` of a JSON value (no spaces, JS number formatting)."""
    if isinstance(value, float):
        return _js_number(value)
    if isinstance(value, dict):
        return "{" + ",".join(f"{js_json(str(k))}:{js_json(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ",".join(js_json(v) for v in value) + "]"
    if value is None or isinstance(value, str | bool | int):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value))


def fnv1a(text: str) -> str:
    """Studio's FNV-1a 32-bit over UTF-16 code units, 8 hex digits."""
    h = 0x811C9DC5
    data = text.encode("utf-16-le")
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def frame_identity(ws: dict, role: str, version: int | None) -> str:
    """``<workspace>|<role>|v<N>|<hash>`` of ``ws`` at ``version`` (None = all)."""
    steps = ws.get("steps") or []
    last = len(steps)
    v = last if version is None or version > last else max(version, 0)
    payload = {
        "datasets": ws.get("datasets"),
        "label": ws.get("label"),
        "merges": ws.get("merges") or [],
        "steps": [_content(s) for s in steps[:v]],
    }
    return f"{ws.get('name')}|{role}|v{v}|{fnv1a(js_json(payload))}"
