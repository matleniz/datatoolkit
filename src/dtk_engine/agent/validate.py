"""Check a tool call's arguments against the tool's ``input_schema``.

The SDK's low-level ``Server`` does not validate ``tools/call`` arguments, so
``server.call_tool`` runs this before the handler. A violation raises
``KeyParamsError`` with ``details`` (``[{loc, msg, type}, ...]``), which the
server returns as an ``is_error`` result: the agent reads it and corrects the
call (MCP spec 2025-11-25, SEP-1303).

A small validator of our own (``jsonschema`` is only a transitive dependency of
``mcp``, not one of ours). It covers exactly the JSON Schema keywords the
registered tools use, see ``SUPPORTED_KEYWORDS``; ``unsupported_keywords`` lets a
test fail when a tool schema uses one that is not covered (it would otherwise
be silently unchecked).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from dtk_engine.errors import KeyParamsError, message_from_validation_details

MAX_ISSUES = 5
_MAX_REPR = 60
Issue = dict[str, Any]

_JSON_NAMES = {
    dict: "object", list: "array", str: "string", bool: "boolean",
    int: "integer", float: "number", type(None): "null",
}
_TYPE_TESTS: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "integer": lambda v: (
        isinstance(v, int | float) and not isinstance(v, bool)
        and (isinstance(v, int) or v.is_integer())
    ),
}


def _json_name(value: Any) -> str:
    return _JSON_NAMES.get(type(value), type(value).__name__)


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _MAX_REPR else text[: _MAX_REPR - 3] + "..."


def _issue(path: list, msg: str, kind: str) -> Issue:
    return {"loc": path, "msg": msg, "type": kind}


def _is_number(value: Any) -> bool:
    return _TYPE_TESTS["number"](value)


def _type(limit: Any, value: Any, path: list) -> list[Issue]:
    names = limit if isinstance(limit, list) else [limit]
    if any(_TYPE_TESTS[n](value) for n in names):
        return []
    return [_issue(path, f"must be {' or '.join(names)}, got {_json_name(value)}", "type")]


def _enum(limit: list, value: Any, path: list) -> list[Issue]:
    if any(value == v and type(value) is type(v) for v in limit):
        return []
    allowed = ", ".join(repr(v) for v in limit)
    return [_issue(path, f"must be one of {allowed}, got {_short(value)}", "enum")]


def _bound(kind: str, op: str, ok: Callable[[Any, Any], bool]) -> Callable:
    def check(limit: Any, value: Any, path: list) -> list[Issue]:
        if not _is_number(value) or ok(value, limit):
            return []
        return [_issue(path, f"must be {op} {limit}, got {value}", kind)]

    return check


def _size(kind: str, what: str, accepts: type | tuple, op: str, ok: Callable) -> Callable:
    def check(limit: Any, value: Any, path: list) -> list[Issue]:
        if not isinstance(value, accepts) or ok(len(value), limit):
            return []
        return [_issue(path, f"must have {op} {limit} {what}, got {len(value)}", kind)]

    return check


def _pattern(limit: str, value: Any, path: list) -> list[Issue]:
    if not isinstance(value, str) or re.search(limit, value):
        return []
    return [_issue(path, f"must match {limit}, got {_short(value)}", "pattern")]


def _required(limit: list, value: Any, path: list) -> list[Issue]:
    if not isinstance(value, dict):
        return []
    missing = [k for k in limit if k not in value]
    if not missing:
        return []
    return [_issue(path, "missing required argument " + ", ".join(map(repr, missing)),
                   "required")]


def _properties(limit: dict, value: Any, path: list) -> list[Issue]:
    if not isinstance(value, dict):
        return []
    return [
        issue for name, sub in limit.items() if name in value
        for issue in _validate(sub, value[name], [*path, name])
    ]


def _additional(limit: Any, value: Any, path: list, schema: dict) -> list[Issue]:
    if limit is not False or not isinstance(value, dict):
        return []
    known = schema.get("properties", {})
    extra = sorted(k for k in value if k not in known)
    if not extra:
        return []
    allowed = ", ".join(sorted(known))
    return [_issue(path, f"unknown argument {', '.join(map(repr, extra))} (allowed: {allowed})",
                   "additionalProperties")]


def _items(limit: dict, value: Any, path: list) -> list[Issue]:
    if not isinstance(value, list):
        return []
    return [i for n, item in enumerate(value) for i in _validate(limit, item, [*path, n])]


def _any_of(limit: list, value: Any, path: list) -> list[Issue]:
    branches = [_validate(sub, value, path) for sub in limit]
    if any(not b for b in branches):
        return []
    return min(branches, key=len)  # the closest branch explains the miss


_CHECKS: dict[str, Callable[[Any, Any, list], list[Issue]]] = {
    "type": _type,
    "enum": _enum,
    "minimum": _bound("minimum", ">=", lambda v, lim: v >= lim),
    "maximum": _bound("maximum", "<=", lambda v, lim: v <= lim),
    "minLength": _size("minLength", "characters", str, "at least", lambda n, lim: n >= lim),
    "maxLength": _size("maxLength", "characters", str, "at most", lambda n, lim: n <= lim),
    "minItems": _size("minItems", "items", list, "at least", lambda n, lim: n >= lim),
    "maxItems": _size("maxItems", "items", list, "at most", lambda n, lim: n <= lim),
    "minProperties": _size("minProperties", "properties", dict, "at least",
                           lambda n, lim: n >= lim),
    "pattern": _pattern,
    "required": _required,
    "properties": _properties,
    "items": _items,
    "anyOf": _any_of,
}
# ``additionalProperties`` needs the sibling ``properties``: handled in _validate.
_ANNOTATIONS = frozenset({"description", "title"})
SUPPORTED_KEYWORDS = frozenset({*_CHECKS, "additionalProperties", *_ANNOTATIONS})


def _validate(schema: dict, value: Any, path: list) -> list[Issue]:
    if "type" in schema:
        issues = _type(schema["type"], value, path)
        if issues:
            return issues  # the other keywords assume the right type
    out: list[Issue] = []
    for keyword, check in _CHECKS.items():
        if keyword in schema and keyword != "type":
            out.extend(check(schema[keyword], value, path))
    if "additionalProperties" in schema:
        out.extend(_additional(schema["additionalProperties"], value, path, schema))
    return out


def unsupported_keywords(schema: Any) -> set[str]:
    """Keywords used in ``schema`` (recursively) that ``check_arguments`` does not enforce."""
    if not isinstance(schema, dict):
        return set()
    found = {k for k in schema if k not in SUPPORTED_KEYWORDS}
    subs: list[Any] = [*schema.get("properties", {}).values(), *schema.get("anyOf", [])]
    if isinstance(schema.get("items"), dict):
        subs.append(schema["items"])
    if isinstance(schema.get("additionalProperties"), dict):
        found.add("additionalProperties(schema)")
    return found.union(*(unsupported_keywords(s) for s in subs))


def strict_schema(schema: dict) -> dict:
    """``schema`` with top-level ``additionalProperties: false`` unless it says otherwise."""
    return {"additionalProperties": False, **schema}


def make_validator(schema: dict) -> dict:
    """What ``check_arguments`` takes: the strict schema, ``session`` (read by every tool)
    always allowed."""
    strict = strict_schema(schema)
    props = {"session": {"type": "string"}, **strict.get("properties", {})}
    return {**strict, "properties": props}


def check_arguments(validator: dict, args: Any) -> None:
    """Raise ``KeyParamsError`` (with ``details``) when ``args`` violates the schema."""
    issues = _validate(validator, args, [])
    if not issues:
        return
    shown = sorted(issues, key=lambda i: [str(p) for p in i["loc"]])[:MAX_ISSUES]
    message = message_from_validation_details(shown)
    if len(issues) > MAX_ISSUES:
        message += f"; and {len(issues) - MAX_ISSUES} more"
    raise KeyParamsError(message, details=shown)
