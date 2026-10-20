"""Check a tool call's arguments against the tool's ``input_schema``.

The SDK's low-level ``Server`` does not validate ``tools/call`` arguments, so
``server.call_tool`` runs this before the handler. A violation raises
``KeyParamsError`` with ``details`` (``[{loc, msg, type}, ...]``), which the
server returns as an ``is_error`` result: the agent reads it and corrects the
call (MCP spec 2025-11-25, SEP-1303). ``jsonschema`` ships with ``mcp``.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from dtk_engine.errors import KeyParamsError, message_from_validation_details

MAX_ISSUES = 5
_JSON_TYPES = {
    dict: "object", list: "array", str: "string", bool: "boolean",
    int: "integer", float: "number", type(None): "null",
}


def strict_schema(schema: dict) -> dict:
    """``schema`` with top-level ``additionalProperties: false`` unless it says otherwise."""
    return {"additionalProperties": False, **schema}


def make_validator(schema: dict) -> Draft202012Validator:
    """Validator for ``strict_schema(schema)``; ``session`` (read by every tool) is always allowed."""
    strict = strict_schema(schema)
    props = {"session": {"type": "string"}, **strict.get("properties", {})}
    return Draft202012Validator({**strict, "properties": props})


def _json_type(value: Any) -> str:
    return _JSON_TYPES.get(type(value), type(value).__name__)


def _allowed(types: Any) -> str:
    return " or ".join(types) if isinstance(types, list) else str(types)


def _issue(error: ValidationError) -> dict[str, Any]:
    """One ``{loc, msg, type}`` for a jsonschema error (``loc`` = path of the argument)."""
    if error.validator in ("anyOf", "oneOf") and error.context:
        error = best_match(error.context) or error
    loc = list(error.absolute_path)
    kind, limit, value = error.validator, error.validator_value, error.instance
    if kind == "type":
        msg = f"must be {_allowed(limit)}, got {_json_type(value)}"
    elif kind == "required":
        missing = [k for k in limit if k not in value]
        msg = "missing required argument " + ", ".join(repr(k) for k in missing)
    elif kind == "additionalProperties":
        extra = sorted(k for k in value if k not in error.schema.get("properties", {}))
        allowed = ", ".join(sorted(error.schema.get("properties", {})))
        msg = f"unknown argument {', '.join(repr(k) for k in extra)} (allowed: {allowed})"
    elif kind in ("minimum", "maximum"):
        msg = f"must be {'>=' if kind == 'minimum' else '<='} {limit}, got {value}"
    elif kind == "enum":
        msg = f"must be one of {', '.join(repr(v) for v in limit)}, got {value!r}"
    else:
        msg = error.message
    return {"loc": loc, "msg": msg, "type": str(kind)}


def check_arguments(validator: Draft202012Validator, args: Any) -> None:
    """Raise ``KeyParamsError`` (with ``details``) when ``args`` violates the schema."""
    errors = sorted(validator.iter_errors(args), key=lambda e: [str(p) for p in e.absolute_path])
    if not errors:
        return
    details = [_issue(e) for e in errors[:MAX_ISSUES]]
    message = message_from_validation_details(details)
    if len(errors) > MAX_ISSUES:
        message += f"; and {len(errors) - MAX_ISSUES} more"
    raise KeyParamsError(message, details=details)
