"""Engine errors raised across the contract."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class UnknownKeyError(KeyError):
    """The requested key id is not registered."""


class KeyParamsError(ValueError):
    """The params passed to a key failed validation.

    Optional ``details`` carries structured pydantic issues
    (``[{loc, msg, type}, ...]``) so HTTP can expose them without the dump.
    """

    def __init__(
        self, message: str, *, details: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(message)
        self.details = details


class SourceError(ValueError):
    """A source could not be loaded (missing file, unreadable content, unknown kind)."""


class UnknownTransformError(KeyError):
    """The requested transform op is not registered."""


def _issue_msg(err: dict[str, Any]) -> str:
    """Prefer the inner ``ctx.error`` (ValueError text) over pydantic's wrapper."""
    ctx = err.get("ctx") or {}
    raw = ctx.get("error")
    if raw is not None:
        return str(raw)
    msg = str(err.get("msg", ""))
    if msg.startswith("Value error, "):
        return msg[len("Value error, ") :]
    return msg


def validation_error_details(exc: ValidationError) -> list[dict[str, Any]]:
    """``[{loc, msg, type}, ...]`` from a pydantic ``ValidationError``."""
    return [
        {
            "loc": list(err.get("loc", ())),
            "msg": _issue_msg(err),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]


def message_from_validation_details(details: list[dict[str, Any]]) -> str:
    """Join ``[{loc, msg, type}, ...]`` as ``"<loc>: <msg>"`` (just ``msg`` when loc is empty)."""
    parts: list[str] = []
    for item in details:
        loc = item.get("loc") or []
        msg = item.get("msg", "")
        if loc:
            parts.append(f"{'.'.join(str(x) for x in loc)}: {msg}")
        else:
            parts.append(str(msg))
    return "; ".join(parts)


def key_params_from_validation(
    exc: ValidationError, *, prefix: str | None = None
) -> KeyParamsError:
    """Wrap a pydantic ``ValidationError`` as a concise ``KeyParamsError``."""
    details = validation_error_details(exc)
    message = message_from_validation_details(details)
    if prefix:
        message = f"{prefix}{message}"
    return KeyParamsError(message, details=details)
