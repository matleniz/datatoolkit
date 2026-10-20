"""Engine errors raised across the contract."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class UnknownKeyError(KeyError):
    """The requested key id is not registered."""

    def __str__(self) -> str:  # KeyError would show the repr (quoted message)
        return str(self.args[0]) if self.args else ""


_LIST_CAP = 20


def _near(name: object, names: list[str]) -> str | None:
    from rapidfuzz import process

    hit = process.extractOne(str(name), names, score_cutoff=70)
    return hit[0] if hit else None


def valid_names_hint(
    label: str, names: list[str], wanted: object = None, *, cap: int = _LIST_CAP
) -> str:
    """``"; did you mean 'x'? <label>: a, b, ... (N total)"`` for an error message.

    A close match to ``wanted`` comes first; the list is capped at ``cap``.
    """
    names = [str(n) for n in names]
    close = _near(wanted, names) if wanted is not None else None
    if close is not None:
        names = [close] + [n for n in names if n != close]
    shown = ", ".join(names[:cap])
    if len(names) > cap:
        shown += f", ... ({len(names)} total)"
    lead = f"; did you mean {close!r}?" if close is not None else ";"
    return f"{lead} {label}: {shown}"


class StaleWriteError(Exception):
    """A workspace write carries a sequence older than one already accepted
    (datatoolkit-issues#263). ``detail`` is ``{rule: "stale_write", last_seq}``."""

    def __init__(self, name: str, seq: int, last_seq: int) -> None:
        super().__init__(
            f"stale write to workspace {name!r}: sequence {seq} is older than "
            f"the last accepted one ({last_seq}); nothing written"
        )
        self.detail = {"rule": "stale_write", "last_seq": last_seq}


class KeyParamsError(ValueError):
    """The params passed to a key failed validation.

    Optional ``details`` carries structured pydantic issues
    (``[{loc, msg, type}, ...]``) so HTTP can expose them without the dump.
    Optional ``detail`` is a machine-readable cause (``{rule, group_id,
    step_id}`` for a step-group invariant), exposed as-is under ``detail``.
    """

    def __init__(
        self,
        message: str,
        *,
        details: list[dict[str, Any]] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.detail = detail


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
    exc: ValidationError,
    *,
    prefix: str | None = None,
    detail: dict[str, Any] | None = None,
) -> KeyParamsError:
    """Wrap a pydantic ``ValidationError`` as a concise ``KeyParamsError``
    (``detail``: the machine-readable cause, passed through)."""
    details = validation_error_details(exc)
    message = message_from_validation_details(details)
    if prefix:
        message = f"{prefix}{message}"
    return KeyParamsError(message, details=details, detail=detail)


MAX_LISTED_COLUMNS = 10


def duplicate_columns_message(names: list, where: str) -> str:
    """``"duplicate column names ['a'] in <where> (rename them ...)"``, shared by
    the csv reader (refuses at read time) and the parquet export."""
    names = list(dict.fromkeys(names))
    shown = names[:MAX_LISTED_COLUMNS]
    more = f" (and {len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return f"duplicate column names {shown}{more} in {where} (rename them to be unique)"
