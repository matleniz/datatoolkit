"""The engine <-> front boundary. Everything in and out is plain JSON."""

from __future__ import annotations

from pydantic import ValidationError

from . import keys  # noqa: F401  (registers every key)
from .errors import KeyParamsError
from .registry import all_keys, get_key


def list_keys() -> list[dict]:
    return [
        {
            "id": k.id,
            "title": k.title,
            "category": k.category,
            "description": k.description,
        }
        for k in all_keys()
    ]


def key_schema(key_id: str) -> dict:
    return get_key(key_id).params_model.model_json_schema()


def run_key(key_id: str, params: dict) -> dict:
    k = get_key(key_id)
    try:
        parsed = k.params_model.model_validate(params)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    return k.run(parsed).model_dump(mode="json")
