"""Key registry and the @key decorator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from .errors import UnknownKeyError
from .result import Result


@dataclass(frozen=True)
class Key:
    id: str
    title: str
    category: str
    description: str
    params_model: type[BaseModel]
    run: Callable[[BaseModel], Result]

    @property
    def needs_target(self) -> bool:
        """The key analyses a label: its ``target`` param is a non-nullable string
        (e.g. ``feature_selection``); an optional ``target`` does not count."""
        field = self.params_model.model_fields.get("target")
        return field is not None and field.annotation is str


_REGISTRY: dict[str, Key] = {}


def key(*, id: str, title: str, category: str, description: str):
    """Register ``run(params: Params) -> Result`` as a key."""

    def decorator(fn: Callable[..., Result]) -> Callable[..., Result]:
        params_model = fn.__annotations__.get("params")
        if isinstance(params_model, str):
            params_model = fn.__globals__[params_model]
        if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
            raise TypeError(
                f"key {id!r}: `params` must be annotated with a pydantic model"
            )
        if id in _REGISTRY:
            raise ValueError(f"duplicate key id {id!r}")
        _REGISTRY[id] = Key(id, title, category, description, params_model, fn)
        return fn

    return decorator


def get_key(key_id: str) -> Key:
    try:
        return _REGISTRY[key_id]
    except KeyError:
        raise UnknownKeyError(key_id) from None


def all_keys() -> list[Key]:
    return list(_REGISTRY.values())
