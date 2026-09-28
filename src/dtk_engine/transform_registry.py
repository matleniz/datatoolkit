"""Transform registry: the @transform decorator and the fit / apply protocol.

A transform is registered on its ``apply`` function::

    class Params(TransformParams):
        columns: list[str]

    def _fit(df: pd.DataFrame, params: Params) -> dict:   # optional
        return {"means": ...}                              # JSON-safe state

    @transform("my_op", params_model=Params, fit=_fit)
    def my_op(df: pd.DataFrame, params: Params, state: dict) -> pd.DataFrame:
        ...

- ``fit(df, params) -> state`` learns what the op needs (a JSON-safe dict, so a
  fitted pipeline can be stored and replayed); omitted -> stateless, state ``{}``.
- ``apply(df, params, state) -> df`` returns a new frame, never mutates ``df``.

Params are a strict pydantic model (unknown / misspelled params raise), so a
front builds the op's form from its JSON Schema. Ops live in
``dtk_engine/ops/transforms/``; workspace replay is in
``dtk_engine/workspace/replay.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from dtk_engine.errors import (
    UnknownTransformError,
    key_params_from_validation,
)
from dtk_engine.params import KeyParams

FitFn = Callable[[pd.DataFrame, Any], dict]
ApplyFn = Callable[[pd.DataFrame, Any, dict], pd.DataFrame]


class TransformParams(KeyParams):
    """Strict base for every transform's params (extra="forbid")."""


def _no_fit(df: pd.DataFrame, params: BaseModel) -> dict:
    return {}


@dataclass(frozen=True)
class Transform:
    op: str
    title: str
    description: str
    params_model: type[BaseModel]
    fit_fn: FitFn
    apply_fn: ApplyFn
    # Fit reads the target column named by ``params.target`` (supervised op):
    # ``DtkTransformer.fit(X, y)`` joins y under that name for fit only.
    needs_target: bool = False

    def parse(self, params: dict | BaseModel) -> BaseModel:
        """Validate raw params; invalid -> KeyParamsError naming the op."""
        if isinstance(params, self.params_model):
            return params
        try:
            return self.params_model.model_validate(params)
        except ValidationError as exc:
            raise key_params_from_validation(
                exc, prefix=f"transform {self.op!r}: "
            ) from exc

    def fit(self, df: pd.DataFrame, params: BaseModel) -> dict:
        """Learn the state on ``df``; it must be JSON-serializable."""
        state = self.fit_fn(df, params)
        try:
            json.dumps(state)
        except TypeError as exc:
            raise TypeError(
                f"transform {self.op!r}: fit must return a JSON-safe dict ({exc})"
            ) from exc
        return state

    def apply(self, df: pd.DataFrame, params: BaseModel, state: dict) -> pd.DataFrame:
        return self.apply_fn(df, params, state)

    def fit_apply(self, df: pd.DataFrame, params: BaseModel) -> pd.DataFrame:
        return self.apply(df, params, self.fit(df, params))


_TRANSFORMS: dict[str, Transform] = {}


def transform(
    op: str,
    *,
    params_model: type[BaseModel],
    fit: FitFn | None = None,
    title: str | None = None,
    description: str | None = None,
    needs_target: bool = False,
):
    """Register ``apply(df, params, state) -> df`` (and an optional ``fit``) as ``op``.

    ``title`` defaults to the op name, ``description`` to the first docstring line.
    ``needs_target``: fit uses the column named by the ``target`` param (it must
    then be a param of ``params_model``).
    """
    if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
        raise TypeError(f"transform {op!r}: params_model must be a pydantic model")

    if needs_target and "target" not in params_model.model_fields:
        raise TypeError(f"transform {op!r}: needs_target requires a 'target' param")

    def decorator(fn: ApplyFn) -> ApplyFn:
        if op in _TRANSFORMS:
            raise ValueError(f"duplicate transform op {op!r}")
        doc = (fn.__doc__ or "").strip().splitlines()
        _TRANSFORMS[op] = Transform(
            op=op,
            title=title or op.replace("_", " ").capitalize(),
            description=description or (doc[0] if doc else ""),
            params_model=params_model,
            fit_fn=fit or _no_fit,
            apply_fn=fn,
            needs_target=needs_target,
        )
        return fn

    return decorator


def get_transform(op: str) -> Transform:
    try:
        return _TRANSFORMS[op]
    except KeyError:
        raise UnknownTransformError(
            f"unknown transform op {op!r}; registered: {sorted(_TRANSFORMS)}"
        ) from None


def all_transforms() -> list[Transform]:
    return [_TRANSFORMS[op] for op in sorted(_TRANSFORMS)]
