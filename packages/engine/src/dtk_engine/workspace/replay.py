"""Transform registry and step replay: current state = sources + steps in order.

A transform is registered with ``@transform(op)`` and has the signature::

    fn(df: pd.DataFrame, params: dict, fit: pd.DataFrame | None) -> pd.DataFrame

``fit`` is the frame the op may learn statistics from (fit-on-train):

- applied to train (target "train" or "both"): ``fit is None`` -> fit on ``df``;
- applied to test by a "both" step: ``fit`` is the train frame as it was right
  before this step (earlier train / both steps already replayed), so the op can
  apply train statistics to test;
- applied to test by a "test"-only step: ``fit is None``.

No op is registered yet; replaying an unknown op raises a SourceError.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.workspace.models import Step

Transform = Callable[[pd.DataFrame, dict, "pd.DataFrame | None"], pd.DataFrame]

_TRANSFORMS: dict[str, Transform] = {}


def transform(op: str):
    """Register ``fn(df, params, fit) -> df`` as the transform ``op``."""

    def decorator(fn: Transform) -> Transform:
        if op in _TRANSFORMS:
            raise ValueError(f"duplicate transform op {op!r}")
        _TRANSFORMS[op] = fn
        return fn

    return decorator


def get_transform(op: str) -> Transform:
    try:
        return _TRANSFORMS[op]
    except KeyError:
        raise SourceError(
            f"unknown transform op {op!r}; registered: {sorted(_TRANSFORMS)}"
        ) from None


def all_transforms() -> list[str]:
    return sorted(_TRANSFORMS)


def needs_train(steps: Iterable[Step]) -> bool:
    """True when replaying for test needs the train frame (a "both" step fits on it)."""
    return any(s.target == "both" for s in steps)


def replay(
    steps: list[Step],
    role: str,
    frame: pd.DataFrame,
    train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply the steps that target ``role`` to ``frame``, in log order.

    For ``role="test"`` with "both" steps, pass the untransformed ``train`` frame:
    it is replayed alongside so each "both" step fits on train as of that step.
    """
    fns = [get_transform(s.op) for s in steps]  # fail before doing any work
    if role == "test" and train is None and needs_train(steps):
        raise SourceError("replaying a 'both' step on test needs the train frame")
    for step, fn in zip(steps, fns, strict=True):
        if role == "train":
            if step.target in ("train", "both"):
                frame = fn(frame, step.params, None)
            continue
        fit = train if step.target == "both" else None
        if step.target in ("test", "both"):
            frame = fn(frame, step.params, fit)
        if train is not None and step.target in ("train", "both"):
            train = fn(train, step.params, None)
    return frame
