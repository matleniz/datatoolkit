"""Step replay: current state = sources + steps in order.

Each step names a registered transform (``dtk_engine.transform_registry``,
``fit(df, params) -> state`` then ``apply(df, params, state) -> df``):

- target "train": fit and apply on train;
- target "test": fit and apply on test;
- target "both": fit on train as it was right before this step (earlier train /
  both steps already replayed), apply that state to train and to test.

An unknown op raises a SourceError and invalid params a KeyParamsError, both
before any work. An op failing on the data (e.g. dropping a missing column)
raises a SourceError naming the step.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import pandas as pd

from dtk_engine.errors import KeyParamsError, SourceError, UnknownTransformError
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace.models import Step


def needs_train(steps: Iterable[Step]) -> bool:
    """True when replaying for test needs the train frame (a "both" step fits on it)."""
    return any(s.target == "both" for s in steps)


def _resolve(steps: list[Step]):
    try:
        transforms = [get_transform(s.op) for s in steps]
    except UnknownTransformError as exc:
        raise SourceError(exc.args[0]) from None
    return [(s, t, t.parse(s.params)) for s, t in zip(steps, transforms, strict=True)]


@contextmanager
def _step_context(i: int, step: Step) -> Iterator[None]:
    try:
        yield
    except (SourceError, KeyParamsError):
        raise
    except (KeyError, ValueError, TypeError) as exc:
        raise SourceError(
            f"step {i} ({step.op!r} on {step.target}) failed: {exc}"
        ) from exc


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
    resolved = _resolve(steps)  # fail before doing any work
    if role == "train":
        for i, (step, t, params) in enumerate(resolved):
            if step.target in ("train", "both"):
                with _step_context(i, step):
                    frame = t.fit_apply(frame, params)
        return frame

    if train is None and needs_train(steps):
        raise SourceError("replaying a 'both' step on test needs the train frame")
    # Train only matters up to the last "both" step.
    last_both = max((i for i, s in enumerate(steps) if s.target == "both"), default=-1)
    for i, (step, t, params) in enumerate(resolved):
        with _step_context(i, step):
            if step.target == "both":
                state = t.fit(train, params)
                frame = t.apply(frame, params, state)
                if i < last_both:
                    train = t.apply(train, params, state)
            elif step.target == "test":
                frame = t.fit_apply(frame, params)
            elif i < last_both:
                train = t.fit_apply(train, params)
    return frame
