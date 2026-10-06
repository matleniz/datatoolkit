"""Step replay: current state = sources + steps in order.

Each step names a registered transform (``dtk_engine.transform_registry``,
``fit(df, params) -> state`` then ``apply(df, params, state) -> df``):

- target "train": fit and apply on train;
- target "test": fit and apply on test;
- target "both": fit on train as it was right before this step (earlier train /
  both steps already replayed), apply that state to train and to test.

``replay`` (one role), ``replay_fitted`` (both roles + fitted states, for
export) and ``replay_step`` (one more step on frames already replayed, for the
Studio preview) share one loop, ``_run``.

Errors, all raised before any work: an unknown op -> UnknownTransformError,
invalid params -> KeyParamsError (``validate_steps`` runs the same check at
save). An op failing on the data (e.g. dropping a missing column) raises a
SourceError naming the step.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import pandas as pd

from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace.models import Step


def needs_train(steps: Iterable[Step]) -> bool:
    """True when replaying for test needs the train frame (a "both" step fits on it)."""
    return any(s.target == "both" for s in steps)


def resolve_version(n_steps: int, version: int | None) -> int:
    """``version`` (None = every step) -> number of steps to replay.

    Invalid type / negative -> ``KeyParamsError``; ``version`` beyond the
    workspace's step count -> ``KeyParamsError`` naming the count.
    """
    if version is None:
        return n_steps
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise KeyParamsError(f"version must be a non-negative int or None, got {version!r}")
    if version > n_steps:
        raise KeyParamsError(
            f"version {version} exceeds workspace step count ({n_steps})"
        )
    return version


def _resolve(steps: list[Step]):
    transforms = [get_transform(s.op) for s in steps]
    return [(s, t, t.parse(s.params)) for s, t in zip(steps, transforms, strict=True)]


def validate_steps(steps: list[Step]) -> None:
    """Unknown op -> UnknownTransformError; invalid params -> KeyParamsError."""
    _resolve(steps)


@contextmanager
def _step_context(i: int, step: Step, split: str | None = None) -> Iterator[None]:
    try:
        yield
    except (SourceError, KeyParamsError):
        raise
    except (KeyError, ValueError, TypeError) as exc:
        raise SourceError(
            f"step {i} ({step.op!r} on {step.target}) failed"
            f"{f' on the {split} set' if split else ''}: {exc}"
        ) from exc


def _run(
    steps: list[Step],
    train: pd.DataFrame | None,
    test: pd.DataFrame | None,
    train_until: int,
    start: int = 0,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[dict]]:
    """The one replay loop: steps in order over the train and test frames.

    A step fits on train ("train" / "both") or on test ("test") and applies to
    the frames it targets. A missing frame (``None``) skips the steps fitted on
    it. Train is only replayed through step ``train_until``: fitted up to and
    including it, applied strictly before it (later train steps are skipped).
    Returns (train, test, fitted) with per step ``{"fitted_on", "state"}``
    (``None`` / ``{}`` for a skipped step). ``start`` is the log index of
    ``steps[0]`` when the frames already went through the earlier steps
    (indices, including ``train_until``, are log indices).
    """
    fitted = []
    resolved = _resolve(steps)  # fails before any work
    for i, (step, t, params) in enumerate(resolved, start=start):
        fitted_on = "test" if step.target == "test" else "train"
        fit_frame = test if fitted_on == "test" else train
        if fit_frame is None or (fitted_on == "train" and i > train_until):
            fitted.append({"fitted_on": None, "state": {}})
            continue
        with _step_context(i, step):
            state = t.fit(fit_frame, params)
            if step.target != "test" and i < train_until:
                train = t.apply(train, params, state)
            if step.target != "train" and test is not None:
                with _step_context(i, step, split="test"):
                    test = t.apply(test, params, state)
        fitted.append({"fitted_on": fitted_on, "state": state})
    return train, test, fitted


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
    if role == "train":
        return _run(steps, frame, None, len(steps))[0]
    validate_steps(steps)  # an invalid step wins over a missing train frame
    if train is None and needs_train(steps):
        raise SourceError("replaying a 'both' step on test needs the train frame")
    # Train only matters up to the last "both" step.
    last_both = max((i for i, s in enumerate(steps) if s.target == "both"), default=-1)
    return _run(steps, train, frame, last_both)[1]


def replay_fitted(
    steps: list[Step], train: pd.DataFrame, test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[dict]]:
    """Replay every step on train and test in one pass, keeping the fitted states.

    Returns (train, test, fitted): same frames as ``replay`` for each role, and
    per step ``{"fitted_on": "train" | "test" | None, "state": dict}`` (the state
    learned by a "both" / "train" step on train, or by a "test" step on test;
    ``None`` / ``{}`` for a "test" step without a test frame).
    """
    return _run(steps, train, test, len(steps))


def replay_step(
    step: Step,
    index: int,
    train: pd.DataFrame | None,
    test: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Fit and apply ``step`` (log position ``index``) on frames already replayed
    through the ``index`` earlier steps.

    Same result as the last step of ``replay_fitted`` on the whole log, without
    replaying the prefix: callers pass cached prefix frames. A ``None`` frame is
    left alone (and a step fitted on it is skipped). Returns (train, test, fitted).
    """
    train, test, fitted = _run([step], train, test, index + 1, start=index)
    return train, test, fitted[0]
