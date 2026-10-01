"""preview_step latency (MAT-212): only the pending step is fitted per call."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from dtk_engine import preview_step
from dtk_engine.sources import registry
from dtk_engine.transform_registry import Transform
from dtk_engine.workspace import dataset
from dtk_engine.workspace.inspect import _diff_cells, _raw_role
from dtk_engine.workspace.models import Step, Workspace
from dtk_engine.workspace.replay import replay_fitted

N_ROWS = 50_000
# Measured ~0.02 s warm at 50k rows after the fix, ~0.08 s before; the bound
# leaves room for slow CI while still catching a return to full replays at
# larger sizes or a pathological regression.
PREVIEW_BUDGET_S = 0.5

SAVED_STEPS = [
    {"op": "impute", "target": "both", "params": {"columns": ["age"]}},
    {
        "op": "impute",
        "target": "both",
        "params": {"columns": ["fare"], "strategy": "mean", "add_indicator": True},
    },
    {
        "op": "impute",
        "target": "both",
        "params": {"columns": ["city"], "strategy": "most_frequent"},
    },
    {"op": "scale", "target": "both", "params": {"columns": ["score"]}},
    {"op": "onehot", "target": "both", "params": {"columns": ["cat"]}},
]
PENDING = {
    "op": "impute",
    "target": "both",
    "params": {"columns": ["city"], "strategy": "constant", "fill_value": "X"},
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    registry._RAW_CACHE.clear()
    dataset._FRAMES.clear()
    yield
    registry._RAW_CACHE.clear()
    dataset._FRAMES.clear()


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "age": np.where(rng.random(n) < 0.1, np.nan, rng.normal(40, 10, n)),
            "fare": np.where(rng.random(n) < 0.1, np.nan, rng.exponential(30, n)),
            "city": rng.choice(["Paris", "Lyon", "Nice", None], n),
            "cat": rng.choice(list("abcdefgh"), n),
            "score": rng.normal(0, 1, n),
            "label": rng.integers(0, 2, n),
        }
    )


@pytest.fixture
def ws(tmp_path) -> dict:
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _frame(N_ROWS, 0).to_csv(train, index=False)
    _frame(N_ROWS // 4, 1).to_csv(test, index=False)
    return {
        "name": "perf",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(train)}, "target_column": "label"},
            "test": {"x": {"kind": "csv", "path": str(test)}, "target_column": "label"},
        },
        "steps": SAVED_STEPS,
    }


def _full_replay_preview(ws: dict, step: dict, role: str) -> tuple[pd.DataFrame, dict]:
    """The pre-MAT-212 path: replay every step from raw for the 'after' frame."""
    w = Workspace.model_validate(ws)
    steps = [*w.steps, Step.model_validate(step)]
    test_raw = _raw_role(w, "test") if w.datasets.test is not None else None
    tr, te, fitted = replay_fitted(steps, _raw_role(w, "train"), test_raw)
    return (tr if role == "train" else te), fitted[-1]


@pytest.mark.parametrize("role", ["train", "test"])
@pytest.mark.parametrize("target", ["train", "test", "both"])
def test_preview_matches_full_replay(ws, role, target):
    step = {**PENDING, "target": target}
    out = preview_step(ws, step, role)
    after, last = _full_replay_preview(ws, step, role)
    assert out["shape"] == [len(after), after.shape[1]]
    assert out["columns"] == [str(c) for c in after.columns]
    assert out["fitted_on"] == last["fitted_on"]
    assert out["state"] == json.loads(json.dumps(last["state"], default=str))


def test_preview_fits_only_the_pending_step(ws, monkeypatch):
    preview_step(ws, PENDING, "train")  # warm the replay cache
    fits = []
    real = Transform.fit

    def counting(self, df, params):
        fits.append(self.op)
        return real(self, df, params)

    monkeypatch.setattr(Transform, "fit", counting)
    preview_step(ws, PENDING, "train")
    preview_step(ws, PENDING, "test")
    assert fits == ["impute", "impute"]


def test_preview_step_latency(ws):
    preview_step(ws, PENDING, "train")  # cold: parse + replay, cached afterwards
    for role in ("train", "test"):
        start = time.perf_counter()
        preview_step(ws, {**PENDING, "params": {**PENDING["params"], "fill_value": role}}, role)
        elapsed = time.perf_counter() - start
        assert elapsed < PREVIEW_BUDGET_S, f"preview_step {role}: {elapsed:.3f}s"


def test_diff_cells_fast_path_matches_reindexed_path():
    before = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", None]})
    after = before.assign(a=[1.0, 2.0, 3.0], b=["x", "z", None])
    fast = _diff_cells(before, after)
    # A shuffled index forces the set / reindex path.
    slow = _diff_cells(before.iloc[::-1], after.iloc[::-1])
    assert fast == slow
    assert fast[1] == 2 and fast[2] == []
