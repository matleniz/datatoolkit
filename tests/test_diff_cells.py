"""``_diff_cells``: vectorised pre-filter must keep JSON round-trip equality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dtk_engine.workspace import inspect as insp
from dtk_engine.workspace.inspect import _diff_cells


def _fixture(n=300):
    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(n, name="_rid")
    before = pd.DataFrame(
        {
            "i": rng.integers(0, 5, n),
            "f": rng.normal(size=n),
            "s": rng.choice(["a", "b", None], n),
            "d": pd.date_range("2024-01-01", periods=n, freq="h"),
            "b": rng.choice([True, False], n),
            "n": np.where(rng.random(n) < 0.3, np.nan, 1.5),
            "ni": pd.array(rng.choice([1, 2, None], n), dtype="Int64"),
            "same": rng.integers(0, 9, n),
            "gone": 1,
        },
        index=idx,
    )
    after = before.drop(columns="gone").drop(index=[3, 10, 77]).copy()
    after["new"] = 1
    after["i"] = after["i"].astype(float)  # dtype-only change
    after.loc[after.index[::7], "f"] += 1.0
    after.loc[after.index[::11], "s"] = "z"
    after.loc[after.index[5:20], "n"] = 2.0
    after.loc[after.index[::13], "d"] += pd.Timedelta(minutes=1)
    after["b"] = after["b"].astype(object)
    after.loc[after.index[::17], "b"] = None
    after.loc[after.index[::9], "ni"] = pd.NA
    return before, after


def test_int_vs_float_same_value_is_not_changed():
    before = pd.DataFrame({"x": [1, 2, 3]}, index=pd.Index([0, 1, 2], name="_rid"))
    after = before.assign(x=before["x"].astype(float))
    assert _diff_cells(before, after) == ([], 0, [])
    after.iloc[1, 0] = 2.5
    changed, total, _ = _diff_cells(before, after)
    assert total == 1
    assert changed == [{"_rid": 1, "column": "x", "before": 2, "after": 2.5}]


def test_nan_to_nan_not_changed_null_to_value_changed():
    before = pd.DataFrame({"x": [np.nan, np.nan, 1.0]})
    after = pd.DataFrame({"x": [np.nan, 4.0, 1.0]})
    changed, total, _ = _diff_cells(before, after)
    assert total == 1 and changed[0]["before"] is None


def test_mixed_dtype_fixture_changes():
    before, after = _fixture()
    changed, total, removed = _diff_cells(before, after)
    assert removed == [3, 10, 77]
    assert total == len(changed)
    cols = {c["column"] for c in changed}
    # dtype-only changes (i int->float, same untouched) never show up
    assert "i" not in cols and "same" not in cols
    assert {"f", "s", "n", "d", "b", "ni"} <= cols
    # row-major order: rid ascending, then column order
    keys = [(c["_rid"], list(before.columns).index(c["column"])) for c in changed]
    assert keys == sorted(keys)


def test_cap_respected_total_uncapped(monkeypatch):
    before, after = _fixture()
    _, full_total, _ = _diff_cells(before, after)
    monkeypatch.setattr(insp, "_CHANGED_CAP", 10)
    changed, total, _ = _diff_cells(before, after)
    assert len(changed) == 10 and total == full_total > 10
