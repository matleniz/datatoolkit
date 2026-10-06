import json

import numpy as np
import pandas as pd
import pytest

from dtk_engine import run_key
from dtk_engine.errors import KeyParamsError


def _table(res):
    return next(t["records"] for t in res["tables"] if t["title"] == "benchmark")


def test_defaults_on_demo_data():
    res = run_key("impute_benchmark", {})
    rows = _table(res)
    assert {r["strategy"] for r in rows} == {"median", "mean"}
    assert {r["column"] for r in rows} == {"Age"}
    for r in rows:
        assert r["coverage"] == 1.0 and r["rmse"] >= r["mae"] > 0
    assert res["metrics"]["best_Age"] in {"median", "mean"}
    assert res["headline"].startswith("Best by RMSE: Age:")
    assert run_key("impute_benchmark", {}) == res  # deterministic
    assert json.dumps(res)


@pytest.fixture
def csv(tmp_path):
    t = np.arange(12.0)
    df = pd.DataFrame(
        {
            "pid": ["a"] * 12 + ["b"] * 12,
            "t": np.tile(t, 2),
            "x": np.concatenate([t * 2, 100 + t * 5]),
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return {"kind": "csv", "path": str(path)}


def test_group_strategies_win_on_trending_entities(csv):
    res = run_key(
        "impute_benchmark",
        {
            "source": csv,
            "columns": ["x"],
            "strategies": ["median", "group_mean", "group_interp"],
            "by": "pid",
            "order": "t",
            "seed": 3,
        },
    )
    assert res["metrics"]["best_x"] == "group_interp"
    rows = {r["strategy"]: r for r in _table(res)}
    assert rows["group_interp"]["rmse"] < rows["group_mean"]["rmse"] < rows["median"]["rmse"]
    assert len({r["n_masked"] for r in rows.values()}) == 1


def test_bad_params_raise(csv):
    with pytest.raises(KeyParamsError):
        run_key("impute_benchmark", {"source": csv, "columns": ["nope"]})
    with pytest.raises(KeyParamsError):
        run_key("impute_benchmark", {"source": csv, "columns": ["pid"]})
    with pytest.raises(KeyParamsError):
        run_key(
            "impute_benchmark",
            {"source": csv, "columns": ["x"], "strategies": ["group_mean"]},
        )
    with pytest.raises(KeyParamsError):
        run_key("impute_benchmark", {"source": csv, "columns": ["x"], "strategies": ["knn"]})
    with pytest.raises(KeyParamsError):
        run_key("impute_benchmark", {"source": csv, "columns": ["x"], "mask_fraction": 1})
