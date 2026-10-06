import numpy as np
import pandas as pd
import pytest

from dtk_engine.ops.impute_benchmark import benchmark, best_per_column, mask_rows


def _frame():
    # Two entities, linear trend per entity (interpolation is exact), one gap.
    t = np.arange(10.0)
    df = pd.DataFrame(
        {
            "pid": ["a"] * 10 + ["b"] * 10,
            "t": np.tile(t, 2),
            "x": np.concatenate([t * 2, 100 + t * 5]),
        }
    )
    df.loc[3, "x"] = np.nan
    return df


def test_mask_rows_known_only_deterministic_and_never_all():
    df = _frame()
    a = mask_rows(df, "x", 0.3, seed=1)
    assert a.tolist() == mask_rows(df, "x", 0.3, seed=1).tolist()
    assert a.tolist() != mask_rows(df, "x", 0.3, seed=2).tolist()
    assert df["x"].iloc[a].notna().all() and len(a) == round(0.3 * 19)
    assert len(mask_rows(pd.DataFrame({"x": [1.0, 2.0]}), "x", 0.99, 0)) == 1
    with pytest.raises(ValueError, match="fewer than 2"):
        mask_rows(pd.DataFrame({"x": [1.0, np.nan]}), "x", 0.5, 0)


def test_group_interp_beats_global_median():
    df = _frame()
    table = benchmark(
        df, ["x"], ["median", "group_interp"], "pid", "t", 0.2, seed=0
    )
    by = table.set_index("strategy")
    assert by.loc["median", "coverage"] == 1.0
    assert by.loc["group_interp", "rmse"] < by.loc["median", "rmse"]
    assert by.loc["median", "n_masked"] == by.loc["group_interp", "n_masked"]
    assert best_per_column(table)["strategy"].tolist() == ["group_interp"]


def test_coverage_is_separate_from_error():
    # Masking the edge of an entity leaves group_interp unable to fill it.
    df = pd.DataFrame(
        {
            "pid": ["a"] * 4,
            "t": [0.0, 1.0, 2.0, 3.0],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    table = benchmark(df, ["x"], ["group_interp"], "pid", "t", 0.5, seed=0)
    row = table.iloc[0]
    assert row["n_masked"] == 2
    assert row["n_filled"] <= row["n_masked"]
    if row["n_filled"] < row["n_masked"]:
        assert row["coverage"] < 1
    # On what it filled, a linear series is recovered exactly.
    if row["n_filled"]:
        assert row["rmse"] == pytest.approx(0.0)
    else:
        assert row["rmse"] is None


def test_input_untouched_and_non_numeric_rejected():
    df = _frame()
    before = df.copy()
    benchmark(df, ["x"], ["mean"], None, None, 0.2, 0)
    pd.testing.assert_frame_equal(df, before)
    with pytest.raises(ValueError, match="numeric"):
        benchmark(df, ["pid"], ["mean"], None, None, 0.2, 0)
    with pytest.raises(Exception, match="by"):
        benchmark(df, ["x"], ["group_mean"], None, None, 0.2, 0)
