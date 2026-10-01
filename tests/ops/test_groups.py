"""Within-entity fills (ops/groups.py): the formula functions' building blocks."""

import time

import numpy as np
import pandas as pd

from dtk_engine.ops.groups import group_codes, group_interp, group_mean, group_prev


def _patients():
    """The issue's patient (age -> ledd) plus a second one, rows shuffled."""
    df = pd.DataFrame(
        {
            "pid": ["a"] * 7 + ["b"] * 3,
            "age": [52.1, 53.0, 53.9, 54.8, 56.9, 57.5, 58.9, 30.0, 31.0, 32.0],
            "ledd": [607, 666, 717, 770, 885, np.nan, 835, np.nan, 10, np.nan],
        }
    )
    perm = np.random.default_rng(0).permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


def _by_age(df, values):
    return pd.Series(values, index=df.index)[df.sort_values(["pid", "age"]).index]


def test_group_codes_missing_group_is_nan():
    codes = group_codes(pd.Series(["x", None, "y", "x"]))
    assert codes[0] == codes[3] != codes[2]
    assert np.isnan(codes[1])


def test_group_interp_patient_example():
    df = _patients()
    out = group_interp(df["ledd"].to_numpy(), group_codes(df["pid"]), df["age"])
    expected = [607, 666, 717, 770, 885, 870, 835, np.nan, 10, np.nan]
    # 57.5 between 56.9 -> 885 and 58.9 -> 835; b has no neighbour on both sides.
    np.testing.assert_allclose(_by_age(df, out), expected)


def test_group_prev_never_backward_and_any_dtype():
    df = _patients()
    out = group_prev(df["ledd"], group_codes(df["pid"]), df["age"].to_numpy())
    expected = [607, 666, 717, 770, 885, 885, 835, np.nan, 10, 10]
    np.testing.assert_allclose(_by_age(df, out).astype(float), expected)
    text = pd.Series(["on", None, "off", None], index=[3, 3, 4, 4])
    out = group_prev(text, np.array([0.0, 0.0, 1.0, 1.0]), np.array([1, 2, 2, 1.0]))
    assert out.tolist()[:3] == ["on", "on", "off"]
    assert pd.isna(out.iloc[3])  # before the entity's first value
    assert out.index.tolist() == [3, 3, 4, 4]


def test_group_mean_is_within_entity():
    df = _patients()
    out = group_mean(df["ledd"].to_numpy(), group_codes(df["pid"]))
    a = np.mean([607, 666, 717, 770, 885, 835])
    np.testing.assert_allclose(_by_age(df, out), [a] * 7 + [10.0] * 3)


def test_missing_group_or_order_is_neither_filled_nor_a_source():
    x = np.array([1.0, np.nan, 5.0, np.nan, 100.0])
    by = np.array([0.0, 0.0, 0.0, np.nan, 0.0])
    order = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert np.isnan(group_mean(x, by)[3])
    assert group_mean(x, by)[0] == np.mean([1.0, 5.0, 100.0])
    np.testing.assert_allclose(
        group_interp(x, by, order), [1.0, 3.0, 5.0, np.nan, np.nan]
    )
    prev = group_prev(pd.Series(x), by, order).tolist()
    assert prev[:3] == [1.0, 1.0, 5.0] and np.isnan(prev[3]) and np.isnan(prev[4])


def test_tied_order_interpolates_to_the_mean():
    x = np.array([2.0, np.nan, 4.0])
    out = group_interp(x, np.zeros(3), np.array([1.0, 1.0, 1.0]))
    assert out.tolist() == [2.0, 3.0, 4.0]


def test_vectorised_on_parkinson_scale():
    rng = np.random.default_rng(0)
    n, k = 55_603, 6_971
    by = rng.integers(0, k, n).astype(float)
    order = rng.random(n)
    x = np.where(rng.random(n) < 0.4, np.nan, rng.random(n))
    start = time.perf_counter()
    group_mean(x, by)
    group_interp(x, by, order)
    group_prev(pd.Series(x), by, order)
    assert time.perf_counter() - start < 2.0  # no Python loop over groups
