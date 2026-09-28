"""Smart per-column defaults: bins, log scale, top-k (MAT-174)."""

import numpy as np
import pandas as pd

from dtk_engine.ops.suggested import (
    suggest_histogram_bins,
    suggest_log_scale,
    suggest_top_k,
    suggested_params,
)


def test_discrete_int_bins_differ_from_continuous():
    """Small-int column -> one bin per value; continuous -> Freedman/Sturges count."""
    rng = np.random.default_rng(0)
    small_int = pd.Series(rng.integers(1, 6, size=500))  # 5 distinct ints
    continuous = pd.Series(rng.normal(size=500))
    int_bins = suggest_histogram_bins(small_int)
    cont_bins = suggest_histogram_bins(continuous)
    assert int_bins == 5
    assert cont_bins != int_bins
    assert cont_bins >= 2


def test_log_scale_on_heavy_right_skew():
    rng = np.random.default_rng(1)
    skewed = pd.Series(rng.lognormal(mean=2, sigma=1.5, size=800))
    normal = pd.Series(rng.normal(size=800))
    assert suggest_log_scale(skewed) is True
    assert suggest_log_scale(normal) is False
    assert suggest_log_scale(pd.Series([-1.0, 0.0, 1.0] * 10)) is False


def test_top_k_adapts_to_cardinality():
    low = pd.Series(["a", "b", "c"] * 20)
    high = pd.Series([f"v{i}" for i in range(80)] * 3)
    assert suggest_top_k(low) == 3
    assert suggest_top_k(high) == 10


def test_suggested_params_shape():
    num = suggested_params(pd.Series([1.0, 2.0, 3.0, 100.0] * 50), kind="number")
    assert set(num) >= {"bins", "log_scale", "top_k"}
    cat = suggested_params(pd.Series(["x", "y"] * 10), kind="text")
    assert set(cat) == {"top_k"}
    assert cat["top_k"] == 2
