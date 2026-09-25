import numpy as np
import pandas as pd
import pytest

from dtk_engine.ops.selection import (
    collinear_pairs,
    feature_scores,
    infer_task,
    near_constant,
    pca_variance,
)


def planted(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            "a": a,
            "b": b,
            "dup": 2 * a + rng.normal(scale=0.01, size=n),
            "const": np.ones(n),
            "y": (2 * a + b > 0).astype(int),
        }
    )


def test_infer_task():
    assert infer_task(pd.Series([0, 1, 1])) == "classification"
    assert infer_task(pd.Series(["x", "y"])) == "classification"
    assert infer_task(pd.Series([True, False])) == "classification"
    assert infer_task(pd.Series(np.linspace(0, 1, 50))) == "regression"
    assert infer_task(pd.Series(range(50))) == "regression"


@pytest.mark.parametrize("wrapper", [False, True])
def test_feature_scores_rank_informative_first(wrapper):
    df = planted()
    columns = ["noise", "a", "b", "dup", "const"]
    table = feature_scores(df, "y", "classification", columns, wrapper=wrapper)
    assert set(table["column"].head(3)) == {"a", "b", "dup"}
    assert table["column"].iloc[-1] in {"const", "noise"}
    const = table.set_index("column").loc["const"]
    assert const["variance"] == 0 and const["f_score"] == 0 and const["f_pvalue"] == 1
    assert ("rfecv_rank" in table) == wrapper


def test_feature_scores_regression_and_missing():
    df = planted()
    df["y"] = 2 * df["a"] + df["b"]
    df.loc[:9, "noise"] = np.nan
    table = feature_scores(df, "y", "regression", ["noise", "a", "b"])
    assert table["column"].tolist()[:2] == ["a", "b"]
    assert table.set_index("column").loc["noise", "pct_missing"] == 3.33


def test_collinear_and_near_constant():
    df = planted()
    pairs = collinear_pairs(df, ["noise", "a", "b", "dup"])
    assert pairs[["a", "b"]].values.tolist() == [["a", "dup"]]
    df["rare"] = 0
    df.loc[:4, "rare"] = 1
    const = near_constant(df, ["a", "const", "rare"])
    assert const["column"].tolist() == ["const", "rare"]


def test_pca_variance_components():
    df = planted()
    table, needed = pca_variance(df, ["a", "b", "dup", "noise"])
    assert table["cumulative"].iloc[-1] == pytest.approx(1.0)
    # a ~ dup: 3 real dimensions carry (almost) all the variance.
    assert needed["pca_components_99pct"] == 3
    assert needed["pca_components_90pct"] <= needed["pca_components_95pct"] <= 3
