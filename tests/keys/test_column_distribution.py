import json

import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.ops.distribution import OTHER_LABEL
from dtk_engine.ops.profile import MISSING_LABEL


def _table(res, title):
    records = next(t["records"] for t in res["tables"] if t["title"] == title)
    return pd.DataFrame.from_records(records)


def test_defaults_on_demo_data():
    res = run_key("column_distribution", {})
    m = res["metrics"]
    assert m["n_groups"] == 1 and m["compare"] == "none"
    assert m["by"] == "none"
    assert m["n_numeric"] >= 2 and m["n_categorical"] >= 1
    columns = _table(res, "columns")
    assert "Name" not in set(columns["column"])  # free text: not auto-picked
    assert "PassengerId" not in set(columns["column"])  # id: not auto-picked
    assert len(res["figures"]) == m["n_columns"]


def test_train_vs_test_by_label_shared_bins():
    res = run_key(
        "column_distribution",
        {
            "columns": ["Age", "Sex"],
            "compare": "train_vs_test",
            "by_label": True,
            "target": "Survived",
        },
    )
    # The demo test has no Survived: it stays one group.
    groups = _table(res, "groups")
    assert list(groups["group"]) == ["train / 0", "train / 1", "test"]
    hist = _table(res, "histograms")
    edges = hist.groupby("group")["bin_left"].apply(tuple)
    assert edges.nunique() == 1  # same bins for every group
    assert np.allclose(hist.groupby("group")["share"].sum(), 1.0)
    counts = _table(res, "value_counts")
    assert set(counts["value"]) == {"male", "female"}
    for _, part in counts.groupby("group"):
        assert part["pct"].sum() == pytest.approx(100, abs=0.1)


def test_by_age_survived_titanic():
    """Titanic: Age by Survived — two class groups, shared histogram bins."""
    res = run_key(
        "column_distribution",
        {"columns": ["Age"], "by": "Survived"},
    )
    assert res["metrics"]["by"] == "Survived"
    assert res["metrics"]["by_label"] == "none"
    groups = list(_table(res, "groups")["group"])
    assert groups == ["0", "1"]
    hist = _table(res, "histograms")
    assert set(hist["group"]) == {"0", "1"}
    assert hist.groupby("group")["bin_left"].apply(tuple).nunique() == 1


def test_by_sex_survived_titanic():
    """Titanic: Sex by Survived — value counts per class."""
    res = run_key(
        "column_distribution",
        {"columns": ["Sex"], "by": "Survived"},
    )
    counts = _table(res, "value_counts")
    assert set(counts["group"]) == {"0", "1"}
    assert set(counts["value"]) == {"male", "female"}
    for _, part in counts.groupby("group"):
        assert part["pct"].sum() == pytest.approx(100, abs=0.1)


def test_by_fare_pclass_titanic():
    """Titanic: Fare by Pclass — one group per class (1/2/3)."""
    res = run_key(
        "column_distribution",
        {"columns": ["Fare"], "by": "Pclass"},
    )
    groups = list(_table(res, "groups")["group"])
    assert groups == ["1", "2", "3"]
    assert res["metrics"]["n_groups"] == 3


def test_by_age_vs_fare_scatter_and_corr():
    """Titanic: Age vs Fare — sampled scatter + pearson / spearman (MAT-159)."""
    res = run_key(
        "column_distribution",
        {"columns": ["Age"], "by": "Fare"},
    )
    assert "pearson" in res["metrics"] and "spearman" in res["metrics"]
    vs = _table(res, "vs_by")
    assert list(vs["column"]) == ["Age"] and list(vs["by"]) == ["Fare"]
    assert vs["n_rows"].iloc[0] > 0
    titles = [f["title"] for f in res["figures"]]
    assert "Age vs Fare" in titles
    scatter = next(f for f in res["figures"] if f["title"] == "Age vs Fare")
    # Plotly scatter data length capped at 2000.
    n_points = len(scatter["plotly"]["data"][0]["x"])
    assert 0 < n_points <= 2000


def test_top_k_other_missing_and_numeric_target_bins():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "city": rng.choice(list("abcdefgh"), size=n).astype(object),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n) * 10,
        }
    )
    df.loc[:9, "city"] = None
    res = api.distribution(
        df, columns=["city", "x"], target="y", by_label=True, top_k=3
    )
    assert json.dumps(res.model_dump(mode="json"))
    counts = pd.DataFrame.from_records(res.tables[-1].records)
    values = list(dict.fromkeys(counts["value"]))
    assert len(values) == 5 and values[-2:] == [OTHER_LABEL, MISSING_LABEL]
    groups = pd.DataFrame.from_records(res.tables[1].records)["group"].tolist()
    assert len(groups) == 4 and groups[0].startswith("(-inf") and "inf]" in groups[-1]


def test_by_categorical_topk_other():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "city": rng.choice(list("abcdefghij"), size=n).astype(object),
            "x": rng.normal(size=n),
        }
    )
    res = api.distribution(df, columns=["x"], by="city", top_k=3)
    groups = list(_table(res.model_dump(mode="json"), "groups")["group"])
    assert OTHER_LABEL in groups
    assert len(groups) == 4  # top-3 + (other)


def test_errors():
    with pytest.raises(ValueError, match="by_label needs a target"):
        api.distribution(pd.DataFrame({"a": [1, 2]}), by_label=True)
    with pytest.raises(ValueError, match="not in the frame"):
        api.distribution(pd.DataFrame({"a": [1, 2]}), columns=["b"])
    df = pd.DataFrame({"a": [1.5, 2.5, 3.5], "y": [0, 1, 0]})
    with pytest.raises(ValueError, match="target"):
        api.distribution(df, columns=["y"], target="y", by_label=True)
    with pytest.raises(ValueError, match="by or by_label"):
        api.distribution(df, columns=["a"], by="y", by_label=True, target="y")
    with pytest.raises(ValueError, match="by"):
        api.distribution(df, columns=["a"], by="nope")


def test_bad_params_raise_key_params_error():
    from dtk_engine import run_key
    from dtk_engine.errors import KeyParamsError

    for params in (
        {"by_label": True},
        {"by_label": True, "target": "nope"},
        {"columns": ["nope"]},
        {"by": "nope"},
        {"by": "Survived", "by_label": True, "target": "Survived"},
        {"bins": 1},
        {"range_min_pct": 80, "range_max_pct": 20},
        {"bin_edges": [1.0]},
    ):
        with pytest.raises(KeyParamsError):
            run_key("column_distribution", params)


def test_auto_bins_differ_small_int_vs_continuous():
    """Acceptance: default bins differ between a small-int and a continuous column."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "grade": rng.integers(1, 6, size=400),
            "score": rng.normal(size=400),
        }
    )
    int_hist = _table(
        api.distribution(df, columns=["grade"]).model_dump(mode="json"), "histograms"
    )
    cont_hist = _table(
        api.distribution(df, columns=["score"]).model_dump(mode="json"), "histograms"
    )
    n_int = int_hist["bin_left"].nunique()
    n_cont = cont_hist["bin_left"].nunique()
    assert n_int == 5
    assert n_cont != n_int


def test_explicit_edges_norm_cumulative_clip():
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 100.0] * 20})
    res = api.distribution(
        df,
        columns=["x"],
        bin_edges=[0, 2, 4, 6],
        range_min_pct=0,
        range_max_pct=95,
        norm="count",
        cumulative=True,
    )
    hist = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "histograms")
    )
    assert list(hist["bin_left"]) == [0.0, 2.0, 4.0]
    assert hist["cumulative_count"].is_monotonic_increasing
    assert res.metrics["norm"] == "count"
    assert res.metrics["cumulative"] == 1


def test_bins_number_and_density():
    df = pd.DataFrame({"x": np.linspace(0, 10, 200)})
    res = api.distribution(df, columns=["x"], bins=8, norm="density")
    hist = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "histograms")
    )
    assert hist["bin_left"].nunique() == 8
    assert (hist["density"] >= 0).all()
    assert res.metrics["bins"] == 8

