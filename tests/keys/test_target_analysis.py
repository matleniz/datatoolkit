import json

import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key


def _table(res, title):
    records = next(t["records"] for t in res["tables"] if t["title"] == title)
    return pd.DataFrame.from_records(records)


def test_defaults_on_demo_data():
    res = run_key("target_analysis", {})
    m = res["metrics"]
    assert m["task"] == "classification" and m["n_classes"] == 2
    ranking = _table(res, "ranking")
    assert list(ranking["rank"]) == list(range(1, len(ranking) + 1))
    assert "Survived" not in set(ranking["column"])
    assert ranking["mutual_info"].is_monotonic_decreasing
    balance = _table(res, "class_balance")
    assert balance["count"].sum() == m["n_rows"]
    by_class = _table(res, "numeric_by_class")
    assert set(by_class["class"]) == {"0", "1"}
    rates = _table(res, "class_rate_by_category")
    per_value = rates.groupby(["column", "value"])["rate"].sum()
    assert np.allclose(per_value, 1.0)


def _planted(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "signal": y * 3 + rng.normal(size=n),
            "noise": rng.normal(size=n),
            "cat_signal": np.where(y == 1, "yes", "no"),
            "cat_noise": rng.choice(["a", "b", "c"], size=n),
            "y": y,
        }
    )


def test_classification_ranks_planted_signal_first():
    res = api.target_analysis(_planted(), target="y")
    ranking = pd.DataFrame.from_records(res.tables[0].records)
    assert set(ranking["column"].head(2)) == {"signal", "cat_signal"}
    cat = ranking.set_index("column").loc["cat_signal"]
    assert cat["measure"] == "cramers_v" and cat["association"] == pytest.approx(1.0)
    assert ranking.set_index("column").loc["noise", "association"] < 0.2
    assert json.dumps(res.model_dump(mode="json"))


def test_regression_binned_mean_and_unlabeled_rows():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "noise": rng.normal(size=n),
            "group": rng.choice(["a", "b"], size=n),
            "price": 5 * x + rng.normal(size=n),
        }
    )
    df.loc[:4, "price"] = np.nan
    df.loc[5:9, "x"] = np.nan
    res = api.target_analysis(df, target="price")
    assert res.metrics["task"] == "regression" and res.metrics["n_unlabeled"] == 5
    assert res.metrics["top_feature"] == "x"
    ranking = pd.DataFrame.from_records(res.tables[0].records).set_index("column")
    assert ranking.loc["x", "spearman"] > 0.9
    binned = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "binned_target_mean")
    )
    present = binned[(binned["column"] == "x") & (binned["bin"] != "(missing)")]
    assert present["mean_target"].is_monotonic_increasing
    assert binned.loc[binned["column"] == "x", "count"].sum() == 295
    titles = [t.title for t in res.tables]
    assert "target_mean_by_category" in titles and "class_balance" not in titles
    assert "target_histogram" in titles
    hist = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "target_histogram")
    )
    assert len(hist) == res.metrics["target_bins"] == 10

    coarse = api.target_analysis(df, target="price", target_bins=4)
    coarse_hist = pd.DataFrame.from_records(
        next(t.records for t in coarse.tables if t.title == "target_histogram")
    )
    assert len(coarse_hist) == 4


def test_errors():
    df = _planted()
    with pytest.raises(ValueError, match="not in the frame"):
        api.target_analysis(df, target="nope")
    with pytest.raises(ValueError, match="target"):
        api.target_analysis(df, target="y", columns=["y"])
    with pytest.raises(ValueError, match="regression needs a numeric"):
        api.target_analysis(df, target="cat_signal", task="regression")


def test_bad_params_raise_key_params_error():
    import pytest

    from dtk_engine import run_key
    from dtk_engine.errors import KeyParamsError

    for params in ({"target": "nope"}, {"columns": ["nope"]}):
        with pytest.raises(KeyParamsError):
            run_key("target_analysis", params)


def test_classification_focus_counts_headline_and_figures():
    df = _planted()
    res = api.target_analysis(df, target="y", columns=["cat_signal", "signal"])
    assert res.metrics["focus_feature"] == "cat_signal"
    counts = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "class_counts_by_category")
    )
    sub = counts[counts["column"] == "cat_signal"]
    assert sub["count"].sum() == len(df)
    assert np.allclose(sub.groupby("value")["pct_of_value"].sum(), 100.0)
    by_bin = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "class_counts_by_bin")
    )
    assert by_bin["count"].sum() == len(df)
    assert (
        res.headline == "y rate ranges from 0 % (no) to 100 % (yes) across cat_signal"
    )
    mains = [f for f in res.figures if f.main]
    assert len(mains) == 1 and "cat_signal" in mains[0].title
    assert {t["type"] for t in mains[0].plotly["data"]} == {"bar"}
    assert mains[0].plotly["layout"]["barmode"] == "group"
    assert any("share" in f.title and "cat_signal" in f.title for f in res.figures)
    assert any(f.title == "Association with the target" for f in res.figures)


def test_regression_curve_band_and_headline():
    rng = np.random.default_rng(2)
    n = 400
    x = rng.integers(1, 11, size=n)
    df = pd.DataFrame({"qual": x, "price": 30000 * x + rng.normal(0, 5000, n)})
    res = api.target_analysis(df, target="price", bins=5)
    assert res.headline.startswith("Mean price rises from ")
    assert res.headline.endswith(" across qual bins")
    main = next(f for f in res.figures if f.main)
    assert "qual" in main.title
    traces = main.plotly["data"]
    assert len(traces) == 3 and traces[1]["fill"] == "tonexty"
    assert len(traces[2]["customdata"][0]) == 4  # bin, rows, q1, q3
    binned = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "binned_target_mean")
    )
    assert {"std_target", "q1_target", "q3_target"} <= set(binned.columns)
