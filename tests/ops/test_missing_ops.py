import numpy as np
import pandas as pd

from dtk_engine.ops.missing import (
    cooccurrence,
    cooccurrence_pairs,
    missing_per_row,
    missing_rates,
    sentinel_counts,
    value_spikes_vs_train,
)


def test_missing_rates_ranked_with_advice():
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": [None, None, None, 1.0], "c": [None, 1, 2, 3.0]}
    )
    rates = missing_rates(df).set_index("column")
    assert missing_rates(df)["column"].tolist() == ["b", "c", "a"]
    assert rates.loc["b", "pct_missing"] == 75.0
    assert rates.loc["b", "recommendation"].startswith("drop")
    assert rates.loc["a", "recommendation"] == "complete"


def test_per_row_spike():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    df.loc[:59, ["b", "c", "d"]] = np.nan  # a failed batch: 3 fields, 60 rows
    df.loc[100:104, "a"] = np.nan
    hist = missing_per_row(df)
    assert hist.loc[hist["spike"], "n_missing"].tolist() == [3]
    assert hist["n_rows"].sum() == 200


def test_sentinels():
    df = pd.DataFrame(
        {
            "age": [30, 40, -999, -999, 50, 60],
            "city": ["Paris", "N/A", "unknown", "", "Lyon", " - "],
            "when": pd.to_datetime(["1900-01-01", "2020-01-01"] * 3),
        }
    )
    got = {(r.column, r.sentinel): r.count for r in sentinel_counts(df).itertuples()}
    assert got[("age", "-999")] == 2
    assert got[("city", "n/a")] == 1
    assert got[("city", "unknown")] == 1
    assert got[("city", "-")] == 1
    assert got[("city", "''")] == 1
    assert got[("when", "1900-01-01")] == 3


def test_question_mark_sentinel_in_numeric_as_text():
    """UCI-style '?': detected in profile / missing_values / advisor suggestion."""
    from dtk_engine.ops.advisor import advise
    from dtk_engine.ops.advisor.cleaning import sentinel_rec
    from dtk_engine.workspace.inspect import _sentinel_candidates, column_kind

    df = pd.DataFrame(
        {
            "age": ["25", "30", "?", "40", "?", "55", "60", "22", "33", "44"] * 3,
            "hours": ["40", "?", "35", "40", "20"] * 6,
            "label": ["a", "b"] * 15,
        }
    )
    hits = sentinel_counts(df)
    assert set(hits["sentinel"]) >= {"?"}
    assert set(hits["column"]) >= {"age", "hours"}
    assert column_kind(df["age"], "age") == "text"
    assert any(c["value"] == "?" for c in _sentinel_candidates(df["age"]))
    rec = sentinel_rec("age", [df])
    assert rec is not None and rec.op == "replace_sentinels"
    assert "?" in rec.params["sentinels"]["age"]
    recs, _ = advise(df, target="label")
    age_sent = recs[(recs.column == "age") & (recs.op == "replace_sentinels")]
    assert not age_sent.empty


def test_zero_only_when_dominant_in_continuous_column():
    rng = np.random.default_rng(1)
    vals = np.concatenate([np.zeros(50), rng.uniform(1, 100, 50)])
    s = sentinel_counts(pd.DataFrame({"x": vals, "flag": [0, 1] * 50}))
    assert s["column"].tolist() == ["x"] and s["sentinel"].tolist() == ["0"]


def test_cooccurrence():
    df = pd.DataFrame(
        {
            "a": [None, None, 1, 2, 3],
            "b": [None, None, 1, 2, 3],
            "c": [1, 2, 3, None, 5],
            "d": [1, 2, 3, 4, 5],
        }
    )
    m = cooccurrence(df)
    assert sorted(m.index) == ["a", "b", "c"]
    assert m.loc["a", "b"] == 1.0 and m.loc["a", "c"] == 0.0
    pairs = cooccurrence_pairs(m)
    assert pairs[["column_a", "column_b"]].values.tolist() == [["a", "b"]]


def test_test_value_spike():
    rng = np.random.default_rng(2)
    train = pd.DataFrame(
        {"age": rng.uniform(20, 90, 1000).round(1), "sex": ["m", "f"] * 500}
    )
    test = pd.DataFrame(
        {"age": rng.uniform(20, 90, 500).round(1), "sex": ["m", "f"] * 250}
    )
    test.loc[:99, "age"] = 56.3
    spikes = value_spikes_vs_train(train, test)
    assert spikes["column"].tolist() == ["age"]
    assert spikes.loc[0, "value"] == "56.3" and spikes.loc[0, "n_test"] >= 100
    assert value_spikes_vs_train(train, train).empty
