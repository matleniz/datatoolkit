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


def test_errors():
    with pytest.raises(ValueError, match="by_label needs a target"):
        api.distribution(pd.DataFrame({"a": [1, 2]}), by_label=True)
    with pytest.raises(ValueError, match="not in the frame"):
        api.distribution(pd.DataFrame({"a": [1, 2]}), columns=["b"])
    df = pd.DataFrame({"a": [1.5, 2.5, 3.5], "y": [0, 1, 0]})
    with pytest.raises(ValueError, match="target"):
        api.distribution(df, columns=["y"], target="y", by_label=True)
