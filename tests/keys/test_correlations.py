import numpy as np
import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.transform_registry import get_transform


def _table(res, title):
    records = next(t["records"] for t in res["tables"] if t["title"] == title)
    return pd.DataFrame.from_records(records)


def test_defaults_on_demo_data():
    res = run_key("correlations", {})
    m = res["metrics"]
    assert m["method"] == "pearson" and m["n_columns"] >= 4
    matrix = _table(res, "matrix").set_index("column")
    assert "PassengerId" not in matrix.index  # ids are not auto-picked
    assert np.allclose(np.diag(matrix.to_numpy()), 1.0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T, equal_nan=True)
    assert res["figures"][0]["plotly"]["data"][0]["type"] == "heatmap"


def _planted(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    y = a + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(
        {
            "b": a * 2 + rng.normal(scale=0.01, size=n),
            "a_exp": np.exp(a),
            "a": a,
            "noise": rng.normal(size=n),
            "const": 1.0,
            "y": y,
        }
    )


def test_pairs_keep_and_step_match_drop_correlated():
    df = _planted()
    res = api.correlations(df, threshold=0.95, target="y")
    pairs = pd.DataFrame.from_records(res.tables[0].records)
    assert {tuple(sorted(p)) for p in pairs[["a", "b"]].to_numpy()} == {("a", "b")}
    assert res.metrics["n_undefined"] == 1 and "const" in res.text
    steps = pd.DataFrame.from_records(
        next(t.records for t in res.tables if t.title == "suggested_steps")
    )
    step = steps.iloc[0]
    assert step["op"] == "drop_correlated" and step["target"] == "both"
    out = api.transform(df, "drop_correlated", **step["params"])
    dropped = set(df.columns) - set(out.columns)
    assert len(dropped) == 1
    assert pairs.iloc[0]["keep"] not in dropped  # same choice as the op


def test_spearman_sees_monotonic_and_no_step():
    df = _planted()
    pearson = api.correlations(df, columns=["a", "a_exp"], threshold=0.99)
    spearman = api.correlations(
        df, columns=["a", "a_exp"], method="spearman", threshold=0.99
    )
    assert pearson.metrics["n_pairs"] == 0 and spearman.metrics["n_pairs"] == 1
    assert spearman.tables[-1].records == []  # drop_correlated is pearson only
    assert "method=pearson" in spearman.text


def test_kendall_method():
    df = _planted()
    res = api.correlations(df, columns=["a", "b"], method="kendall", threshold=0.5)
    assert res.metrics["method"] == "kendall"
    assert res.metrics["n_pairs"] >= 1
    assert res.tables[-1].records == []  # suggested step is pearson only


def test_errors():
    df = _planted().assign(text=["x"] * 200)
    with pytest.raises(ValueError, match="not numeric"):
        api.correlations(df, columns=["a", "text"])
    with pytest.raises(ValueError, match="target"):
        api.correlations(df, columns=["a", "y"], target="y")
    with pytest.raises(ValueError, match="not in the frame"):
        api.correlations(df, target="nope")


def test_transform_registry_still_has_drop_correlated():
    assert get_transform("drop_correlated").needs_target


def test_bad_params_raise_key_params_error():
    from dtk_engine.errors import KeyParamsError

    for params in ({"target": "nope"}, {"columns": ["nope"]}):
        with pytest.raises(KeyParamsError):
            run_key("correlations", params)


def _decode(z):
    """A Plotly array field: nested lists, or ``{dtype, bdata, shape}``."""
    if isinstance(z, dict):
        import base64

        arr = np.frombuffer(base64.b64decode(z["bdata"]), dtype=z["dtype"])
        return arr.reshape([int(d) for d in z["shape"].split(",")]).astype(float)
    return np.array(z, dtype=float)


def _wide(n_cols=20, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(100, n_cols)),
        columns=[f"col_{i:02d}_long_name_x" for i in range(n_cols)],
    )
    df["col_19_long_name_x"] = df["col_00_long_name_x"] * 2 + rng.normal(
        scale=0.01, size=100
    )
    return df


def test_heatmap_lower_triangle_and_top_n():
    res = api.correlations(_wide(), top_n=6)
    assert res.metrics["n_columns"] == 20 and res.metrics["n_columns_shown"] == 6
    assert res.headline.startswith("Strongest pair: col_00") or "col_19" in res.headline
    assert "1 pair with |corr| >= 0.9" in res.headline
    main = [f for f in res.figures if f.main]
    assert len(main) == 1
    z = _decode(main[0].plotly["data"][0]["z"])
    assert z.shape == (5, 5)
    assert np.isnan(z[np.triu_indices(5, k=1)]).all()  # upper triangle masked
    assert not np.isnan(z[np.tril_indices(5)]).any()
    shown_x = main[0].plotly["data"][0]["x"]
    assert all(len(x) <= 18 for x in shown_x)
    assert any("col_00" in x for x in shown_x) and any(
        "col_19" in y for y in main[0].plotly["data"][0]["y"]
    )
    assert len(next(t for t in res.tables if t.title == "matrix").records) == 20


def test_strongest_pairs_figure():
    res = api.correlations(_wide())
    fig = next(f for f in res.figures if f.title == "Strongest pairs")
    assert not fig.main
    ys = [y for tr in fig.plotly["data"] for y in tr["y"]]
    assert len(ys) == 10 and all(" × " in y for y in ys)


def test_top_n_bounds():
    with pytest.raises(ValueError):
        api.correlations(_wide(), top_n=1)
