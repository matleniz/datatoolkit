import numpy as np
import pandas as pd
import pytest

from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.ops.advisor import MODEL_FAMILIES, advise, as_steps
from dtk_engine.ops.missing import sentinel_counts
from dtk_engine.sources import CsvSource, load
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay_fitted

N = 200


def _frames():
    """Crafted train / test with one planted issue per column."""
    rng = np.random.default_rng(0)
    age = rng.integers(18, 80, N).astype(float)
    age[:5] = -999  # sentinel
    temp = rng.normal(0, 5, N).round(1)
    temp[:3] = -1  # -1 inside the range of the data: not a sentinel
    income = rng.lognormal(10, 1, N)  # right skew
    income[:4] = np.nan
    size = rng.choice(["low", "medium", "high"], N)
    # High cardinality: 5 frequent cities, 55 rare ones.
    city = np.array([f"c{i}" for i in rng.integers(0, 5, N)], dtype=object)
    city[:55] = [f"rare{i}" for i in range(55)]
    sex = rng.choice(["male", "female"], N).astype(object)
    sex[:3] = [" Male", "MALE", "male "]  # variants
    color = rng.choice(["red", "blue"], N).astype(object)
    color[: N // 4] = None  # 25% missing categorical
    y = rng.integers(0, 2, N)
    train = pd.DataFrame(
        {
            "row_id": np.arange(N),
            "age": age,
            "temp": temp,
            "income": income,
            "size": size,
            "city": city,
            "sex": sex,
            "color": color,
            "flag": np.ones(N),
            "sparse": np.where(np.arange(N) < N * 0.3, 1.0, np.nan),
            "y_mean_by_city": rng.random(N),
            "leaky": y * 2.0 + 0.001 * rng.random(N),
            "after_the_fact": rng.random(N),
            "y": y,
        }
    )
    test = train.drop(columns=["y", "after_the_fact"]).head(50).copy()
    test["age"] = test["age"].astype(object)
    test.loc[test.index[:2], "age"] = "unknown"
    test["age"] = test["age"].astype(str)
    test.loc[test.index[0], "color"] = "green"  # unseen category
    test["extra"] = 1
    return train, test


def _by_column(recs: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in recs.itertuples():
        out.setdefault(r.column, []).append(r.op)
    return out


def _rec(recs, column, op):
    rows = recs[(recs["column"] == column) & (recs["op"] == op)]
    assert len(rows) == 1, (column, op, recs[recs["column"] == column])
    return rows.iloc[0]


@pytest.fixture(scope="module")
def frames():
    return _frames()


def test_drops_and_leaks(frames):
    train, test = frames
    recs, columns = advise(train, test, "linear", target="y")
    ops = _by_column(recs)
    for col in ("row_id", "y_mean_by_city", "leaky", "after_the_fact"):
        assert ops[col] == ["drop_columns"], col
        assert _rec(recs, col, "drop_columns")["category"] == "leak"
    assert ops["flag"] == ["drop_columns"]  # constant
    assert ops["sparse"] == ["drop_columns"]  # 70% missing
    assert "y" not in ops  # the target is not a feature
    extra = _rec(recs, "extra", "drop_columns")
    assert extra["target"] == "test"
    assert set(columns.loc[columns["action"] == "drop", "column"]) >= {
        "row_id",
        "flag",
        "sparse",
    }


def test_short_target_name_is_a_token_match(frames):
    train, _ = frames
    train = train.rename(columns={"city": "velocity"})
    recs, _ = advise(train, model_family="tree", target="y")
    assert "drop_columns" not in _by_column(recs).get("velocity", [])


def test_cleaning(frames):
    train, test = frames
    recs, _ = advise(train, test, "linear", target="y")
    sentinels = _rec(recs, "age", "replace_sentinels")["params"]["sentinels"]["age"]
    # -999 in train; "-999.0" and "unknown" in the text column of test.
    assert set(sentinels) == {-999, "-999.0", "unknown"}
    assert _rec(recs, "age", "cast")["params"] == {"dtypes": {"age": "float64"}}
    assert "replace_sentinels" not in _by_column(recs)["temp"]
    mapping = _rec(recs, "sex", "standardize_text")["params"]["mapping"]
    assert {mapping.get(v, v) for v in ("Male", "MALE", "male")} == {"male"}


def test_free_text_variants_standardize_instead_of_drop():
    """Messy survey site column (site-a / site_a / Site A / SITE A among mostly
    unique free text): standardize_text, not drop_columns (MAT-165)."""
    notes = [f"note {i} some unique free text here" for i in range(16)]
    notes += ["site-a", "site_a", "Site A", "SITE A"]
    train = pd.DataFrame({"comment": notes, "y": [0, 1] * 10})
    recs, columns = advise(train, model_family="tree", target="y")
    ops = _by_column(recs)
    assert ops["comment"] == ["standardize_text"]
    rec = _rec(recs, "comment", "standardize_text")
    params = rec["params"]
    assert params["columns"] == ["comment"]
    assert params["strip"] is True
    assert params["lower"] is True
    assert params["unify_separators"] is True  # variants differ by -_. punctuation
    mapping = params["mapping"]
    assert {mapping.get(v, v) for v in ("site-a", "site_a", "Site A", "SITE A")} == {
        mapping.get("site-a", "site-a")
    }
    assert columns.loc[columns["column"] == "comment", "action"].iloc[0] == (
        "standardize_text"
    )


def test_free_text_mixed_dates_parse_instead_of_drop():
    """Free-text column holding dates in >= 2 formats: parse_dates (no single
    unambiguous format to fill in), not drop_columns (MAT-165)."""
    iso_dates = [f"2020-01-{d:02d}" for d in range(10, 17)]  # 7 distinct
    us_dates = [f"{m:02d}/25/2020" for m in range(1, 9)]  # 8 distinct, day=25
    values = iso_dates + us_dates
    values += values[:5]  # pad to 20 rows, keep ratio in (0.5, 0.95): "text"
    train = pd.DataFrame({"signup": values, "y": [0, 1] * 10})
    recs, columns = advise(train, model_family="tree", target="y")
    ops = _by_column(recs)
    assert ops["signup"] == ["parse_dates"]
    rec = _rec(recs, "signup", "parse_dates")
    assert rec["params"] == {"columns": ["signup"]}  # mixed formats: no format guess
    assert columns.loc[columns["column"] == "signup", "action"].iloc[0] == (
        "parse_dates"
    )


def test_free_text_single_date_format_fills_in_format():
    """A free-text date column in one consistent format (plus a couple of non-
    date values, the reason inconsistencies flags it at all) gets `format`
    filled in (MAT-165: 'the detected format when unambiguous')."""
    dates = [f"2020-01-{d:02d}" for d in range(10, 23)]  # 13 distinct, ISO
    values = dates + ["n/a", "n/a"] + dates[:5]  # 20 rows, 14 distinct
    train = pd.DataFrame({"signup": values, "y": [0, 1] * 10})
    recs, _ = advise(train, model_family="tree", target="y")
    rec = _rec(recs, "signup", "parse_dates")
    assert rec["params"] == {"columns": ["signup"], "format": "%Y-%m-%d"}


def test_genuine_free_text_still_drops():
    """No variants, no dates: the free-text drop stays (MAT-165 boundary)."""
    notes = [f"note {i} totally unique free text blah blah" for i in range(20)]
    train = pd.DataFrame({"comment": notes, "y": [0, 1] * 10})
    recs, _ = advise(train, model_family="tree", target="y")
    assert _by_column(recs)["comment"] == ["drop_columns"]
    assert "free text" in _rec(recs, "comment", "drop_columns")["advice"]


def test_currency_column_suggests_to_numeric():
    train = pd.DataFrame(
        {
            "price": ["$2.39 ", "$1,250.00", "$50.00", "$3.10"] * 5,
            "y": [0, 1] * 10,
        }
    )
    recs, _ = advise(train, model_family="tree", target="y")
    rec = _rec(recs, "price", "to_numeric")
    assert rec["params"]["columns"] == ["price"]
    assert rec["params"]["decimal"] == "."
    assert rec["params"]["thousands"] == ","
    get_transform("to_numeric").parse(rec["params"])


@pytest.mark.parametrize(
    ("values", "params"),
    [
        # MAT-168: thousands commas (LaunchCode transaction_total).
        (["$1,029.55", "$34,484.45", "$50.00", "$3.10"] * 5, {"decimal": ".", "thousands": ","}),
        # Percent with no currency symbol (TT Steam avg_peak_perc).
        (["65.9567%", "12.5%", "3.0%", "99.9%"] * 5, {"decimal": ".", "percent": True}),
        # EU space-thousands, comma-decimal amount with no currency code.
        (["1 200,50", "3 400,00", "50,00", "999,99"] * 5, {"decimal": ",", "thousands": " "}),
        # Whole-unit ',-' notation mixed with regular EU amounts.
        (["990,-", "1 200,50", "50,00", "3 400,-"] * 5, {"decimal": ",", "thousands": " "}),
    ],
)
def test_numeric_text_column_suggests_to_numeric(values, params):
    train = pd.DataFrame({"amount": values, "y": [0, 1] * 10})
    recs, _ = advise(train, model_family="tree", target="y")
    rec = _rec(recs, "amount", "to_numeric")
    assert rec["params"]["columns"] == ["amount"]
    for k, v in params.items():
        assert rec["params"][k] == v
    get_transform("to_numeric").parse(rec["params"])


def test_missing_skew_scaling(frames):
    train, test = frames
    recs, columns = advise(train, test, "linear", target="y")
    income = _rec(recs, "income", "impute")["params"]
    assert income["strategy"] == "median" and "add_indicator" not in income
    color = _rec(recs, "color", "impute")["params"]
    assert color["strategy"] == "constant"
    assert _rec(recs, "income", "log1p")["params"] == {"columns": ["income"]}
    assert _rec(recs, "age", "scale")["params"]["method"] == "standard"
    assert "log1p" not in _by_column(recs)["temp"]  # negative values
    # The age sentinels count as missing once replaced.
    assert columns.set_index("column").loc["age", "pct_missing"] == 2.5

    tree, _ = advise(train, test, "tree", target="y")
    assert not {"scale", "log1p"} & set(tree["op"])


def test_robust_scaler_when_outliers_kept():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    x[:15] = 50  # 5% extreme values
    recs, _ = advise(pd.DataFrame({"x": x}), model_family="distance")
    assert _rec(recs, "x", "scale")["params"]["method"] == "robust"


def test_encoding(frames):
    train, test = frames
    recs, columns = advise(train, test, "linear", target="y")
    assert _rec(recs, "size", "ordinal")["params"] == {
        "categories": {"size": ["low", "medium", "high"]}
    }
    city = _rec(recs, "city", "onehot")
    assert city["severity"] == "warning"  # 60 categories
    assert city["params"]["min_frequency"] == 10
    assert _rec(recs, "sex", "onehot")["params"]["drop_first"] is True
    # color: unseen "green" in test -> no drop_first (it would alias to a level)
    assert "drop_first" not in _rec(recs, "color", "onehot")["params"]
    cost = columns.set_index("column")["onehot_columns"]
    assert cost["sex"] == 1 and cost["color"] == 3  # red, blue, MISSING

    tree, _ = advise(train, test, "tree", target="y")
    assert _rec(tree, "city", "ordinal")["params"]["categories"]["city"]


def test_datetime_and_bool():
    df = pd.DataFrame(
        {
            "when": pd.date_range("2024-01-01", periods=30, freq="D"),
            "ok": [True, False] * 15,
        }
    )
    recs, _ = advise(df)
    assert _by_column(recs) == {
        "when": ["datetime_parts", "drop_columns"],
        "ok": ["cast"],
    }


def test_duplicates_and_missing_target():
    df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0], "t": [0, 0, None, 1]})
    recs, _ = advise(df, target="t")
    assert list(recs["op"][:2]) == ["drop_missing_target", "drop_duplicates"]
    assert set(recs["target"][:2]) == {"train"}


@pytest.mark.parametrize("family", [None, *MODEL_FAMILIES])
@pytest.mark.parametrize("data", ["crafted", "demo"])
def test_every_recommendation_is_a_valid_step(frames, family, data):
    if data == "crafted":
        train, test = frames
        target = "y"
    else:
        train, test = load(CsvSource(path=TRAIN_CSV)), load(CsvSource(path=TEST_CSV))
        target = "Survived"
    recs, _ = advise(train, test, family, target)
    assert list(recs["order"]) == list(range(1, len(recs) + 1))
    steps = [Step(**s) for s in as_steps(recs)]
    for step in steps:
        get_transform(step.op).parse(step.params)  # registered op, valid params
    # Applied in order, the plan runs and leaves model-ready frames.
    out_train, out_test, _ = replay_fitted(steps, train, test)
    features = out_train.drop(columns=target)
    assert sorted(features.columns) == sorted(out_test.columns)
    assert not features.isna().any().any()
    assert not out_test.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(features[c]) for c in features)


def test_input_frames_untouched(frames):
    train, test = frames
    before = train.copy(), test.copy()
    advise(train, test, "neural", target="y")
    pd.testing.assert_frame_equal(train, before[0])
    pd.testing.assert_frame_equal(test, before[1])


def test_bad_arguments(frames):
    train, _ = frames
    with pytest.raises(ValueError, match="model_family"):
        advise(train, model_family="svm")
    with pytest.raises(ValueError, match="target"):
        advise(train, target="nope")


def test_imputed_missing_category_is_not_a_sentinel():
    """datatoolkit-issues#78: the engine's own "MISSING" fill must not loop."""
    rng = np.random.default_rng(1)
    gene = rng.choice(["a", "b", "c"], 100).astype(object)
    gene[:32] = None
    train = pd.DataFrame({"gene": gene, "y": rng.integers(0, 2, 100)})
    recs, _ = advise(train, model_family="tree", target="y")
    assert "impute" in _by_column(recs)["gene"]
    rec = _rec(recs, "gene", "impute")
    params = get_transform("impute").parse(rec["params"])
    filled = get_transform("impute").fit_apply(train, params)
    assert (filled["gene"] == "MISSING").sum() == 32
    recs, _ = advise(filled, model_family="tree", target="y")
    assert not {"impute", "replace_sentinels"} & set(_by_column(recs)["gene"])


def test_typed_missing_word_stays_a_sentinel():
    train = pd.DataFrame({"c": ["a", "b", "Missing", "a", "b"] * 4, "y": [0, 1] * 10})
    recs, _ = advise(train, model_family="tree", target="y")
    assert _rec(recs, "c", "replace_sentinels")["params"]["sentinels"] == {
        "c": ["Missing"]
    }


def test_999_inside_a_continuous_range_is_not_a_sentinel():
    """datatoolkit-issues#79: ledd-like column, 999 is a real dose."""
    rng = np.random.default_rng(2)
    ledd = rng.integers(50, 1800, 3000).astype(float)
    ledd[:10] = 999
    train = pd.DataFrame({"ledd": ledd, "y": rng.integers(0, 2, 3000)})
    assert sentinel_counts(train[["ledd"]]).empty
    recs, _ = advise(train, model_family="tree", target="y")
    assert "replace_sentinels" not in _by_column(recs).get("ledd", [])


def test_999_spike_above_the_max_is_still_a_sentinel():
    rng = np.random.default_rng(3)
    ledd = rng.integers(50, 900, 3000).astype(float)
    ledd[:10] = 999
    table = sentinel_counts(pd.DataFrame({"ledd": ledd}))
    assert table[["sentinel", "count"]].values.tolist() == [["999", 10]]


def test_999_spike_inside_the_range_is_still_a_sentinel():
    rng = np.random.default_rng(4)
    ledd = rng.integers(50, 1800, 3000).astype(float)
    ledd[ledd == 999] = 998
    ledd[:60] = 999
    table = sentinel_counts(pd.DataFrame({"ledd": ledd}))
    assert table["sentinel"].tolist() == ["999"]
