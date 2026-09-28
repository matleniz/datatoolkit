import numpy as np
import pandas as pd
import pytest

from dtk_engine.ops.profile import (
    ID_MIN_NON_NULL,
    NUMERIC_STATS_COLUMNS,
    SEMANTIC_TYPES,
    category_summary,
    category_values,
    column_profile,
    columns_of_type,
    datetime_stats,
    id_stats,
    numeric_histograms,
    numeric_stats,
    numeric_text_format,
    pct_numeric_parsable,
    semantic_type,
    semantic_types,
    text_stats,
)

N = ID_MIN_NON_NULL


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.5, 2.5, 2.5, 3.0], "numeric"),
        ([True, False, True, True], "boolean"),
        ([0, 1, 1, 0], "boolean"),
        (pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]), "datetime"),
        (["2024-01-01", "2024-02-01", "2024-03-01"], "datetime"),
        (["a", "b", "a", "b", "a", "c"], "categorical"),
        (["the quick fox", "a lazy dog", "hello world", "foo bar"], "text"),
        ([f"id{i}" for i in range(N)], "id_like"),
        (list(range(10, 10 + N)), "id_like"),
        # Index 0..n-1 with one duplicated row is still an identifier.
        ([*range(N), 0], "id_like"),
        # Too few non-null values to call it an identifier (demo `Cabin`).
        (["C85", "E46", "B28", "C123", "D33", *[None] * 36], "text"),
        ([10, 11, 12, 13], "numeric"),
        # All-distinct integers spread over a wide range: a feature, not an id.
        ([i * 997 + i % 7 for i in range(N)], "numeric"),
        # Entity key repeated over rows (e.g. patient_id).
        ([f"P{i % 150:04d}" for i in range(1500)], "group_id"),
        ([i % 150 for i in range(1500)], "group_id"),
        # Measured ints with holes in their range stay numeric (Pima Glucose).
        (
            [0, 199, *list(range(1, 135))]
            + [v % 136 for v in range(136, 768)],
            "numeric",
        ),
        # Few categories stay categorical; sparse spread integers stay numeric.
        ([f"c{i % 20}" for i in range(1500)], "categorical"),
        ([(i % 150) * 1000 for i in range(1500)], "numeric"),
        # Year strings are not dates.
        (["2024", "2023", "2024", "2022", "2023", "2024"], "categorical"),
        # Numbers polluted by a token stay text-like (not an identifier).
        (["12", "unknown", "31", "45", "7"], "text"),
        # Mostly-numeric text with sentinel tokens is not a group_id.
        (
            [str(i % 200) if i % 17 else "?" for i in range(1500)],
            "categorical",
        ),
        (["x", "x", None], "constant"),
        ([None, None], "constant"),
    ],
)
def test_semantic_type(values, expected):
    assert semantic_type(pd.Series(values)) == expected
    assert expected in SEMANTIC_TYPES


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # MAT-168: high-cardinality currency / percent / EU-amount text must not
        # be sniffed as an identifier (LaunchCode transaction_total, TT Steam
        # avg_peak_perc, EU montant columns).
        ([f"${1000 + i:,}.{i % 100:02d}" for i in range(N)], "text"),
        ([f"{i % 100}.{i:04d}%" for i in range(N)], "text"),
        ([f"1 {200 + i},50" for i in range(N)], "text"),
        # Plain leading-zero ids / zip codes: no comma/percent/currency/
        # thousands marker, so numbers-as-text detection stays out of the way
        # (already parsed by plain pd.to_numeric -> not id_like, unaffected).
        ([f"{i:05d}" for i in range(N)], "text"),
    ],
)
def test_semantic_type_numeric_text_not_identifier(values, expected):
    assert semantic_type(pd.Series(values)) == expected


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["$1,029.55", "$34,484.45", "$50.00"], {"decimal": ".", "thousands": ",", "percent": False}),
        (["65.9567%", "12.5%", "3.0%"], {"decimal": ".", "thousands": None, "percent": True}),
        (["1 200,50", "3 400,00"], {"decimal": ",", "thousands": " ", "percent": False}),
        (["1 200,50", "3 400,00"], {"decimal": ",", "thousands": " ", "percent": False}),
        (["990,-", "1 200,50"], {"decimal": ",", "thousands": " ", "percent": False}),
        (["$2.39 ", "$1,250.00", "$50.00", "$3.10"], {"decimal": ".", "thousands": ",", "percent": False}),
        (["12,5 %", "3,0 %"], {"decimal": ",", "thousands": None, "percent": True}),
        (["1 250,00 EUR", "50,00 EUR"], {"decimal": ",", "thousands": " ", "percent": False}),
        # Plain numbers-as-text with only a '.' decimal: not a "marker" case,
        # leave it to plain pd.to_numeric / the cast advisor path.
        (["35.0", "-999.0", "12.0"], None),
        # Plain digit ids: no separators/symbols at all.
        (["00123", "00456"], None),
    ],
)
def test_numeric_text_format(values, expected):
    assert numeric_text_format(pd.Series(values)) == expected


def test_pima_glucose_is_numeric_not_identifier():
    """Pima Glucose: 136 distinct ints over 768 rows spanning 0..199 → numeric."""
    unique = [int(i * 199 / 135) for i in range(136)]
    values = (unique * (768 // len(unique) + 1))[:768]
    s = pd.Series(values)
    assert semantic_type(s) == "numeric"


def test_column_profile():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 3.0], "b": ["x", "y", "x", "x"]})
    prof = column_profile(df).set_index("column")
    assert list(prof.index) == ["a", "b"]
    assert prof.loc["a", "n_missing"] == 1
    assert prof.loc["a", "pct_missing"] == 25.0
    assert prof.loc["a", "n_unique"] == 2
    assert prof.loc["a", "semantic_type"] == "numeric"
    assert prof.loc["b", "semantic_type"] == "categorical"
    assert prof.loc["b", "sample_values"] == "x, y"


def test_column_profile_empty_frame():
    prof = column_profile(pd.DataFrame({"a": []}))
    assert prof.loc[0, "pct_missing"] == 0.0


def test_column_profile_pct_numeric_parsable():
    df = pd.DataFrame(
        {
            "age": ["12", "unknown", "31", " 45", None],
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": ["a", "b", "a", "b", "a"],
            "flag": [True, False, True, True, False],
        }
    )
    prof = column_profile(df).set_index("column")
    assert prof.loc["age", "pct_numeric_parsable"] == 75.0
    assert prof.loc["age", "semantic_type"] == "text"
    assert prof.loc["num", "pct_numeric_parsable"] == 100.0
    assert prof.loc["cat", "pct_numeric_parsable"] == 0.0
    assert prof.loc["flag", "pct_numeric_parsable"] == 0.0


def test_pct_numeric_parsable_empty():
    assert pct_numeric_parsable(pd.Series([None, None], dtype=object)) == 0.0


# --- per-semantic-type statistics --------------------------------------------


def _mixed_frame() -> pd.DataFrame:
    n = 200
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    x[0] = 50.0  # outlier
    return pd.DataFrame(
        {
            "x": x,
            "cat": ["a"] * 150 + ["b"] * 49 + ["c"],
            "flag": [0, 1] * 100,
            "when": pd.date_range("2024-01-01", periods=n, freq="D"),
            "txt": [f"free text number {i}" for i in range(n)],
            "idx": range(n),
            "const": 1,
        }
    )


def test_numeric_stats_values():
    df = pd.DataFrame({"v": [-2.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0] * 3})
    out = numeric_stats(df)
    assert list(out.columns) == NUMERIC_STATS_COLUMNS
    row = out.iloc[0]
    assert row["column"] == "v" and row["count"] == 30
    assert row["min"] == -2 and row["max"] == 100
    assert row["mean"] == pytest.approx(df["v"].mean())
    assert row["median"] == pytest.approx(df["v"].median())
    assert row["n_zeros"] == 6 and row["n_negative"] == 3
    assert row["n_outliers_iqr"] == 3 and row["pct_outliers_iqr"] == 10.0
    assert row["n_outliers_z"] == 0  # 100 is < 3 sd here (heavy inflated sd)
    assert row["skew"] > 0


def test_numeric_stats_z_outliers_and_missing():
    x = np.zeros(100)
    x[:50] = np.linspace(-1, 1, 50)
    x[0] = 1000.0
    out = numeric_stats(pd.DataFrame({"v": [*x, np.nan]}))
    row = out.iloc[0]
    assert row["count"] == 100
    assert row["n_outliers_z"] == 1 and row["pct_outliers_z"] == 1.0


def test_numeric_stats_only_numeric_columns():
    out = numeric_stats(_mixed_frame())
    assert list(out["column"]) == ["x"]
    assert numeric_stats(pd.DataFrame({"s": list("abab")})).empty


def test_numeric_histograms_counts_sum_to_non_null():
    hist = numeric_histograms(_mixed_frame(), bins=10)
    assert len(hist) == 10
    assert hist["count"].sum() == 200


def test_category_values_missing_sorted_and_pct():
    df = pd.DataFrame(
        {"b": ["x", "y", "y", None], "a": ["u", "u", "v", "v"]},
    )
    out = category_values(df, ["b", "a"])
    assert list(out.columns) == ["column", "value", "count", "pct"]
    assert list(out["column"]) == ["a", "a", "b", "b", "b"]
    b = out[out["column"] == "b"]
    assert list(b["value"]) == ["y", "(missing)", "x"]  # tie: by value
    assert list(b["count"]) == [2, 1, 1]
    assert list(b["pct"]) == [50.0, 25.0, 25.0]
    assert category_values(df, []).empty


def test_category_summary_top_and_rare():
    out = category_summary(_mixed_frame())
    assert set(out["column"]) == {"cat", "flag"}
    cat = out[out["column"] == "cat"].iloc[0]
    assert cat["n_unique"] == 3 and cat["top_value"] == "a"
    assert cat["top_pct"] == 75.0
    assert cat["pct_rare"] == 0.5  # "c": 1 row of 200 = 0.5 % < 1 %


def test_datetime_stats_native_and_string():
    df = pd.DataFrame(
        {
            "native": pd.date_range("2024-01-01", periods=3, freq="D"),
            "text": ["2024-01-01", "2024-01-11", None],
        }
    )
    out = datetime_stats(df).set_index("column")
    assert out.loc["native", "span_days"] == 2.0
    assert out.loc["native", "min"].startswith("2024-01-01")
    assert out.loc["text", "count"] == 2 and out.loc["text", "span_days"] == 10.0


def test_text_stats_lengths():
    df = pd.DataFrame({"t": ["the quick fox", "a lazy dog", "hello world", None]})
    row = text_stats(df).iloc[0]
    assert row["count"] == 3 and row["n_unique"] == 3
    assert (row["min_len"], row["max_len"]) == (10, 13)
    assert row["mean_len"] == pytest.approx(11.33, abs=0.01)


def test_id_stats_duplicates():
    df = pd.DataFrame(
        {"i": [*range(N), 0], "g": [f"P{i % 150:04d}" for i in range(N + 1)]}
    )
    out = id_stats(df).set_index("column")
    assert out.loc["i", "n_duplicates"] == 1 and out.loc["i", "n_unique"] == N
    assert out.loc["i", "semantic_type"] == "id_like"


def test_type_dispatch_uses_semantic_type():
    df = _mixed_frame()
    sem = semantic_types(df)
    assert columns_of_type(df, "constant", semantic=sem) == ["const"]
    assert list(datetime_stats(df)["column"]) == ["when"]
    assert list(text_stats(df)["column"]) == ["txt"]
    assert list(id_stats(df)["column"]) == ["idx"]
