import time

import pandas as pd
import pytest

from dtk_engine.ops.consistency import (
    FUZZY_MAX_DISTINCT,
    ambiguous_dates,
    date_format,
    mixed_date_formats,
    mixed_types,
    normalize,
    variants,
)


def test_variants_merge_and_canonical():
    df = pd.DataFrame(
        {"city": ["Paris", "Paris", "Paris", " paris", "PARIS ", "Lyon", None]}
    )
    summary, mapping = variants(df, ["city"])
    row = summary.iloc[0]
    assert (row["distinct_before"], row["distinct_after"]) == (4, 2)
    assert set(mapping["canonical"]) == {"Paris"}
    assert set(mapping["variant"]) == {"Paris", " paris", "PARIS "}
    assert set(mapping["method"]) == {"exact"}
    assert set(mapping["similarity"]) == {100.0}


def test_variants_none_when_clean():
    summary, mapping = variants(pd.DataFrame({"c": ["a", "b"]}), ["c"])
    assert summary.empty and mapping.empty


def test_normalize_unifies_separators():
    assert normalize("site-a") == normalize("site_a") == normalize("Site  A")
    assert normalize("site.a") == normalize("site-a")
    assert normalize("site-a") == "site a"


def test_variants_merges_separator_punctuation():
    df = pd.DataFrame({"site": ["site-a", "site_a", "Site A", "site-b"]})
    summary, mapping = variants(df, ["site"])
    row = summary.iloc[0]
    assert (row["distinct_before"], row["distinct_after"]) == (4, 2)
    site_a_forms = set(mapping.loc[mapping["variant"] == "site-a", "canonical"])
    assert site_a_forms  # site-a was merged with something
    canonical = site_a_forms.pop()
    merged = set(mapping.loc[mapping["canonical"] == canonical, "variant"])
    assert merged == {"site-a", "site_a", "Site A"}


def test_mixed_types():
    df = pd.DataFrame({"a": [1, "x", 2.5], "b": [1, 2, 3], "c": ["u", "v", "w"]})
    assert mixed_types(df)["column"].tolist() == ["a"]


def test_ambiguous_dates():
    df = pd.DataFrame(
        {
            "d": ["01/02/2023", "13/02/2023", "05/05/2023"],
            "e": ["25/12/2023", "13/01/2023", "30/06/2023"],
            "t": ["foo", "bar", "baz"],
        }
    )
    out = ambiguous_dates(df, ["d", "e", "t"])
    assert out["column"].tolist() == ["d"]
    assert out["n_ambiguous"].tolist() == [1]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2020-01-15", "yyyy-mm-dd"),
        ("2020-01-15T10:30:00", "yyyy-mm-dd"),
        ("2020/01/15", "yyyy/mm/dd"),
        ("01/15/2020", "mm/dd/yyyy"),
        ("15-01-2020", "dd-mm-yyyy"),
        ("01/02/20", "nn/nn/yy"),
        ("15 Jan 2020", "dd mon yyyy"),
        ("Jan 15, 2020", "mon dd yyyy"),
        ("garbage", None),
        ("2024", None),
        ("15 Foo 2020", None),
        ("40/40/2020", None),
    ],
)
def test_date_format(value, expected):
    assert date_format(value) == expected


def test_mixed_date_formats():
    df = pd.DataFrame(
        {
            "joined": ["2020-01-01", "01/15/2020", "15-01-2020", "not-a-date", None],
            # Day-first only (15/01 settles 02/03): one format, not reported.
            "eu": ["15/01/2020", "02/03/2020", "20/05/2020", "2/2/2020", None],
            "iso": ["2020-01-01", "2021-02-03", "2022-03-04", "2023-04-05", None],
            # Mostly not dates: not a date column.
            "txt": ["a", "b", "c", "2020-01-01", None],
            # Day-first and month-first both seen: two formats.
            "flip": ["15/01/2020", "01/15/2020", "20/02/2020", "02/20/2020", None],
        }
    )
    out = mixed_date_formats(df, list(df.columns)).set_index("column")
    assert list(out.index) == ["joined", "flip"]
    joined = out.loc["joined"]
    assert (joined["n_formats"], joined["n_not_date"]) == (3, 1)
    assert "not-a-date" in joined["examples"]
    assert out.loc["flip", "formats"] == "dd/mm/yyyy: 2, mm/dd/yyyy: 2"


# --- fuzzy (rapidfuzz) matching, MAT-170 -----------------------------------


def test_variants_fuzzy_typos_and_punctuation():
    """Approximate spelling variants close enough to merge: a transposition
    typo, dotted vs. plain acronym, a doubled letter, and (via the existing
    exact pass) a hyphen vs. space separator."""
    country = (
        ["United States"] * 5
        + ["Untied States"]
        + ["USA"] * 4
        + ["U.S.A"] * 2
        + ["France"] * 6
        + ["Paris"] * 3
        + ["Pariss"]
        + ["new york"] * 3
        + ["new-york"] * 2
    )
    df = pd.DataFrame({"country": country})
    summary, mapping = variants(df, ["country"])
    row = summary.loc[summary["column"] == "country"].iloc[0]
    assert (row["distinct_before"], row["distinct_after"]) == (9, 5)

    by_variant = mapping.set_index("variant")
    assert by_variant.loc["Untied States", "canonical"] == "United States"
    assert by_variant.loc["Untied States", "method"] == "fuzzy"
    assert 90 <= by_variant.loc["Untied States", "similarity"] < 100

    assert by_variant.loc["U.S.A", "canonical"] == "USA"
    assert by_variant.loc["U.S.A", "method"] == "fuzzy"

    assert by_variant.loc["Pariss", "canonical"] == "Paris"
    assert by_variant.loc["Pariss", "method"] == "fuzzy"

    # "new-york" already merges with "new york" via the exact pass (unify
    # separators), no fuzzy matching needed.
    assert by_variant.loc["new-york", "canonical"] == "new york"
    assert by_variant.loc["new-york", "method"] == "exact"


def test_variants_acronyms_vs_full_names_not_fuzzy_matched():
    """Keep it honest (MAT-170): 'usa' vs 'United States' score far below the
    similarity threshold and are left apart for a manual map."""
    df = pd.DataFrame({"country": ["usa"] * 5 + ["United States"] * 5})
    summary, mapping = variants(df, ["country"])
    assert summary.empty
    assert mapping.empty


def test_variants_fuzzy_pass_ignores_numeric_and_id_like_values():
    """Values that are mostly digits (dates) or carry digits (id-like codes)
    are not fuzzy-clustered: a 1-digit difference is not a spelling variant."""
    df = pd.DataFrame(
        {
            "signup": ["2020-01-10"] * 3 + ["2020-01-11"] * 3 + ["2020-01-12"] * 3,
            "code": ["rare0"] * 3 + ["rare1"] * 3 + ["rare10"] * 3,
        }
    )
    summary, mapping = variants(df, ["signup", "code"])
    assert summary.empty
    assert mapping.empty


def test_variants_high_cardinality_column_skips_fuzzy_pass():
    """A column above fuzzy_max_distinct skips the O(n^2) fuzzy comparison;
    only exact merges still apply."""
    df = pd.DataFrame({"city": ["Paris"] * 3 + ["Pariss"] + ["Lyon"] * 2})
    summary, mapping = variants(df, ["city"], fuzzy_max_distinct=1)
    assert summary.empty
    assert mapping.empty


def test_variants_high_cardinality_column_is_fast():
    """Real-scale guard for the FUZZY_MAX_DISTINCT bound: many distinct,
    mostly-unique values must not trigger the O(n^2) fuzzy pass."""
    n = FUZZY_MAX_DISTINCT + 200
    values = [f"value{i}" for i in range(n)]
    df = pd.DataFrame({"c": values})
    start = time.perf_counter()
    summary, mapping = variants(df, ["c"])
    elapsed = time.perf_counter() - start
    assert summary.empty
    assert mapping.empty
    assert elapsed < 5
