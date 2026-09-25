import pandas as pd
import pytest

from dtk_engine.ops.consistency import (
    ambiguous_dates,
    date_format,
    mixed_date_formats,
    mixed_types,
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


def test_variants_none_when_clean():
    summary, mapping = variants(pd.DataFrame({"c": ["a", "b"]}), ["c"])
    assert summary.empty and mapping.empty


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
