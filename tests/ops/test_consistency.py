import pandas as pd

from dtk_engine.ops.consistency import ambiguous_dates, mixed_types, variants


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
