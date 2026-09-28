import pandas as pd
import pytest

from dtk_engine.errors import SourceError
from dtk_engine.sources import CsvSource, load
from dtk_engine.sources.csv_pandas import SNIFF_CHARS, resolve_path, sniff_sep


@pytest.mark.parametrize("sep", [",", ";"])
def test_auto_sep_sniffing(tmp_path, sep):
    path = tmp_path / "data.csv"
    path.write_text(f"a{sep}b{sep}c\n1{sep}x{sep}2.5\n3{sep}y{sep}4.5\n")
    df = load(CsvSource(path=str(path)))
    assert list(df.columns) == ["a", "b", "c"]
    assert df["c"].tolist() == [2.5, 4.5]


def test_explicit_sep_and_decimal(tmp_path):
    path = tmp_path / "fr.csv"
    path.write_text("a;b\n1,5;x\n2,5;y\n")
    df = load(CsvSource(path=str(path), sep=";", decimal=","))
    assert df["a"].tolist() == [1.5, 2.5]


def test_no_header(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("1,2\n3,4\n")
    df = load(CsvSource(path=str(path), header=None))
    assert df.shape == (2, 2)


def test_missing_file_raises_source_error(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        load(CsvSource(path=str(tmp_path / "nope.csv")))


def test_unreadable_file_raises_source_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(SourceError):
        load(CsvSource(path=str(path)))


def test_spec_is_strict():
    with pytest.raises(ValueError):
        CsvSource(path="x.csv", seperator=";")


def test_demo_data_loads():
    from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV

    train, test = load(CsvSource(path=TRAIN_CSV)), load(CsvSource(path=TEST_CSV))
    assert isinstance(train, pd.DataFrame)
    # Deliberate inconsistencies kept for the future train/test key.
    assert "Survived" in train and "Survived" not in test
    assert set(test["Embarked"]) - set(train["Embarked"])
    assert train["Age"].dtype != test["Age"].dtype


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Users\x\f.csv", "/mnt/c/Users/x/f.csv"),
        ("C:/Users/x/f.csv", "/mnt/c/Users/x/f.csv"),
        (r"d:\data\f.csv", "/mnt/d/data/f.csv"),
        ("/home/x/f.csv", "/home/x/f.csv"),
        ("data/f.csv", "data/f.csv"),
    ],
)
def test_resolve_windows_path_on_linux(path, expected):
    assert resolve_path(path, platform="linux", exists=lambda _: False) == expected


def test_resolve_path_keeps_existing_or_non_linux():
    win = r"C:\Users\x\f.csv"
    assert resolve_path(win, platform="win32", exists=lambda _: False) == win
    assert resolve_path(win, platform="linux", exists=lambda _: True) == win


def test_windows_path_missing_everywhere_is_source_error():
    with pytest.raises(SourceError, match="not found"):
        load(CsvSource(path=r"Q:\nope\f.csv"))


@pytest.mark.parametrize("sep", [",", ";", "\t", "|"])
def test_sniff_sep(sep):
    assert sniff_sep(f"a{sep}b{sep}c\n1{sep}x{sep}2\n3{sep}y{sep}4\n") == sep


def test_sniff_sep_ignores_truncated_last_line():
    line = "1;22;333\n"
    sample = ("a;b;c\n" + line * SNIFF_CHARS)[:SNIFF_CHARS]
    assert sniff_sep(sample) == ";"


def test_sniff_sep_fails_gracefully():
    assert sniff_sep("") is None


def test_auto_sep_on_big_file(tmp_path):
    path = tmp_path / "big.csv"
    rows = "".join(f"{i};name {i};{i / 2}\n" for i in range(20_000))
    path.write_text("id;name;v\n" + rows)
    df = load(CsvSource(path=str(path)))
    assert df.shape == (20_000, 3)
    assert df["v"].iloc[-1] == 9_999.5


def test_auto_sep_falls_back_to_python_engine(tmp_path, monkeypatch):
    import dtk_engine.sources.csv_pandas as mod

    monkeypatch.setattr(mod, "sniff_sep", lambda _: None)
    path = tmp_path / "d.csv"
    path.write_text("a;b\n1;2\n3;4\n")
    assert list(load(CsvSource(path=str(path))).columns) == ["a", "b"]


def test_single_column_file(tmp_path):
    path = tmp_path / "y.csv"
    path.write_text("target\n0.5\n1.5\n")
    df = load(CsvSource(path=str(path)))
    assert list(df.columns) == ["target"] and df["target"].tolist() == [0.5, 1.5]


def _csv(tmp_path, content: str | bytes, name: str = "d.csv") -> str:
    path = tmp_path / name
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)
    return str(path)


@pytest.mark.parametrize("encoding", ["cp1252", "latin-1", "utf-16"])
def test_auto_encoding(tmp_path, encoding):
    path = _csv(tmp_path, "nom,âge\nHélène,34\nZoé,41\n".encode(encoding))
    df = load(CsvSource(path=path))
    assert list(df.columns) == ["nom", "âge"]
    assert df["nom"].tolist() == ["Hélène", "Zoé"]


def test_utf8_bom_is_stripped(tmp_path):
    path = _csv(tmp_path, b"\xef\xbb\xbfa,b\n1,2\n")
    assert list(load(CsvSource(path=path)).columns) == ["a", "b"]
    assert list(load(CsvSource(path=path, encoding="utf-8")).columns) == ["a", "b"]


def test_explicit_encoding_still_strict(tmp_path):
    path = _csv(tmp_path, "a\né\n".encode("cp1252"))
    with pytest.raises(SourceError, match="utf-8"):
        load(CsvSource(path=path, encoding="utf-8"))


def test_auto_decimal_comma(tmp_path):
    path = _csv(tmp_path, "id;prix;ville\n1;12,5;Paris\n2;8,75;Lyon\n3;10;Nice\n")
    df = load(CsvSource(path=path))
    assert df["prix"].tolist() == [12.5, 8.75, 10.0]


@pytest.mark.parametrize(
    "content",
    ["a;b\n1.5;x\n2.5;y\n", 'a,b\n1.5,"1,5"\n2.5,"2,5"\n'],
)
def test_auto_decimal_keeps_dot(tmp_path, content):
    assert load(CsvSource(path=_csv(tmp_path, content)))["a"].tolist() == [1.5, 2.5]


def test_leading_zeros_kept(tmp_path):
    path = _csv(tmp_path, 'zip,n,code\n"08123",1,10\n02139,2,11\n,3,12\n')
    df = load(CsvSource(path=path))
    assert df["zip"].tolist()[:2] == ["08123", "02139"]
    assert df["zip"].isna().iloc[2]
    assert df["n"].tolist() == [1, 2, 3] and df["code"].tolist() == [10, 11, 12]
    assert load(CsvSource(path=path, keep_leading_zeros=False))["zip"].iloc[0] == 8123


def test_leading_zeros_with_usecols_and_pinned_dtype(tmp_path):
    path = _csv(tmp_path, "a,zip,b\n1,08123,007\n2,02139,008\n")
    df = load(CsvSource(path=path, usecols=["zip", "b"], dtype={"b": "int64"}))
    assert df["zip"].tolist() == ["08123", "02139"]
    assert df["b"].tolist() == [7, 8]


def test_decimal_zero_is_not_a_leading_zero(tmp_path):
    path = _csv(tmp_path, "v,w\n0.5,0\n0.25,01.5\n")
    df = load(CsvSource(path=path))
    assert df["v"].tolist() == [0.5, 0.25] and df["w"].tolist() == [0.0, 1.5]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ('id,name\n1,"Hey, I missed " it"\n', "text after a closing quote"),
        ('id,name,n\n1,"never closed,3\n2,x,4\n', "unclosed quote"),
        ("a,b,c\n1,2,3\n4,5,6,7\n", "line 3: 4 fields, expected 3"),
        ("a,b,c\n1,2,3,4\n", "line 2: 4 fields, expected 3"),
    ],
)
def test_malformed_rows_fail_with_line(tmp_path, content, reason):
    with pytest.raises(SourceError, match="on_bad_lines") as exc:
        load(CsvSource(path=_csv(tmp_path, content)))
    assert reason in str(exc.value)


def test_benign_rows_still_load(tmp_path):
    content = 'a,b,c\n1,This "q" x,3\n2,"multi\nline",4\n\n5,"say ""hi""",\n6,7\n'
    df = load(CsvSource(path=_csv(tmp_path, content)))
    assert df["b"].tolist()[:3] == ['This "q" x', "multi\nline", 'say "hi"']
    assert df.shape == (4, 3)


@pytest.mark.parametrize("mode", ["warn", "skip"])
def test_on_bad_lines_recovers(tmp_path, mode):
    path = _csv(tmp_path, "a,b\n1,2\n3,4,5\n6,7\n")
    if mode == "warn":
        with pytest.warns(pd.errors.ParserWarning):
            df = load(CsvSource(path=path, on_bad_lines=mode))
    else:
        df = load(CsvSource(path=path, on_bad_lines=mode))
    assert df["a"].tolist() == [1, 6]


def test_bad_line_after_title_rows_counts_from_header(tmp_path):
    path = _csv(tmp_path, "Title\n\na,b\n1,2\n")
    assert load(CsvSource(path=path, header=1)).shape == (1, 2)


def test_sniff_sep_on_two_line_file():
    assert sniff_sep("foo;bar;baz\n1;2\n") == ";"


def test_store_b_title_line_does_not_decide_delimiter():
    """Course store_b.csv: title line + ';' + decimal ',' → 3×4 numeric prices."""
    from pathlib import Path

    from dtk_engine import api, run_key

    path = Path(__file__).resolve().parent.parent / "fixtures" / "store_b.csv"
    text = path.read_text()
    assert sniff_sep(text) == ";"
    m = run_key("file_inspect", {"path": str(path)})["metrics"]
    spec = __import__("json").loads(m["load_spec"])
    assert spec["sep"] == ";" and spec["decimal"] == "," and spec["header"] == 1
    df = api.load(spec)
    assert df.shape == (3, 4)
    assert list(df.columns) == ["id", "product", "price", "qty"]
    assert df["price"].tolist() == [1.5, 0.95, 3.2]
