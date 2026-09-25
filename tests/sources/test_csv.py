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
