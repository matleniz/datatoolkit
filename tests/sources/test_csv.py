import pandas as pd
import pytest
from dtk_engine.errors import SourceError
from dtk_engine.sources import CsvSource, load


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
