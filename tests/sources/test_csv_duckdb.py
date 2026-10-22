"""CsvSource(engine="duckdb"): the optional ``large`` extra (skipped without duckdb)."""

import subprocess
import sys

import pandas as pd
import pytest

from dtk_engine.errors import SourceError
from dtk_engine.sources import CsvSource, csv_duckdb, load


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))


def read(path, **kw):
    return load(CsvSource(path=str(path), engine="duckdb", **kw))


def write(tmp_path, data: bytes, name="t.csv"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_default_install_never_imports_duckdb():
    code = (
        "import sys, dtk_engine.api, dtk_engine.contract;"
        "from dtk_engine.sources import CsvSource, load;"
        "import tempfile, os; p = os.path.join(tempfile.mkdtemp(), 'a.csv');"
        "open(p, 'w').write('a,b\\n1,2\\n'); load(CsvSource(path=p));"
        "print('duckdb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_missing_extra_names_it(tmp_path, monkeypatch):
    path = write(tmp_path, b"a\n1\n")
    monkeypatch.setitem(sys.modules, "duckdb", None)  # `import duckdb` -> ImportError
    with pytest.raises(SourceError, match=r"\[large\]"):
        read(path)


pytest.importorskip("duckdb")


def test_matches_pandas_on_a_plain_file(tmp_path):
    path = write(tmp_path, b"a,b,c,d\n1,x,2.5,true\n2,y,,false\n3,,4.5,true\n")
    got = read(path)
    want = load(CsvSource(path=str(path)))
    assert list(got.columns) == list(want.columns)
    assert got["a"].tolist() == [1, 2, 3]
    assert got["c"].isna().tolist() == [False, True, False]
    assert got["b"].isna().tolist() == [False, False, True]
    assert got["d"].tolist() == want["d"].tolist()


def test_tricky_file(tmp_path):
    data = (
        b"\xef\xbb\xbfzip,d,e,q,m,n\n"
        b'00123,2020-01-05,,"a\nb",1,x\n'
        b'00456,2021-02-03,"",c,2.5,NA\n'
        b'7,2022-03-04,z,"d,e",abc,3\n'
    )
    df = read(write(tmp_path, data))
    assert list(df.columns) == ["zip", "d", "e", "q", "m", "n"]  # BOM dropped
    assert df["zip"].tolist() == ["00123", "00456", "7"]  # leading zeros kept
    assert df["d"].tolist() == ["2020-01-05", "2021-02-03", "2022-03-04"]  # str, not date
    assert df["e"].isna().tolist() == [True, True, False]  # "" and quoted "" are missing
    assert df["q"].tolist() == ["a\nb", "c", "d,e"]  # quoted newline and comma
    assert df["m"].tolist() == ["1", "2.5", "abc"]  # mixed -> strings
    assert df["n"].isna().tolist() == [False, True, False]  # NA string is missing


def test_leading_zeros_found_after_the_type_sample(tmp_path):
    rows = "".join(f"{1000 + k}\n" for k in range(csv_duckdb.SAMPLE_ROWS + 10))
    path = write(tmp_path, f"z\n{rows}0042\n".encode())
    df = read(path)
    assert df["z"].iloc[-1] == "0042" and df["z"].iloc[0] == "1000"
    off = read(path, keep_leading_zeros=False)
    assert off["z"].iloc[-1] == 42


def test_type_breaking_value_after_the_sample_makes_a_string_column(tmp_path):
    rows = "".join(f"{k}\n" for k in range(csv_duckdb.SAMPLE_ROWS + 10))
    df = read(write(tmp_path, f"i\n{rows}abc\n".encode()))
    assert df["i"].iloc[-1] == "abc" and df["i"].iloc[5] == "5"


def test_latin1_is_never_guessed(tmp_path):
    path = write(tmp_path, "name;v\ncafé;1\n".encode("latin-1"))
    with pytest.raises(SourceError, match="latin-1"):
        read(path)
    assert read(path, encoding="latin-1")["name"].tolist() == ["café"]
    assert read(path, encoding="iso-8859-1")["v"].tolist() == [1]


def test_utf16_with_bom(tmp_path):
    path = write(tmp_path, "a,b\n1,é\n".encode("utf-16"))
    df = read(path)
    assert df["b"].tolist() == ["é"]


def test_other_encodings_are_refused(tmp_path):
    path = write(tmp_path, "a\né\n".encode("cp1252"))
    with pytest.raises(SourceError, match="utf-8, utf-16 .* and latin-1"):
        read(path, encoding="cp1252")
    with pytest.raises(SourceError, match="unknown encoding"):
        read(path, encoding="nope")


@pytest.mark.parametrize(
    ("kw", "name"),
    [
        ({"parse_dates": ["d"]}, "parse_dates"),
        ({"mixed_sep": "normalize"}, "mixed_sep"),
        ({"on_bad_lines": "skip"}, "on_bad_lines"),
        ({"skiprows": "auto"}, "skiprows"),
        ({"header": 2}, "header"),
    ],
)
def test_unsupported_options_are_refused_not_ignored(tmp_path, kw, name):
    path = write(tmp_path, b"d\n2020-01-01\n")
    with pytest.raises(SourceError, match=name):
        read(path, **kw)


def test_options_honoured(tmp_path):
    path = write(tmp_path, b"junk\nid;price;x\n001;1,5;a\n002;2,5;b\n")
    df = read(path, skiprows=1, sep=";", decimal=",", usecols=["price", "id"], dtype={"id": "str"})
    assert list(df.columns) == ["price", "id"]
    assert df["price"].tolist() == [1.5, 2.5]
    assert df["id"].tolist() == ["001", "002"]
    na = read(write(tmp_path, b"a,b\n-,1\nx,2\n", "n.csv"), na_values=["-"])
    assert na["a"].isna().tolist() == [True, False]
    with pytest.raises(SourceError, match="dtype 'category'"):
        read(path, skiprows=1, sep=";", dtype={"id": "category"})


def test_no_header_and_duplicate_names(tmp_path):
    assert list(read(write(tmp_path, b"1,2\n3,4\n"), header=None).columns) == ["column0", "column1"]
    assert list(read(write(tmp_path, b"a,a\n1,2\n", "d.csv")).columns) == ["a", "a_1"]


def test_bad_file_errors_are_source_errors(tmp_path):
    with pytest.raises(SourceError, match="not found"):
        read(tmp_path / "missing.csv")
    with pytest.raises(SourceError, match="cannot read csv"):
        read(write(tmp_path, b"a,b\n1,2,3\n"))  # ragged row
    with pytest.raises(SourceError, match="cannot read csv"):
        read(write(tmp_path, b"a,b\n1,2\n", "u.csv"), usecols=["nope"])


def test_converted_once_then_cached_and_invalidated(tmp_path):
    path = write(tmp_path, b"a\n1\n")
    spec = CsvSource(path=str(path), engine="duckdb")
    first = csv_duckdb.csv_to_parquet(spec)
    assert first.parent == csv_duckdb.cache_dir() and first.suffix == ".parquet"
    mtime = first.stat().st_mtime_ns
    assert csv_duckdb.csv_to_parquet(spec) == first and first.stat().st_mtime_ns == mtime
    assert pd.read_parquet(first)["a"].tolist() == [1]  # plain parquet, ParquetSource-readable
    path.write_bytes(b"a\n1\n2\n")
    second = csv_duckdb.csv_to_parquet(spec)
    assert second != first and not first.exists()  # one entry per source path
    assert len(list(csv_duckdb.cache_dir().iterdir())) == 1  # no partial file left


@pytest.mark.parametrize("limit", ["1GB", "256MB"])
def test_row_order_is_kept(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(csv_duckdb, "MEMORY_LIMIT", limit)
    n = 600_000  # enough chunks for DuckDB to reorder them when it may
    rows = "".join(f"{k},s{k % 7},{k * 0.5}\n" for k in range(n))
    df = read(write(tmp_path, f"id,s,x\n{rows}".encode()))
    assert len(df) == n
    assert (df["id"].to_numpy() == range(n)).all()
