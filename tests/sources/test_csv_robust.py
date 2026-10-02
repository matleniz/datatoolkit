"""csv_robust: junk header lines and mixed separators."""

import json
from pathlib import Path

import pandas as pd
import pytest

from dtk_engine.keys.file_inspect import Params, run
from dtk_engine.sources.csv_pandas import MixedSeparatorError, read_csv
from dtk_engine.sources.spec import CsvSource

FIX = Path(__file__).parent.parent / "fixtures" / "csv_robust"
COLS = ["id", "name", "qty"]


def _inspect(name: str) -> dict:
    return run(Params(path=str(FIX / name))).metrics


def _load_spec(metrics: dict) -> CsvSource:
    return CsvSource(**json.loads(metrics["load_spec"]))


def test_skiprows_auto_drops_junk_lines():
    df = read_csv(CsvSource(path=str(FIX / "junk_header.csv"), skiprows="auto"))
    assert list(df.columns) == COLS and len(df) == 3


def test_skiprows_explicit_count():
    df = read_csv(CsvSource(path=str(FIX / "junk_header.csv"), skiprows=3))
    assert list(df.columns) == COLS and len(df) == 3


def test_skiprows_auto_fixes_wrong_sniffed_separator():
    path = str(FIX / "junk_misleads_sniffer.csv")
    df = read_csv(CsvSource(path=path, skiprows="auto"))
    assert list(df.columns) == COLS and df["qty"].tolist() == [5, 6]


def test_skiprows_default_is_unchanged(tmp_path):
    path = tmp_path / "plain.csv"
    path.write_text("a,b\n1,2\n")
    assert list(read_csv(CsvSource(path=str(path), skiprows="auto")).columns) == ["a", "b"]
    assert read_csv(CsvSource(path=str(path))).shape == (1, 2)


def test_inspect_junk_lines_load_spec_loads():
    m = _inspect("junk_header.csv")
    assert m["title_lines_above_header"] == 3
    df = read_csv(_load_spec(m))
    assert list(df.columns) == COLS and len(df) == 3


def test_inspect_junk_misleading_sniffer_reports_and_loads():
    m = _inspect("junk_misleads_sniffer.csv")
    assert m["delimiter"] == "';'"
    assert m["junk_lines"].startswith("3 line(s) above the header")
    spec = _load_spec(m)
    assert spec.skiprows == 3
    df = read_csv(spec)
    assert list(df.columns) == COLS and len(df) == 2


def test_mixed_sep_error_names_the_lines():
    with pytest.raises(MixedSeparatorError) as exc:
        read_csv(CsvSource(path=str(FIX / "mixed_sep.csv"), sep=","))
    assert exc.value.lines == [3, 5]
    msg = str(exc.value)
    assert "';' on line(s) 3, 5" in msg and "mixed_sep='normalize'" in msg


def test_mixed_sep_error_is_a_source_error():
    from dtk_engine.errors import SourceError

    with pytest.raises(SourceError):
        read_csv(CsvSource(path=str(FIX / "mixed_sep.csv")))


def test_mixed_sep_normalize_loads_every_row():
    spec = CsvSource(path=str(FIX / "mixed_sep.csv"), mixed_sep="normalize")
    df = read_csv(spec)
    assert list(df.columns) == COLS
    assert df["name"].tolist() == ["apple", "pear", "plum", "fig", "kiwi"]
    assert df["qty"].tolist() == [5, 6, 7, 8, 9]


def test_mixed_sep_ignore_keeps_old_behaviour():
    df = read_csv(CsvSource(path=str(FIX / "mixed_sep.csv"), mixed_sep="ignore"))
    assert len(df) == 5


RAGGED = {
    "tab": "a\tb\tc\n1\t2\t3\n4\t5,6,7\n",
    "comma": "id,note,v\n1,ok,3\n2,see a;b;c\n3,x,4\n",
}


@pytest.mark.parametrize("name", RAGGED)
def test_ragged_row_holding_sep_loads_as_before(tmp_path, name):
    path = tmp_path / "f.csv"
    path.write_text(RAGGED[name])
    sep = "\t" if name == "tab" else ","
    expected = pd.read_csv(path, sep=sep)
    pd.testing.assert_frame_equal(read_csv(CsvSource(path=str(path))), expected)
    assert run(Params(path=str(path))).metrics["mixed_separators"] == "none"


def test_mixed_sep_error_line_numbers_count_skipped_junk(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("junk\njunk2\nid,name,qty\n1,a,5\n2;b;6\n3,c,7\n")
    with pytest.raises(MixedSeparatorError) as exc:
        read_csv(CsvSource(path=str(path), skiprows=2))
    assert exc.value.lines == [5]


def test_quoted_lines_are_not_mixed(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text('id,name,qty\n1,"a;b;c",5\n2,b,6\n')
    assert len(read_csv(CsvSource(path=str(path)))) == 2


def test_inspect_mixed_reports_and_load_spec_loads():
    m = _inspect("mixed_sep.csv")
    assert "';': line(s) 3, 5" in m["mixed_separators"]
    assert json.loads(m["separator_line_counts"]) == {",": 4, ";": 2}
    spec = _load_spec(m)
    assert spec.mixed_sep == "normalize"
    assert read_csv(spec)["qty"].tolist() == [5, 6, 7, 8, 9]


def test_inspect_clean_file_reports_no_mixed_separators(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("a,b\n1,2\n")
    m = run(Params(path=str(path))).metrics
    assert m["mixed_separators"] == "none" and "junk_lines" not in m
