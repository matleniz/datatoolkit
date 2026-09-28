import json

import pandas as pd
import pytest

from dtk_engine import api, run_key
from dtk_engine.errors import SourceError


def _facts(tmp_path, content: bytes) -> dict:
    path = tmp_path / "f.csv"
    path.write_bytes(content)
    return run_key("file_inspect", {"path": str(path)})["metrics"]


def test_runs_on_demo_data():
    m = run_key("file_inspect", {})["metrics"]
    assert m["encoding_guess"] == "utf-8"
    assert m["bom"] == "none"
    assert m["delimiter"] == "','"


def test_bom_cp1252_crlf_semicolon(tmp_path):
    m = _facts(tmp_path, "a;b\r\nné;2\r\n".encode("cp1252"))
    assert (m["encoding_guess"], m["line_endings"], m["delimiter"]) == (
        "cp1252",
        "CRLF",
        "';'",
    )
    assert _facts(tmp_path, b"\xef\xbb\xbfa,b\n1,2\n")["bom"] == "utf-8-sig"


def test_title_lines_and_unnamed(tmp_path):
    m = _facts(tmp_path, b"Report 2024\n\na,b,,d\n1,2,3,4\n5,6,7,8\n")
    assert m["header_line"] == 3
    assert m["title_lines_above_header"] == 2
    assert m["unnamed_columns"] == "Unnamed: 2"


def test_excel_lists_sheets(tmp_path):
    path = tmp_path / "f.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1, 2]}).to_excel(w, sheet_name="s1", index=False)
    result = run_key("file_inspect", {"path": str(path)})
    assert result["tables"][0]["title"] == "sheets"
    assert len(result["tables"][0]["records"]) == 1
    spec = json.loads(result["metrics"]["load_spec"])
    assert spec == {
        "kind": "excel",
        "path": str(path),
        "sheet": "s1",
        "header": 0,
    }
    assert list(api.load(spec).columns) == ["a"]


def test_parquet_load_spec(tmp_path):
    path = tmp_path / "t.parquet"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(path)
    m = run_key("file_inspect", {"path": str(path)})["metrics"]
    spec = json.loads(m["load_spec"])
    assert spec == {"kind": "parquet", "path": str(path)}
    assert "first_bytes" not in m
    df = api.load(spec)
    assert list(df.columns) == ["a", "b"] and len(df) == 2


def test_jsonl_load_spec(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"event_id": "e0", "n": 1}\n{"event_id": "e1", "n": 2}\n')
    m = run_key("file_inspect", {"path": str(path)})["metrics"]
    spec = json.loads(m["load_spec"])
    assert spec == {"kind": "json", "path": str(path), "lines": True}
    df = api.load(spec)
    assert list(df.columns) == ["event_id", "n"] and len(df) == 2


def test_json_load_spec_keeps_record_paths(tmp_path):
    path = tmp_path / "api.json"
    path.write_text(
        '{"meta": {"page": 1}, "data": {"items": [{"id": 1}, {"id": 2}], "x": [{"k": 1}]}}'
    )
    result = run_key("file_inspect", {"path": str(path)})
    spec = json.loads(result["metrics"]["load_spec"])
    assert spec["kind"] == "json" and spec["lines"] is False
    assert spec["record_path"] == "data.items"
    assert result["metrics"]["suggested_record_path"] == "data.items"
    assert [r["record_path"] for r in result["tables"][-1]["records"]] == [
        "data.items",
        "data.x",
    ]
    assert list(api.load(spec).columns) == ["id"]
    flat = tmp_path / "flat.json"
    flat.write_text('[{"a": 1}]')
    flat_m = run_key("file_inspect", {"path": str(flat)})["metrics"]
    assert "suggested_record_path" not in flat_m
    assert json.loads(flat_m["load_spec"]) == {
        "kind": "json",
        "path": str(flat),
        "lines": False,
    }


def test_missing_file(tmp_path):
    with pytest.raises(SourceError):
        run_key("file_inspect", {"path": str(tmp_path / "nope.csv")})


def test_excel_suggests_header_row(tmp_path):
    path = tmp_path / "bom.xlsx"
    rows = [
        ["Company BOM Export", None, None],
        ["Rev 3", None, None],
        ["Part", "Qty", "MPN"],
        ["r1", 2, "x1"],
        ["r2", 3, "x2"],
    ]
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    sheet = run_key("file_inspect", {"path": str(path)})["tables"][0]["records"][0]
    assert sheet["suggested_header"] == 2
    df = api.load(
        {"kind": "excel", "path": str(path), "header": sheet["suggested_header"]}
    )
    assert list(df.columns) == ["Part", "Qty", "MPN"]


def _excel_header(tmp_path, rows) -> int:
    path = tmp_path / "s.xlsx"
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    records = run_key("file_inspect", {"path": str(path)})["tables"][0]["records"]
    return records[0]["suggested_header"]


def test_excel_header_with_missing_cells(tmp_path):
    rows = [["PassengerId", "Name", "Age", "Cabin"]]
    rows += [[i, f"n{i}", 20 + i, None] for i in range(8)] + [[9, "n9", 30, "C85"]]
    assert _excel_header(tmp_path, rows) == 0


def test_excel_header_with_nan_heavy_numeric_column(tmp_path):
    rows = [["id", "x", "y"]] + [[i, None, float(i)] for i in range(6)]
    rows += [[7, 1.5, 7.0]]
    assert _excel_header(tmp_path, rows) == 0


def test_excel_header_numeric_labels(tmp_path):
    rows = [["name", 2020, 2021], ["a", 1, 2], ["b", 3, 4]]
    assert _excel_header(tmp_path, rows) == 0


def test_quoted_commas_do_not_move_the_header(tmp_path):
    content = (
        b'sku,name,price\nA,"Go, 2nd ed.",1\nB,"x, y, z",2\nC,"a,b,c,d,e",3\nD,"q",4\n'
    )
    m = _facts(tmp_path, content)
    assert (m["header_line"], m["title_lines_above_header"]) == (1, 0)
    assert m["unnamed_columns"] == ""


def test_multiline_quoted_field(tmp_path):
    m = _facts(tmp_path, b'Title\na,b\n1,"two\nlines"\n2,x\n3,y\n')
    assert m["header_line"] == 2


@pytest.mark.parametrize(
    "content",
    [b"39,State-gov,77516\n50,Self-emp,83311\n38,Private,215646\n", b"1,2\n3,4\n5,6\n"],
)
def test_headerless_file(tmp_path, content):
    m = _facts(tmp_path, content)
    assert m["header_guess"] == "none"
    assert "header_line" not in m
    assert json.loads(m["load_spec"])["header"] is None


def test_text_only_file_assumes_header(tmp_path):
    assert _facts(tmp_path, b"a,b\nx,y\nz,w\n")["header_guess"] == "present"


def test_decimal_and_load_spec(tmp_path):
    m = _facts(tmp_path, "nom;prix\nHélène;12,5\nZoé;8,75\n".encode("cp1252"))
    assert m["decimal_guess"] == ","
    spec = json.loads(m["load_spec"])
    assert spec | {"path": "x"} == {
        "kind": "csv",
        "path": "x",
        "sep": ";",
        "encoding": "cp1252",
        "decimal": ",",
        "header": 0,
    }
    df = api.load(json.loads(m["load_spec"]))
    assert df["prix"].tolist() == [12.5, 8.75]


def test_load_spec_skips_blank_title_lines(tmp_path):
    path = tmp_path / "f.csv"
    path.write_bytes(b"Report 2024\n\na,b,c\n1,2,3\n4,5,6\n")
    m = run_key("file_inspect", {"path": str(path)})["metrics"]
    spec = json.loads(m["load_spec"])
    assert spec["header"] == 1
    assert list(api.load(spec).columns) == ["a", "b", "c"]


def test_leading_zero_columns(tmp_path):
    m = _facts(tmp_path, b"zip,n\n08123,1\n90210,2\n")
    assert m["leading_zero_columns"] == "zip"
    assert json.loads(m["load_spec"])["dtype"] == {"zip": "str"}


def test_bad_line(tmp_path):
    assert _facts(tmp_path, b"a,b\n1,2\n3,4,5\n")["bad_line"] == (
        "line 3: 3 fields, expected 2"
    )
    m = _facts(tmp_path, b'a,b\n1,"x" y\n')
    assert m["bad_line"].startswith("line 2: text after a closing quote")
    assert _facts(tmp_path, b"a,b\n1,2\n")["bad_line"] == "none"


def test_large_cp1252_file_guess_round_trips(tmp_path):
    rows = "".join(f"Île-de-France;{i},5\n" for i in range(6000))
    content = ("region;val\n" + rows).encode("cp1252")
    assert len(content) > 64 * 1024
    m = _facts(tmp_path, content)
    assert m["encoding_guess"] == "cp1252"
    df = api.load(json.loads(m["load_spec"]))
    assert df["region"].iloc[0] == "Île-de-France"


def test_large_utf8_file_cut_mid_character_stays_utf8(tmp_path):
    from dtk_engine.keys.file_inspect import SNIFF_CHARS

    head = "a;b\n" + "x" * (SNIFF_CHARS - 5)
    content = (head + "é" * 10 + "\n" * 2).encode("utf-8")
    # 'é' is 2 bytes: the sample boundary falls inside one iff the offset is odd.
    assert content[:SNIFF_CHARS][-1:] == b"\xc3"
    assert _facts(tmp_path, content)["encoding_guess"] == "utf-8"
