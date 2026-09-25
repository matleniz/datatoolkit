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


def test_missing_file(tmp_path):
    with pytest.raises(SourceError):
        run_key("file_inspect", {"path": str(tmp_path / "nope.csv")})


def test_json_suggests_record_path(tmp_path):
    path = tmp_path / "api.json"
    path.write_text(
        '{"meta": {"page": 1}, "data": {"items": [{"id": 1}, {"id": 2}], "x": [{"k": 1}]}}'
    )
    result = run_key("file_inspect", {"path": str(path)})
    assert result["metrics"]["suggested_record_path"] == "data.items"
    records = result["tables"][-1]["records"]
    assert [r["record_path"] for r in records] == ["data.items", "data.x"]
    flat = tmp_path / "flat.json"
    flat.write_text('[{"a": 1}]')
    assert (
        "suggested_record_path"
        not in run_key("file_inspect", {"path": str(flat)})["metrics"]
    )


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
