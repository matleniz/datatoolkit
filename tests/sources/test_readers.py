import json

import pandas as pd
import pytest
from sqlalchemy import create_engine

from dtk_engine import api
from dtk_engine.errors import SourceError
from dtk_engine.sources import (
    CsvSource,
    ExcelSource,
    JsonSource,
    ParquetSource,
    SqlSource,
    load,
)


def test_csv_options(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("zip,when,v,skip\n01,2024-01-02,N/A,x\n02,2024-02-03,3,y\n")
    df = load(
        CsvSource(
            path=str(path),
            na_values=["N/A"],
            dtype={"zip": "str"},
            parse_dates=["when"],
            usecols=["zip", "when", "v"],
        )
    )
    assert list(df.columns) == ["zip", "when", "v"]
    assert df["zip"].tolist() == ["01", "02"]
    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    assert df["v"].isna().tolist() == [True, False]


def test_parquet_columns_and_filters(tmp_path):
    path = tmp_path / "d.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")}).to_parquet(path)
    df = load(ParquetSource(path=str(path), columns=["a"], filters=[("a", ">", 1)]))
    assert df["a"].tolist() == [2, 3]
    assert list(df.columns) == ["a"]


def test_parquet_partitioned_dir(tmp_path):
    pd.DataFrame({"a": [1, 2], "p": ["u", "v"]}).to_parquet(
        tmp_path / "ds", partition_cols=["p"]
    )
    assert len(load(ParquetSource(path=str(tmp_path / "ds")))) == 2


def test_excel_sheet_by_name_and_index(tmp_path):
    path = tmp_path / "d.xlsx"
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="one", index=False)
        pd.DataFrame({"b": [2, 3], "c": [4, 5]}).to_excel(
            w, sheet_name="two", index=False
        )
    assert list(load(ExcelSource(path=str(path))).columns) == ["a"]
    df = load(ExcelSource(path=str(path), sheet="two", usecols=["c"]))
    assert df["c"].tolist() == [4, 5]
    assert list(load(ExcelSource(path=str(path), sheet=1)).columns) == ["b", "c"]


def test_json_flatten_and_record_path(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(
        json.dumps({"data": [{"id": 1, "u": {"n": "a"}, "t": [1, 2]}, {"id": 2}]})
    )
    df = load(JsonSource(path=str(path), record_path="data"))
    assert list(df.columns) == ["id", "t", "u_n"]
    assert df["t"][0] == [1, 2]


def test_jsonl(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert api.load(path)["a"].tolist() == [1, 2]


def test_sql_sqlite(tmp_path, monkeypatch):
    db = tmp_path / "d.db"
    pd.DataFrame({"a": [1, 2, 3]}).to_sql("t", create_engine(f"sqlite:///{db}"))
    monkeypatch.setenv("DTK_TEST_URL", f"sqlite:///{db}")
    df = load(SqlSource(url_env="DTK_TEST_URL", query="SELECT a FROM t WHERE a > 1"))
    assert df["a"].tolist() == [2, 3]


def test_api_load_suffixes(tmp_path):
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "d.parquet")
    pd.DataFrame({"a": [1]}).to_excel(tmp_path / "d.xlsx", index=False)
    (tmp_path / "d.json").write_text('[{"a": 1}]')
    for name in ("d.parquet", "d.xlsx", "d.json"):
        assert api.load(tmp_path / name)["a"].tolist() == [1]


def test_errors_raise_source_error(tmp_path, monkeypatch):
    monkeypatch.delenv("DTK_NOPE", raising=False)
    monkeypatch.setenv("DTK_BAD", f"sqlite:///{tmp_path}/x.db")
    bad = tmp_path / "bad"
    bad.write_text("not a real file")
    (tmp_path / "bad.json").write_text("{oops")
    for spec in [
        ParquetSource(path=str(tmp_path / "nope.parquet")),
        ParquetSource(path=str(bad)),
        ExcelSource(path=str(tmp_path / "nope.xlsx")),
        ExcelSource(path=str(bad)),
        JsonSource(path=str(tmp_path / "nope.json")),
        JsonSource(path=str(tmp_path / "bad.json")),
        SqlSource(url_env="DTK_NOPE", query="SELECT 1"),
        SqlSource(url_env="DTK_BAD", query="SELECT * FROM missing"),
    ]:
        with pytest.raises(SourceError):
            load(spec)


def test_ndjson_suffix(tmp_path):
    path = tmp_path / "d.ndjson"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert api.load(path)["a"].tolist() == [1, 2]


def test_json_bom_and_encoding(tmp_path):
    bom = tmp_path / "bom.jsonl"
    bom.write_bytes(b'\xef\xbb\xbf{"a": 1}\n{"a": 2}\n')
    assert api.load(bom)["a"].tolist() == [1, 2]
    plain = tmp_path / "bom.json"
    plain.write_bytes(b'\xef\xbb\xbf[{"a": 1}]')
    assert api.load(plain)["a"].tolist() == [1]
    latin = tmp_path / "latin.json"
    latin.write_bytes('[{"a": "café"}]'.encode("cp1252"))
    with pytest.raises(SourceError):
        api.load(latin)
    df = api.load({"kind": "json", "path": str(latin), "encoding": "cp1252"})
    assert df["a"].tolist() == ["café"]
    with pytest.raises(SourceError, match="unknown encoding"):
        api.load({"kind": "json", "path": str(latin), "encoding": "nope"})
