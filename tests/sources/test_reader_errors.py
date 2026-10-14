"""Error paths of the file readers: missing / empty / garbage input -> SourceError
naming the file, without parser internals (exception class names, tracebacks)."""

import pandas as pd
import pytest

from dtk_engine.errors import SourceError
from dtk_engine.sources import ExcelSource, JsonSource, ParquetSource, SqlSource, load

SPECS = {
    "excel": (ExcelSource, "data.xlsx"),
    "json": (JsonSource, "data.json"),
    "parquet": (ParquetSource, "data.parquet"),
}

# Parser internals that must not leak to the user.
INTERNALS = (
    "Traceback",
    "BadZipFile",
    "InvalidFileException",
    "ArrowInvalid",
    "ArrowException",
    "JSONDecodeError",
    "Expecting value",
    "pyarrow",
    "openpyxl",
    "File is not a zip file",
    "Parquet magic bytes",
    "Buffer>",
)


def _valid_xlsx(path):
    pd.DataFrame({"a": range(50), "b": list("x" * 50)}).to_excel(path, index=False)


def _make(kind, flavor, path):
    if flavor == "missing":
        return
    if flavor == "empty":
        path.write_bytes(b"")
    elif flavor == "garbage":
        path.write_bytes(b"\x00\x01garbage \xff\xfe not a real file" * 20)
    elif flavor == "truncated":
        full = path.with_name("full_" + path.name)
        if kind == "excel":
            _valid_xlsx(full)
        elif kind == "parquet":
            pd.DataFrame({"a": range(500)}).to_parquet(full)
        else:
            full.write_text('{"records": [{"a": 1}, {"a": 2}, {"a": 3}]}')
        raw = full.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])


@pytest.mark.parametrize("flavor", ["missing", "empty", "garbage", "truncated"])
@pytest.mark.parametrize("kind", list(SPECS))
def test_bad_file_is_a_clean_source_error(tmp_path, kind, flavor):
    cls, name = SPECS[kind]
    path = tmp_path / name
    _make(kind, flavor, path)
    with pytest.raises(SourceError) as info:
        load(cls(path=str(path)))
    msg = str(info.value)
    assert name in msg
    for token in INTERNALS:
        assert token not in msg, msg


def test_json_lines_garbage_is_a_clean_source_error(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(SourceError) as info:
        load(JsonSource(path=str(path), lines=True))
    assert "data.jsonl" in str(info.value)
    assert "JSONDecodeError" not in str(info.value)


def test_sql_unset_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("DTK_TEST_URL", raising=False)
    with pytest.raises(SourceError, match="DTK_TEST_URL"):
        load(SqlSource(url_env="DTK_TEST_URL", query="select 1"))


def test_sql_missing_database_file_does_not_leak_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path}/nope/missing.db"
    monkeypatch.setenv("DTK_TEST_URL", url)
    with pytest.raises(SourceError) as info:
        load(SqlSource(url_env="DTK_TEST_URL", query="select 1"))
    assert "sqlite:///" not in str(info.value)
    assert "DTK_TEST_URL" in str(info.value)


def test_sql_garbage_database_file(tmp_path, monkeypatch):
    db = tmp_path / "garbage.db"
    db.write_bytes(b"this is not a sqlite database" * 50)
    monkeypatch.setenv("DTK_TEST_URL", f"sqlite:///{db}")
    with pytest.raises(SourceError) as info:
        load(SqlSource(url_env="DTK_TEST_URL", query="select * from t"))
    assert "DTK_TEST_URL" in str(info.value)
    assert str(db) not in str(info.value)
    assert "Traceback" not in str(info.value)


def test_sql_empty_database_file(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    db.write_bytes(b"")
    monkeypatch.setenv("DTK_TEST_URL", f"sqlite:///{db}")
    with pytest.raises(SourceError, match="DTK_TEST_URL"):
        load(SqlSource(url_env="DTK_TEST_URL", query="select * from t"))
