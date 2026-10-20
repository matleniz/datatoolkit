"""Sources: a JSON SourceSpec -> load() -> pd.DataFrame. Never knows about keys."""

from . import (  # noqa: F401  (registers the readers)
    csv_pandas,
    excel,
    json_reader,
    parquet,
    sql,
)
from .registry import load, reader, string_columns
from .spec import (
    CsvSource,
    DatasetSource,
    ExcelSource,
    FileSourceSpec,
    JsonSource,
    ParquetSource,
    SourceSpec,
    SqlSource,
)

__all__ = [
    "CsvSource",
    "DatasetSource",
    "ExcelSource",
    "FileSourceSpec",
    "JsonSource",
    "ParquetSource",
    "SourceSpec",
    "SqlSource",
    "load",
    "reader",
    "string_columns",
]
