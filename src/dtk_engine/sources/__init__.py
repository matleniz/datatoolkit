"""Sources: a JSON SourceSpec -> load() -> pd.DataFrame. Never knows about keys."""

from . import csv_pandas, dataset  # noqa: F401  (registers the readers)
from .registry import load, reader
from .spec import CsvSource, DatasetSource, FileSourceSpec, SourceSpec

__all__ = [
    "CsvSource",
    "DatasetSource",
    "FileSourceSpec",
    "SourceSpec",
    "load",
    "reader",
]
