"""Sources: a JSON SourceSpec -> load() -> pd.DataFrame. Never knows about keys."""

from . import csv_pandas  # noqa: F401  (registers the csv reader)
from .registry import load, reader
from .spec import CsvSource, SourceSpec

__all__ = ["CsvSource", "SourceSpec", "load", "reader"]
