"""CSV reader backed by pandas.read_csv."""

from __future__ import annotations

import csv

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import CsvSource


@reader("csv")
def read_csv(spec: CsvSource) -> pd.DataFrame:
    sniff = spec.sep == "auto"
    try:
        return pd.read_csv(
            spec.path,
            sep=None if sniff else spec.sep,
            engine="python" if sniff else "c",
            encoding=spec.encoding,
            decimal=spec.decimal,
            header=spec.header,
        )
    except FileNotFoundError:
        raise SourceError(f"csv source not found: {spec.path}") from None
    # ParserError, EmptyDataError and UnicodeDecodeError are ValueErrors;
    # LookupError = unknown encoding; csv.Error = sniffing failed (sep="auto").
    except (OSError, ValueError, LookupError, csv.Error) as exc:
        raise SourceError(f"cannot read csv {spec.path}: {exc}") from exc
