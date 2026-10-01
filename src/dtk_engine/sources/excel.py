"""Excel reader backed by pandas.read_excel (openpyxl)."""

from __future__ import annotations

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.csv_pandas import resolve_path
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import ExcelSource


@reader("excel")
def read_excel(spec: ExcelSource) -> pd.DataFrame:
    path = resolve_path(spec.path)
    if path.lower().endswith(".xls"):
        raise SourceError(
            f"cannot read excel {spec.path}: legacy .xls is not supported, "
            "only .xlsx is supported (re-save the file as .xlsx)"
        )
    try:
        return pd.read_excel(
            path,
            sheet_name=spec.sheet,
            header=spec.header,
            usecols=spec.usecols,
            engine="openpyxl",
        )
    except FileNotFoundError:
        raise SourceError(f"excel source not found: {spec.path}") from None
    # Unknown sheet -> ValueError; bad zip / not an xlsx -> BadZipFile / InvalidFileException.
    except Exception as exc:
        raise SourceError(f"cannot read excel {spec.path}: {exc}") from exc
