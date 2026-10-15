"""Excel reader backed by pandas.read_excel (openpyxl)."""

from __future__ import annotations

from typing import Any
from zipfile import BadZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

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
    return read_excel_checked(
        path,
        spec.path,
        sheet_name=spec.sheet,
        header=spec.header,
        usecols=spec.usecols,
    )


def read_excel_checked(path: str, shown: str, **kw: Any) -> Any:
    """``pd.read_excel`` (openpyxl) with file/format errors mapped to SourceError."""
    try:
        return pd.read_excel(path, engine="openpyxl", **kw)
    except FileNotFoundError:
        raise SourceError(f"excel source not found: {shown}") from None
    except (BadZipFile, InvalidFileException):
        raise SourceError(
            f"cannot read excel {shown}: not a valid .xlsx file "
            "(empty, truncated or corrupted)"
        ) from None
    except ImportError:
        raise SourceError(
            f"cannot read excel {shown}: openpyxl is not installed (uv sync)"
        ) from None
    # Unknown sheet / bad header / unreadable cells -> ValueError; the rest is I/O.
    except (ValueError, OSError) as exc:
        raise SourceError(f"cannot read excel {shown}: {exc}") from exc
