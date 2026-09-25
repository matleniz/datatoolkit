"""Parquet reader backed by pyarrow (file or partitioned directory)."""

from __future__ import annotations

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.csv_pandas import resolve_path
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import ParquetSource


@reader("parquet")
def read_parquet(spec: ParquetSource) -> pd.DataFrame:
    path = resolve_path(spec.path)
    try:
        return pd.read_parquet(
            path,
            columns=spec.columns,
            filters=[tuple(f) for f in spec.filters] if spec.filters else None,
        )
    except FileNotFoundError:
        raise SourceError(f"parquet source not found: {spec.path}") from None
    # ArrowInvalid / ArrowKeyError (unknown column) are ValueError / KeyError.
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SourceError(f"cannot read parquet {spec.path}: {exc}") from exc
