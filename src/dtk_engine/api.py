"""Notebook door: DataFrame in, ``Result`` (displayed by Jupyter) or DataFrame out.

    from dtk_engine import api

    train = api.load("train.csv")
    api.overview(train)                      # Result, rendered by _repr_html_
    api.check(train, api.load("test.csv"))
    api.transform(train, "drop_columns", columns=["Name"])

Same code as the keys (``keys/*`` build their Result from the same functions);
nothing here is reachable from the JSON contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, TypeAdapter, ValidationError

from dtk_engine.contract import list_transforms
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.keys.dataset_overview import overview_result
from dtk_engine.keys.duplicates import duplicates_result
from dtk_engine.keys.inconsistencies import inconsistencies_result
from dtk_engine.keys.train_test_check import check_result
from dtk_engine.result import Result
from dtk_engine.sources import SourceSpec
from dtk_engine.sources import load as load_spec
from dtk_engine.transform_registry import get_transform

__all__ = [
    "check",
    "duplicates",
    "inconsistencies",
    "list_transforms",
    "load",
    "overview",
    "transform",
]

# File suffix -> source kind for `load(path)`; other options need a full spec.
SUFFIX_KINDS = {".csv": "csv", ".tsv": "csv"}

_SPEC = TypeAdapter(SourceSpec)


def load(source: str | Path | dict | BaseModel) -> pd.DataFrame:
    """A DataFrame from a file path (kind guessed from the suffix) or a SourceSpec
    (dict or model), e.g. ``load({"kind": "csv", "path": "x.csv", "sep": ";"})``.
    """
    if isinstance(source, str | Path):
        suffix = Path(source).suffix.lower()
        if suffix not in SUFFIX_KINDS:
            raise SourceError(
                f"no reader for {suffix or 'a suffix-less path'!r} "
                f"(known: {sorted(SUFFIX_KINDS)}); pass a source spec dict instead"
            )
        source = {"kind": SUFFIX_KINDS[suffix], "path": str(source)}
    if isinstance(source, dict):
        try:
            source = _SPEC.validate_python(source)
        except ValidationError as exc:
            raise KeyParamsError(str(exc)) from exc
    return load_spec(source)


def overview(df: pd.DataFrame, head_rows: int = 5) -> Result:
    """``dataset_overview`` on a DataFrame."""
    return overview_result(df, head_rows)


def check(
    train: pd.DataFrame, test: pd.DataFrame, id_columns: list[str] | None = None
) -> Result:
    """``train_test_check`` on two DataFrames."""
    return check_result(train, test, id_columns)


def transform(df: pd.DataFrame, op: str, **params) -> pd.DataFrame:
    """Fit ``op`` on ``df`` and apply it (to fit on train and apply to test,
    use ``DtkTransformer``)."""
    t = get_transform(op)
    return t.fit_apply(df, t.parse(params))


def duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> Result:
    """``duplicates`` on a DataFrame (``subset`` null = detected id columns)."""
    return duplicates_result(df, subset)


def inconsistencies(df: pd.DataFrame, columns: list[str] | None = None) -> Result:
    """``inconsistencies`` on a DataFrame."""
    return inconsistencies_result(df, columns)
