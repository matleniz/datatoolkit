"""Notebook door: DataFrame in, ``Result`` (displayed by Jupyter) or DataFrame out.

    from dtk_engine import api

    train = api.load("train.csv")
    api.overview(train)                      # Result, rendered by _repr_html_
    api.check(train, api.load("test.csv"))
    api.transform(train, "drop_columns", columns=["Name"])
    api.advise(train, test, model_family="linear", target="Survived")
    api.export_workspace("titanic", "out/")  # parquet + manifest.json

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
from dtk_engine.keys.missing_values import missing_result
from dtk_engine.keys.outliers import outliers_result
from dtk_engine.keys.preprocessing_advisor import advisor_result
from dtk_engine.keys.train_test_check import check_result
from dtk_engine.result import Result
from dtk_engine.sources import SourceSpec
from dtk_engine.sources import load as load_spec
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace.export import export_workspace as _export_workspace

__all__ = [
    "advise",
    "check",
    "duplicates",
    "export_workspace",
    "inconsistencies",
    "list_transforms",
    "load",
    "missing",
    "outliers",
    "overview",
    "transform",
]

# File suffix -> source kind for `load(path)`; other options need a full spec.
SUFFIX_KINDS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".json": "json",
    ".jsonl": "json",
}

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
        if suffix == ".jsonl":
            source["lines"] = True
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


def missing(
    df: pd.DataFrame, test: pd.DataFrame | None = None, target: str | None = None
) -> Result:
    """``missing_values`` on a DataFrame (and an optional test frame)."""
    return missing_result(df, test, target)


def outliers(
    df: pd.DataFrame,
    iqr_k: float = 1.5,
    z_threshold: float = 3.0,
    contamination: float = 0.01,
    random_state: int = 0,
) -> Result:
    """``outliers`` on a DataFrame."""
    return outliers_result(df, iqr_k, z_threshold, contamination, random_state)


def advise(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    model_family: str | None = None,
    target: str | None = None,
) -> Result:
    """``preprocessing_advisor`` on a DataFrame (and an optional test frame).

    Each row of the ``recommendations`` table is a step (``op``, ``target``,
    ``params``); ``dtk_engine.ops.advisor.as_steps`` lists them for a workspace.
    """
    return advisor_result(df, test, model_family, target)


def export_workspace(
    name: str, out_dir: str | Path, overwrite: bool = False, store=None
) -> dict:
    """Write the workspace's processed parquet + ``manifest.json`` under ``out_dir``;
    returns the manifest (see ``dtk_engine.workspace.export``)."""
    return _export_workspace(name, out_dir, overwrite=overwrite, store=store)
