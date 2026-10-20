"""Notebook door: DataFrame in, ``Result`` (displayed by Jupyter) or DataFrame out.

    from dtk_engine import api

    train = api.load("train.csv")
    api.overview(train)                      # Result, rendered by _repr_html_
    api.check(train, api.load("test.csv"))
    api.label_join_preview(X, y)
    api.transform(train, "drop_columns", columns=["Name"])
    api.advise(train, test, model_family="linear", target="Survived")
    api.select_features(train, target="Survived")
    api.distribution(train, test, columns=["Age", "Sex"])
    api.target_analysis(train, target="Survived")
    api.correlations(train, method="spearman")
    api.chart(train, chart="scatter", x="Age", y="Fare", trendline=True)
    api.export_workspace("titanic", "out/")  # parquet + manifest.json
    api.workspace_frames(sources)            # (train, test) before any step; what an exported script loads
    api.keep_label_last(train, "Survived")   # label moved to the end (exported scripts, label_last)

Same code as the keys (``keys/*`` build their Result from the same functions);
nothing here is reachable from the JSON contract.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, TypeAdapter, ValidationError

from dtk_engine.errors import (
    KeyParamsError,
    SourceError,
    key_params_from_validation,
    valid_names_hint,
)
from dtk_engine.keys.chart import chart_result
from dtk_engine.keys.column_distribution import distribution_result
from dtk_engine.keys.correlations import correlations_result
from dtk_engine.keys.dataset_overview import overview_result
from dtk_engine.keys.duplicates import duplicates_result
from dtk_engine.keys.feature_selection import selection_result
from dtk_engine.keys.inconsistencies import inconsistencies_result
from dtk_engine.keys.label_join_preview import (
    preview_result as label_join_preview_result,
)
from dtk_engine.keys.missing_values import missing_result
from dtk_engine.keys.outliers import outliers_result
from dtk_engine.keys.preprocessing_advisor import advisor_result
from dtk_engine.keys.target_analysis import target_result
from dtk_engine.keys.train_test_check import check_result
from dtk_engine.ops import transforms  # noqa: F401  (registers every transform op)
from dtk_engine.result import Result
from dtk_engine.sources import SourceSpec, string_columns
from dtk_engine.sources import load as load_spec
from dtk_engine.transform_registry import get_transform
from dtk_engine.transform_registry import transform_catalog as list_transforms
from dtk_engine.workspace.dataset import preview as preview_workspace
from dtk_engine.workspace.dataset import raw_workspace_frame, workspace_validation_error
from dtk_engine.workspace.export import export_workspace as _export_workspace
from dtk_engine.workspace.models import Workspace
from dtk_engine.workspace.replay import _label_last

__all__ = [
    "advise",
    "chart",
    "check",
    "correlations",
    "distribution",
    "duplicates",
    "export_workspace",
    "inconsistencies",
    "keep_label_last",
    "label_join_preview",
    "list_transforms",
    "load",
    "missing",
    "outliers",
    "overview",
    "preview_workspace",
    "select_features",
    "target_analysis",
    "transform",
    "workspace_frames",
]

# File suffix -> source kind for `load(path)`; other options need a full spec.
SUFFIX_KINDS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",  # routed so the reader can say "only .xlsx is supported"
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}

JSON_LINES_SUFFIXES = {".jsonl", ".ndjson"}

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
        if suffix in JSON_LINES_SUFFIXES:
            source["lines"] = True
    if isinstance(source, dict):
        try:
            source = _SPEC.validate_python(source)
        except ValidationError as exc:
            raise key_params_from_validation(exc) from exc
    return load_spec(source)


def _string_labels(fn):
    """Notebook frames may carry int column labels; the keys expect str names."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = [string_columns(a) if isinstance(a, pd.DataFrame) else a for a in args]
        kwargs = {
            k: string_columns(v) if isinstance(v, pd.DataFrame) else v
            for k, v in kwargs.items()
        }
        return fn(*args, **kwargs)

    return wrapper


@_string_labels
def overview(df: pd.DataFrame, head_rows: int = 5) -> Result:
    """``dataset_overview`` on a DataFrame."""
    return overview_result(df, head_rows)


@_string_labels
def check(
    train: pd.DataFrame, test: pd.DataFrame, id_columns: list[str] | None = None
) -> Result:
    """``train_test_check`` on two DataFrames."""
    return check_result(train, test, id_columns)


@_string_labels
def label_join_preview(
    x: pd.DataFrame, y: pd.DataFrame, key_columns: list[str] | None = None
) -> Result:
    """``label_join_preview`` on X and y DataFrames (before joining the labels)."""
    return label_join_preview_result(x, y, key_columns)


@_string_labels
def transform(df: pd.DataFrame, op: str, **params) -> pd.DataFrame:
    """Fit ``op`` on ``df`` and apply it (to fit on train and apply to test,
    use ``DtkTransformer``)."""
    t = get_transform(op)
    parsed = t.parse(params)
    try:
        return t.fit_apply(df, parsed)
    except KeyError as exc:
        _raise_missing_columns(df, op, parsed, exc)
        raise


def _raise_missing_columns(
    df: pd.DataFrame, op: str, params: BaseModel, exc: KeyError
) -> None:
    """KeyParamsError (with the valid columns) when the op's KeyError comes from
    a ``columns`` / ``column`` param naming columns absent from ``df``."""
    picked = getattr(params, "columns", None)
    names = list(picked) if isinstance(picked, list) else []
    single = getattr(params, "column", None)
    if isinstance(single, str):
        names.append(single)
    absent = [c for c in names if c not in df.columns]
    if absent:
        raise KeyParamsError(
            f"{op}: columns not in the frame {absent}"
            + valid_names_hint("columns", [str(c) for c in df.columns], absent[0])
        ) from exc


@_string_labels
def duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> Result:
    """``duplicates`` on a DataFrame (``subset`` null = detected id columns)."""
    return duplicates_result(df, subset)


@_string_labels
def inconsistencies(df: pd.DataFrame, columns: list[str] | None = None) -> Result:
    """``inconsistencies`` on a DataFrame."""
    return inconsistencies_result(df, columns)


@_string_labels
def missing(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    target: str | None = None,
    columns: list[str] | None = None,
    sort: str = "pct_missing",
    threshold: float = 0.0,
) -> Result:
    """``missing_values`` on a DataFrame (and an optional test frame)."""
    return missing_result(df, test, target, columns, sort, threshold)


@_string_labels
def outliers(
    df: pd.DataFrame,
    iqr_k: float = 1.5,
    z_threshold: float = 3.0,
    contamination: float = 0.01,
    random_state: int = 0,
    columns: list[str] | None = None,
    method: str = "all",
    mad_threshold: float = 3.5,
) -> Result:
    """``outliers`` on a DataFrame."""
    return outliers_result(
        df,
        iqr_k,
        z_threshold,
        contamination,
        random_state,
        columns,
        method,
        mad_threshold,
    )


@_string_labels
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


@_string_labels
def select_features(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    columns: list[str] | None = None,
    wrapper: bool = False,
    random_state: int = 0,
) -> Result:
    """``feature_selection`` on a DataFrame.

    The ``suggested_steps`` table lists selection ops (``select_k_best``,
    ``select_from_model``, ``drop_correlated``, ``pca``...) as "both" steps.
    """
    return selection_result(df, target, task, columns, wrapper, random_state)


@_string_labels
def distribution(
    df: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: list[str] | None = None,
    target: str | None = None,
    by_label: bool = False,
    bins: int | str = "auto",
    top_k: int = 10,
    target_bins: int = 4,
    by: str | None = None,
    bin_edges: list[float] | None = None,
    range_min_pct: float = 0.0,
    range_max_pct: float = 100.0,
    log_x: bool = False,
    log_y: bool = False,
    norm: str = "share",
    cumulative: bool = False,
) -> Result:
    """``column_distribution`` on a DataFrame (``test`` given: train vs test
    overlay; ``by_label``: split by the classes of ``target``; ``by``: split by
    any column)."""
    return distribution_result(
        df,
        test,
        columns,
        target,
        by_label,
        bins,
        top_k,
        target_bins,
        by,
        bin_edges,
        range_min_pct,
        range_max_pct,
        log_x,
        log_y,
        norm,
        cumulative,
    )


@_string_labels
def target_analysis(
    df: pd.DataFrame,
    target: str,
    columns: list[str] | None = None,
    task: str = "auto",
    top_k: int = 10,
    bins: int = 10,
    random_state: int = 0,
    target_bins: int = 10,
) -> Result:
    """``target_analysis`` on a DataFrame: each feature vs ``target``."""
    return target_result(
        df, target, columns, task, top_k, bins, random_state, target_bins
    )


@_string_labels
def correlations(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    threshold: float = 0.9,
    target: str | None = None,
    top_n: int = 15,
) -> Result:
    """``correlations`` on a DataFrame: heatmap + pairs with |corr| >= threshold."""
    return correlations_result(df, columns, method, threshold, target, top_n)


@_string_labels
def chart(df: pd.DataFrame, chart: str = "histogram", **params) -> Result:
    """``chart`` on a DataFrame: a Plotly Express figure from typed params."""
    return chart_result(df, chart, **params)


def export_workspace(
    name: str,
    out_dir: str | Path,
    overwrite: bool = False,
    store=None,
    formats: list[str] | None = None,
) -> dict:
    """Write the workspace's processed data (``formats``: parquet by default,
    csv, ipynb, py) + ``manifest.json`` under ``out_dir``; returns the manifest
    (see ``dtk_engine.workspace.export``)."""
    return _export_workspace(
        name, out_dir, overwrite=overwrite, store=store, formats=formats
    )


def keep_label_last(frame: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
    """``frame`` with the ``label`` column moved to the end, as the replay does
    after each step of a ``label_last`` workspace (None and an absent label pass)."""
    return _label_last(frame, label)


def workspace_frames(sources: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """``(train, test)`` before any step from a workspace's ``{datasets, label,
    merges}`` (labels joined, merges applied); test is None without a test set.
    What an exported notebook / script starts from."""
    try:
        ws = Workspace.model_validate({"name": "notebook", **sources})
    except ValidationError as exc:
        raise workspace_validation_error(exc, "invalid workspace sources: ") from exc
    train = raw_workspace_frame(ws, "train", labeled=True)
    test = raw_workspace_frame(ws, "test", labeled=True) if ws.datasets.test else None
    return train, test
