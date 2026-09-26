"""Imputation ops (fill missing values, fitted on train).

``impute`` stores the learned fill values (medians, means, modes) in its state.
``impute_knn`` / ``impute_iterative`` are model-based: their fitted sklearn
imputer *is* the training matrix (KNN) or a chain of regressors (iterative), which
is not JSON. Their state is therefore the training matrix of the imputed
columns (JSON lists, ``None`` for missing), and ``apply`` refits the sklearn
imputer on it deterministically (fixed ``random_state``) before transforming.
Same result as a fitted imputer, at the cost of a state of size rows x columns
and a refit per apply.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

from dtk_engine.ops._util import py as _py
from dtk_engine.ops._util import require_numeric as _numeric
from dtk_engine.params import column_field, columns_field
from dtk_engine.transform_registry import TransformParams, transform

INDICATOR_SUFFIX = "_was_missing"
CATEGORICAL_FILL = "MISSING"


def _matrix_state(df: pd.DataFrame, columns: list[str], op: str) -> dict:
    _numeric(df, columns, op)
    empty = [c for c in columns if df[c].isna().all()]
    if empty:
        raise ValueError(f"{op}: columns entirely missing in the fit frame {empty}")
    X = df[columns].astype(float).to_numpy()
    rows = [[None if np.isnan(v) else float(v) for v in row] for row in X]
    return {"columns": columns, "train": rows}


def _train_matrix(state: dict) -> np.ndarray:
    return np.array(state["train"], dtype=float)  # None -> nan


def _put_back(df: pd.DataFrame, columns: list[str], X: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out[columns] = pd.DataFrame(X, index=df.index, columns=columns)
    return out


# --- impute -------------------------------------------------------------------


class ImputeParams(TransformParams):
    columns: list[str] = columns_field(
        "Columns to fill", source="step", required=True, min_length=1
    )
    strategy: Literal["median", "mean", "most_frequent", "constant"] = Field(
        default="median",
        description="Fill value learned on train: median / mean (numeric), "
        "most_frequent (any), constant (fill_value)",
    )
    fill_value: str | float | int | None = Field(
        default=None,
        description='strategy "constant": the value; default "MISSING" for '
        "non-numeric columns, 0 for numeric ones",
    )
    add_indicator: bool = Field(
        default=False,
        description="Add a binary <col>_was_missing column next to each column",
    )


def _fit_impute(df: pd.DataFrame, params: ImputeParams) -> dict:
    fill = {}
    for col in params.columns:
        s = df[col]
        if params.strategy == "constant":
            if params.fill_value is not None:
                value = params.fill_value
            else:
                value = 0 if pd.api.types.is_numeric_dtype(s) else CATEGORICAL_FILL
        else:
            if params.strategy in ("median", "mean"):
                _numeric(df, [col], f"impute strategy {params.strategy!r}")
            if s.isna().all():
                raise ValueError(f"impute: column {col!r} is entirely missing on fit")
            if params.strategy == "median":
                value = float(s.median())
            elif params.strategy == "mean":
                value = float(s.mean())
            else:  # ties -> smallest value, as sklearn's SimpleImputer
                value = _py(s.mode(dropna=True).iloc[0])
        if pd.api.types.is_numeric_dtype(s) and isinstance(value, str):
            raise ValueError(f"impute: string fill {value!r} on numeric column {col!r}")
        fill[col] = value
    return {"fill": fill}


@transform("impute", params_model=ImputeParams, fit=_fit_impute, title="Impute")
def impute(df: pd.DataFrame, params: ImputeParams, state: dict) -> pd.DataFrame:
    """Fill missing values with a statistic learned on train (median, mean, mode, constant)."""
    out = df.copy()
    for col in params.columns:
        missing = out[col].isna()
        out[col] = out[col].fillna(state["fill"][col])
        if params.add_indicator:
            name = f"{col}{INDICATOR_SUFFIX}"
            if name in out.columns:
                raise ValueError(f"impute: indicator column {name!r} already exists")
            out.insert(out.columns.get_loc(col) + 1, name, missing.astype(int))
    return out


# --- impute_knn ---------------------------------------------------------------


class ImputeKnnParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns, used both as neighbor features and filled. "
        "Distances are raw: scale the features first",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )
    n_neighbors: int = Field(default=5, ge=1, description="Neighbors averaged")
    weights: Literal["uniform", "distance"] = Field(
        default="uniform", description="Neighbor weighting"
    )


def _fit_knn(df: pd.DataFrame, params: ImputeKnnParams) -> dict:
    return _matrix_state(df, params.columns, "impute_knn")


@transform(
    "impute_knn", params_model=ImputeKnnParams, fit=_fit_knn, title="Impute (KNN)"
)
def impute_knn(df: pd.DataFrame, params: ImputeKnnParams, state: dict) -> pd.DataFrame:
    """Fill numeric columns from the k nearest train rows (scale features first).

    Distances use the raw values, so a large-range column dominates: put a
    ``scale`` step before this one.
    """
    _numeric(df, params.columns, "impute_knn")
    imputer = KNNImputer(n_neighbors=params.n_neighbors, weights=params.weights)
    imputer.fit(_train_matrix(state))
    X = imputer.transform(df[params.columns].astype(float).to_numpy())
    return _put_back(df, params.columns, X)


# --- impute_iterative ---------------------------------------------------------


class ImputeIterativeParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns, each regressed on the others",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )
    max_iter: int = Field(default=10, ge=1, description="Imputation rounds")
    random_state: int = Field(default=0, description="Seed (fixed: replay is exact)")


def _fit_iterative(df: pd.DataFrame, params: ImputeIterativeParams) -> dict:
    return _matrix_state(df, params.columns, "impute_iterative")


@transform(
    "impute_iterative",
    params_model=ImputeIterativeParams,
    fit=_fit_iterative,
    title="Impute (iterative)",
)
def impute_iterative(
    df: pd.DataFrame, params: ImputeIterativeParams, state: dict
) -> pd.DataFrame:
    """Fill numeric columns by regressing each one on the others (MICE-style)."""
    _numeric(df, params.columns, "impute_iterative")
    imputer = IterativeImputer(
        max_iter=params.max_iter, random_state=params.random_state
    )
    imputer.fit(_train_matrix(state))
    X = imputer.transform(df[params.columns].astype(float).to_numpy())
    return _put_back(df, params.columns, X)


# --- ffill --------------------------------------------------------------------


class FfillParams(TransformParams):
    sort_by: str = column_field(
        ..., "Column giving the order (e.g. a timestamp)", source="step"
    )
    columns: list[str] | None = columns_field(
        "Columns to fill; default all but sort_by", source="step", nullable=True
    )
    limit: int | None = Field(
        default=None, ge=1, description="Max consecutive missing values filled"
    )

    @model_validator(mode="after")
    def _not_sort_column(self):
        if self.columns is not None and self.sort_by in self.columns:
            raise ValueError("ffill: sort_by cannot be one of the filled columns")
        return self


@transform("ffill", params_model=FfillParams, title="Forward fill")
def ffill(df: pd.DataFrame, params: FfillParams, state: dict) -> pd.DataFrame:
    """Carry the last seen value forward in sort_by order (never backward).

    Rows keep their original order; leading missing values stay missing (no
    backward fill: that would use the future). Stateless: a test frame is
    filled from its own past rows only.
    """
    columns = params.columns or [c for c in df.columns if c != params.sort_by]
    if df[params.sort_by].isna().any():
        raise ValueError(f"ffill: sort column {params.sort_by!r} has missing values")
    # Positions, not labels: works with a duplicated index.
    pos = np.argsort(df[params.sort_by].to_numpy(), kind="stable")
    filled = df[columns].iloc[pos].ffill(limit=params.limit)
    out = df.copy()
    out[columns] = filled.iloc[np.argsort(pos)].set_axis(df.index)
    return out
