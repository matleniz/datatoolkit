"""Imputation ops (fill missing values, fitted on train).

``impute`` stores the learned fill values (medians, means, modes) in its state;
strategy ``formula`` is stateless instead (the expression is the state, as in
the ``formula`` op): it is evaluated on each frame and fills only that frame's
missing cells. The group strategies (``group_mean`` / ``group_prev`` /
``group_interp``, ``by`` an entity column) are stateless the same way: each
frame is filled from the same entity's rows in that frame (``ops/groups.py``);
only their optional ``fallback`` statistic is learned on train.
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
from dtk_engine.ops.groups import group_codes, group_interp, group_mean, group_prev
from dtk_engine.ops.missing import CATEGORICAL_FILL
from dtk_engine.ops.transforms.formula import check_expr, evaluate
from dtk_engine.params import column_field, columns_field, when
from dtk_engine.transform_registry import TransformParams, transform

INDICATOR_SUFFIX = "_was_missing"
GROUP_STRATEGIES = ("group_mean", "group_prev", "group_interp")
ORDERED_STRATEGIES = ("group_prev", "group_interp")
NUMERIC_STRATEGIES = ("median", "mean", "formula", "group_mean", "group_interp")


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
    strategy: Literal[
        "median",
        "mean",
        "most_frequent",
        "constant",
        "formula",
        "group_mean",
        "group_prev",
        "group_interp",
    ] = Field(
        default="median",
        description="Fill value learned on train: median / mean (numeric), "
        "most_frequent (any), constant (fill_value); formula: the value of "
        "expr on the same row (one numeric column, nothing learned); "
        "group_mean / group_prev (last earlier value, any type) / group_interp "
        "(linear between neighbours): from the same `by` entity's rows of the "
        "frame being filled",
    )
    fill_value: str | float | int | None = Field(
        default=None,
        description='strategy "constant": the value; default "MISSING" for '
        "non-numeric columns, 0 for numeric ones",
        json_schema_extra=when(strategy="constant"),
    )
    expr: str | None = Field(
        default=None,
        description='strategy "formula": expression over other columns (formula '
        "syntax, no @variables); a row where it is undefined stays missing",
        json_schema_extra=when(strategy="formula"),
    )
    by: str | None = column_field(
        None,
        "Group strategies: entity column (e.g. patient_id)",
        source="step",
        semantic="group_id",
        extra=when(strategy=list(GROUP_STRATEGIES)),
    )
    order: str | None = column_field(
        None,
        "group_prev / group_interp: order of the entity's rows (e.g. age, a date)",
        source="step",
        extra=when(strategy=list(ORDERED_STRATEGIES)),
    )
    fallback: Literal["median", "mean", "most_frequent"] | None = Field(
        default=None,
        description="Group strategies: statistic learned on train for the cells "
        "the entity cannot fill (no observed row / neighbour); none = left missing",
        json_schema_extra=when(strategy=list(GROUP_STRATEGIES)),
    )
    add_indicator: bool = Field(
        default=False,
        description="Add a binary <col>_was_missing column next to each column",
    )

    @model_validator(mode="after")
    def _check_strategy(self) -> ImputeParams:
        if self.strategy == "formula":
            self._check_formula()
        elif self.strategy in GROUP_STRATEGIES:
            self._check_group()
        return self

    def _check_formula(self) -> None:
        if not self.expr or not self.expr.strip():
            raise ValueError('impute: strategy "formula" needs an expr')
        if len(self.columns) != 1:
            raise ValueError('impute: strategy "formula" fills exactly one column')
        check_expr(self.expr)

    def _check_group(self) -> None:
        if self.by is None:
            raise ValueError(f"impute: strategy {self.strategy!r} needs by")
        if self.strategy in ORDERED_STRATEGIES and self.order is None:
            raise ValueError(f"impute: strategy {self.strategy!r} needs order")
        filled = sorted({self.by, self.order} & set(self.columns))
        if filled:
            raise ValueError(f"impute: by / order cannot be filled columns {filled}")


def _mode(s: pd.Series):
    return _py(s.mode(dropna=True).iloc[0])  # ties -> smallest, as SimpleImputer


_FITTED_FILLS = {
    "median": lambda s: float(s.median()),
    "mean": lambda s: float(s.mean()),
    "most_frequent": _mode,
}


def _learned(df: pd.DataFrame, col: str, stat: str):
    if stat in ("median", "mean"):
        _numeric(df, [col], f"impute strategy {stat!r}")
    if df[col].isna().all():
        raise ValueError(f"impute: column {col!r} is entirely missing on fit")
    return _FITTED_FILLS[stat](df[col])


def _fill_value(df: pd.DataFrame, col: str, params: ImputeParams):
    numeric = pd.api.types.is_numeric_dtype(df[col])
    if params.strategy != "constant":
        return _learned(df, col, params.strategy)
    value = params.fill_value
    if value is None:
        value = 0 if numeric else CATEGORICAL_FILL
    if numeric and isinstance(value, str):
        raise ValueError(f"impute: string fill {value!r} on numeric column {col!r}")
    return value


def _fit_impute(df: pd.DataFrame, params: ImputeParams) -> dict:
    if params.strategy in NUMERIC_STRATEGIES:
        _numeric(df, params.columns, f"impute strategy {params.strategy!r}")
    if params.strategy == "formula":
        return {"fill": {}}
    if params.strategy in GROUP_STRATEGIES:  # only the fallback is learned
        stat = params.fallback
        return {
            "fill": {c: _learned(df, c, stat) for c in params.columns} if stat else {}
        }
    return {"fill": {col: _fill_value(df, col, params) for col in params.columns}}


def _order_key(series: pd.Series) -> np.ndarray:
    """Numbers as is, dates as timestamps; anything else unparseable -> NaN."""
    if pd.api.types.is_datetime64_any_dtype(series):
        stamps = series.to_numpy(dtype="datetime64[ns]")
        key = stamps.astype("int64").astype(float)
        key[np.isnat(stamps)] = np.nan
        return key
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _group_fill(df: pd.DataFrame, col: str, params: ImputeParams) -> np.ndarray:
    by = group_codes(df[params.by])
    if params.strategy == "group_mean":
        return group_mean(df[col].to_numpy(dtype=float), by)
    order = _order_key(df[params.order])
    if params.strategy == "group_prev":
        return group_prev(df[col], by, order).to_numpy()
    return group_interp(df[col].to_numpy(dtype=float), by, order)


def _fills(df: pd.DataFrame, params: ImputeParams, state: dict) -> dict:
    """Column -> fill: a scalar, or a per-row array (formula, group strategies)."""
    if params.strategy in NUMERIC_STRATEGIES:
        _numeric(df, params.columns, f"impute strategy {params.strategy!r}")
    if params.strategy == "formula":
        return {params.columns[0]: evaluate(df, params.expr)}
    if params.strategy in GROUP_STRATEGIES:
        return {col: _group_fill(df, col, params) for col in params.columns}
    return state["fill"]


def _filled(series: pd.Series, fill) -> pd.Series:
    if not isinstance(fill, np.ndarray):
        return series.fillna(fill)
    missing = series.isna()  # positional: safe with a duplicated index
    return series.mask(missing, fill) if missing.any() else series


@transform(
    "impute",
    params_model=ImputeParams,
    fit=_fit_impute,
    title="Impute",
    description="Fill missing values with a statistic learned on train (median, "
    "mean, mode, constant), a formula of other columns, or from the same "
    "entity's rows (group mean / previous / interpolated value).",
)
def impute(df: pd.DataFrame, params: ImputeParams, state: dict) -> pd.DataFrame:
    """Fill missing values: train statistic, constant, formula or group fill."""
    out = df.copy()
    fills = _fills(df, params, state)
    fallback = state["fill"] if params.strategy in GROUP_STRATEGIES else {}
    for col in params.columns:
        missing = out[col].isna()
        out[col] = _filled(out[col], fills[col])
        if col in fallback:
            out[col] = out[col].fillna(fallback[col])
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
        "Columns to fill; default all but sort_by and by",
        source="step",
        nullable=True,
    )
    by: str | None = column_field(
        None,
        "Entity column (e.g. patient_id): carry values only within each entity",
        source="step",
        semantic="group_id",
    )
    limit: int | None = Field(
        default=None, ge=1, description="Max consecutive missing values filled"
    )

    @model_validator(mode="after")
    def _not_sort_column(self):
        if self.by is not None and self.by == self.sort_by:
            raise ValueError("ffill: by and sort_by must be different columns")
        if self.columns is not None and self.sort_by in self.columns:
            raise ValueError("ffill: sort_by cannot be one of the filled columns")
        if self.columns is not None and self.by in self.columns:
            raise ValueError("ffill: by cannot be one of the filled columns")
        return self


@transform("ffill", params_model=FfillParams, title="Forward fill")
def ffill(df: pd.DataFrame, params: FfillParams, state: dict) -> pd.DataFrame:
    """Carry the last seen value forward in sort_by order (never backward).

    Rows keep their original order; leading missing values stay missing (no
    backward fill: that would use the future). With ``by``, values are carried
    only within each entity (rows with a missing ``by`` form one group).
    Stateless: a test frame is filled from its own past rows only.
    """
    keys = {params.sort_by, params.by}
    columns = params.columns or [c for c in df.columns if c not in keys]
    if df[params.sort_by].isna().any():
        raise ValueError(f"ffill: sort column {params.sort_by!r} has missing values")
    # Positions, not labels: works with a duplicated index.
    pos = np.argsort(df[params.sort_by].to_numpy(), kind="stable")
    rows = df[columns].iloc[pos]
    if params.by is not None:
        groups = df[params.by].iloc[pos].to_numpy()
        filled = rows.groupby(groups, dropna=False, sort=False).ffill(
            limit=params.limit
        )
    else:
        filled = rows.ffill(limit=params.limit)
    out = df.copy()
    out[columns] = filled.iloc[np.argsort(pos)].set_axis(df.index)
    return out
