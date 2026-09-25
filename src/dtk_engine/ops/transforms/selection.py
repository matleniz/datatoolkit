"""Feature selection ops (fitted on train) and PCA.

A selector's state is the list of columns it keeps and the list it drops;
``apply`` drops exactly the fitted ``dropped`` list, so test gets train's
selection. Columns outside the candidates (non-numeric, the target) pass
through untouched: the target is never selected away.

Supervised ops (``select_k_best``, ``select_from_model``, ``drop_correlated``
with a target) read the target column named by ``target`` at fit: in a
workspace it is a column of the train frame; in sklearn ``DtkTransformer``
joins ``y`` under that name for fit only.

Candidates must be complete for the model-based ops: missing values raise
(add an ``impute`` step first).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

from dtk_engine.ops.selection import (
    candidate_columns,
    feature_matrix,
    filter_scores,
    l1_model,
    model_importance,
    require_target,
    resolve_task,
    standardize,
    target_vector,
    tree_model,
)
from dtk_engine.transform_registry import TransformParams, transform

Task = Literal["auto", "classification", "regression"]


def _columns_field():
    return Field(
        default=None,
        min_length=1,
        description="Candidate numeric columns; default every numeric column but "
        "the target",
    )


def _selection_state(columns: list[str], keep: list[bool], **extra) -> dict:
    return {
        "selected": [c for c, k in zip(columns, keep, strict=True) if k],
        "dropped": [c for c, k in zip(columns, keep, strict=True) if not k],
        **extra,
    }


def _drop_selected_out(df: pd.DataFrame, state: dict, op: str) -> pd.DataFrame:
    absent = [c for c in state["dropped"] if c not in df.columns]
    if absent:
        raise KeyError(f"{op}: fitted columns not in the frame {absent}")
    return df.drop(columns=state["dropped"])


def _target(df: pd.DataFrame, params, op: str) -> tuple[str, np.ndarray]:
    task = resolve_task(require_target(df, params.target, op), params.task)
    return task, target_vector(df, params.target, task, op)


def _floats(values) -> list[float]:
    return [float(v) for v in values]


# --- drop_low_variance --------------------------------------------------------


class DropLowVarianceParams(TransformParams):
    threshold: float = Field(
        default=0.0,
        ge=0,
        description="Drop columns whose train variance is <= this (0 = constants)",
    )
    columns: list[str] | None = _columns_field()
    target: str | None = Field(
        default=None, description="Target column: never a candidate"
    )


def _fit_low_variance(df: pd.DataFrame, params: DropLowVarianceParams) -> dict:
    columns = candidate_columns(df, params.target, params.columns, "drop_low_variance")
    variance = df[columns].astype(float).var(ddof=0).fillna(0.0)
    keep = [v > params.threshold for v in variance]
    return _selection_state(columns, keep, variance=_floats(variance))


@transform(
    "drop_low_variance",
    params_model=DropLowVarianceParams,
    fit=_fit_low_variance,
    title="Drop low variance",
)
def drop_low_variance(
    df: pd.DataFrame, params: DropLowVarianceParams, state: dict
) -> pd.DataFrame:
    """Drop numeric columns whose train variance is at or below a threshold."""
    return _drop_selected_out(df, state, "drop_low_variance")


# --- drop_correlated ----------------------------------------------------------


class DropCorrelatedParams(TransformParams):
    threshold: float = Field(
        default=0.95,
        gt=0,
        le=1,
        description="Drop one column of each pair with |corr| >= this",
    )
    columns: list[str] | None = _columns_field()
    target: str | None = Field(
        default=None,
        description="Target column: when given, the column more correlated with "
        "it is kept (else the first in column order)",
    )


def _fit_correlated(df: pd.DataFrame, params: DropCorrelatedParams) -> dict:
    op = "drop_correlated"
    columns = candidate_columns(df, params.target, params.columns, op)
    corr = df[columns].astype(float).corr().abs().fillna(0.0)
    order = list(columns)
    if params.target is not None:
        numeric = pd.api.types.is_numeric_dtype(require_target(df, params.target, op))
        # A text target is correlated through its (sorted) class codes.
        task = "regression" if numeric else "classification"
        y = target_vector(df, params.target, task, op)
        with np.errstate(divide="ignore", invalid="ignore"):
            to_target = df[columns].astype(float).corrwith(pd.Series(y, index=df.index))
        to_target = to_target.abs().fillna(0.0)
        # Most correlated with the target first; ties keep column order.
        order = sorted(columns, key=lambda c: -to_target[c])
    kept: list[str] = []
    for col in order:
        if all(corr.loc[col, k] < params.threshold for k in kept):
            kept.append(col)
    return _selection_state(columns, [c in kept for c in columns])


@transform(
    "drop_correlated",
    params_model=DropCorrelatedParams,
    fit=_fit_correlated,
    title="Drop correlated",
    needs_target=True,
)
def drop_correlated(
    df: pd.DataFrame, params: DropCorrelatedParams, state: dict
) -> pd.DataFrame:
    """Drop one column of each highly correlated pair (keeps the more target-correlated)."""
    return _drop_selected_out(df, state, "drop_correlated")


# --- select_k_best ------------------------------------------------------------


class SelectKBestParams(TransformParams):
    target: str = Field(description="Target column (read at fit only)")
    score: Literal["mutual_info", "f_test"] = Field(
        default="mutual_info",
        description="mutual_info (any dependence) | f_test (linear: ANOVA F / "
        "f_regression)",
    )
    k: int | None = Field(
        default=None, ge=1, description="Keep the k best (capped at the candidates)"
    )
    percentile: float | None = Field(
        default=None,
        gt=0,
        le=100,
        description="Or keep this % of the candidates (rounded up)",
    )
    task: Task = Field(default="auto", description="auto | classification | regression")
    columns: list[str] | None = _columns_field()
    random_state: int = Field(default=0, description="Mutual information seed")

    @model_validator(mode="after")
    def _k_or_percentile(self) -> SelectKBestParams:
        if (self.k is None) == (self.percentile is None):
            raise ValueError("select_k_best: give exactly one of k and percentile")
        return self


def _fit_k_best(df: pd.DataFrame, params: SelectKBestParams) -> dict:
    op = "select_k_best"
    columns = candidate_columns(df, params.target, params.columns, op)
    task, y = _target(df, params, op)
    X = feature_matrix(df, columns, op)
    scores = filter_scores(X, y, task, params.random_state)
    score = scores["mutual_info" if params.score == "mutual_info" else "f_score"]
    if params.k is not None:
        k = min(params.k, len(columns))
    else:
        k = max(1, math.ceil(len(columns) * params.percentile / 100))
    best = set(np.argsort(-score, kind="stable")[:k].tolist())
    keep = [i in best for i in range(len(columns))]
    return _selection_state(columns, keep, task=task, scores=_floats(score))


@transform(
    "select_k_best",
    params_model=SelectKBestParams,
    fit=_fit_k_best,
    title="Select k best",
    needs_target=True,
)
def select_k_best(
    df: pd.DataFrame, params: SelectKBestParams, state: dict
) -> pd.DataFrame:
    """Filter: keep the k best columns by mutual information or F-test with the target."""
    return _drop_selected_out(df, state, "select_k_best")


# --- select_from_model --------------------------------------------------------


class SelectFromModelParams(TransformParams):
    target: str = Field(description="Target column (read at fit only)")
    model: Literal["l1", "tree"] = Field(
        default="tree",
        description="l1: L1 logistic / LassoCV on standardized columns (zeroes "
        "coefficients); tree: random-forest importance",
    )
    threshold: Literal["mean", "median"] | float | None = Field(
        default=None,
        description='Keep importance >= this: "mean", "median" or a number '
        "(default: 1e-5 for l1, mean for tree)",
    )
    max_features: int | None = Field(
        default=None, ge=1, description="Keep at most this many (best first)"
    )
    task: Task = Field(default="auto", description="auto | classification | regression")
    columns: list[str] | None = _columns_field()
    random_state: int = Field(default=0, description="Model seed")


def _fit_from_model(df: pd.DataFrame, params: SelectFromModelParams) -> dict:
    op = "select_from_model"
    columns = candidate_columns(df, params.target, params.columns, op)
    task, y = _target(df, params, op)
    X = feature_matrix(df, columns, op)
    if params.model == "l1":
        n_classes = len(np.unique(y)) if task == "classification" else 0
        estimator, X = l1_model(task, params.random_state, n_classes), standardize(X)
    else:
        estimator = tree_model(task, params.random_state)
    estimator.fit(X, y)
    selector = SelectFromModel(
        estimator,
        threshold=params.threshold,
        max_features=params.max_features,
        prefit=True,
    )
    keep = selector.get_support().tolist()
    return _selection_state(
        columns, keep, task=task, importance=_floats(model_importance(estimator))
    )


@transform(
    "select_from_model",
    params_model=SelectFromModelParams,
    fit=_fit_from_model,
    title="Select from model",
    needs_target=True,
)
def select_from_model(
    df: pd.DataFrame, params: SelectFromModelParams, state: dict
) -> pd.DataFrame:
    """Embedded: keep the columns an L1 model or a random forest finds important."""
    return _drop_selected_out(df, state, "select_from_model")


# --- pca ----------------------------------------------------------------------


class PcaParams(TransformParams):
    n_components: int | float = Field(
        default=0.95,
        gt=0,
        description="Components kept: an int, or a variance fraction in (0, 1) "
        "(smallest number reaching it)",
    )
    columns: list[str] | None = _columns_field()
    standardize: bool = Field(
        default=True,
        description="Standardize the inputs with train mean / std first. PCA needs "
        "scaled inputs (variance has units): turn off only after a scale step",
    )
    whiten: bool = Field(
        default=False, description="Rescale components to unit variance"
    )
    target: str | None = Field(
        default=None, description="Target column: never a candidate"
    )
    prefix: str = Field(
        default="pc", min_length=1, description="Output names: pc1..pcN"
    )

    @model_validator(mode="after")
    def _fraction_or_count(self) -> PcaParams:
        n = self.n_components
        if isinstance(n, float) and not n.is_integer() and n >= 1:
            raise ValueError("pca: n_components is an int or a fraction in (0, 1)")
        return self


def _n_components(n: float) -> float:
    return n if n < 1 else int(n)


def _fit_pca(df: pd.DataFrame, params: PcaParams) -> dict:
    columns = candidate_columns(df, params.target, params.columns, "pca")
    X = feature_matrix(df, columns, "pca")
    n = _n_components(params.n_components)
    if isinstance(n, int) and n > min(X.shape):
        raise ValueError(f"pca: n_components={n} > min(rows, columns)={min(X.shape)}")
    center = X.mean(axis=0) if params.standardize else np.zeros(X.shape[1])
    std = X.std(axis=0) if params.standardize else np.ones(X.shape[1])
    scale = np.where(std == 0, 1.0, std)
    pca = PCA(n_components=n, whiten=params.whiten, svd_solver="full")
    pca.fit((X - center) / scale)
    return {
        "columns": columns,
        "center": _floats(center),
        "scale": _floats(scale),
        "mean": _floats(pca.mean_),
        "components": [_floats(row) for row in pca.components_],
        "explained_variance": _floats(pca.explained_variance_),
        "explained_variance_ratio": _floats(pca.explained_variance_ratio_),
    }


@transform("pca", params_model=PcaParams, fit=_fit_pca, title="PCA")
def pca(df: pd.DataFrame, params: PcaParams, state: dict) -> pd.DataFrame:
    """Replace numeric columns by their principal components pc1..pcN (fitted on train).

    Inputs are standardized with train statistics unless ``standardize`` is off
    (then scale them in an earlier step: variance has units).
    """
    columns = state["columns"]
    X = feature_matrix(df, columns, "pca")
    Z = (X - np.array(state["center"])) / np.array(state["scale"])
    components = np.array(state["components"])
    P = (Z - np.array(state["mean"])) @ components.T
    if params.whiten:
        P = P / np.sqrt(np.array(state["explained_variance"]))
    names = [f"{params.prefix}{i + 1}" for i in range(len(components))]
    rest = df.drop(columns=columns)
    clash = [n for n in names if n in rest.columns]
    if clash:
        raise ValueError(f"pca: output columns already exist {clash}")
    pos = sum(1 for c in df.columns[: df.columns.get_loc(columns[0])] if c in rest)
    out = rest.copy()
    for i, name in enumerate(names):
        out.insert(pos + i, name, P[:, i])
    return out
