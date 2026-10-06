"""Feature engineering ops (derived, datetime, cyclical, binned, aggregated columns)."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from sklearn.preprocessing import (
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer,
    StandardScaler,
)

from dtk_engine.ops._util import json_scalar as _py
from dtk_engine.ops.selection import feature_matrix
from dtk_engine.params import column_field, columns_field, when
from dtk_engine.transform_registry import TransformParams, transform

MAX_INTERACTION_COLUMNS = 10
MAX_POLYNOMIAL_OUTPUT = 100
MAX_SPLINE_OUTPUT = 100


def _replace_columns(
    df: pd.DataFrame, columns: list[str], new: dict[str, np.ndarray], op: str
) -> pd.DataFrame:
    """Drop ``columns`` and insert ``new`` at the position of the first dropped one."""
    clash = [n for n in new if n in df.columns and n not in columns]
    if clash:
        raise ValueError(f"{op}: output columns already exist {clash}")
    rest = df.drop(columns=columns)
    pos = sum(1 for c in df.columns[: df.columns.get_loc(columns[0])] if c in rest)
    out = rest.copy()
    for i, (name, values) in enumerate(new.items()):
        out.insert(pos + i, name, values)
    return out


def _poly_feature_names(raw: np.ndarray) -> list[str]:
    """sklearn uses a space for products (``a b``); we prefer ``a*b`` like interactions."""
    return [str(n).replace(" ", "*") for n in raw]


# --- derive ---------------------------------------------------------------


class DeriveParams(TransformParams):
    a: str = column_field(..., "Left column", source="step")
    b: str = column_field(..., "Right column", source="step")
    op: Literal["ratio", "difference", "product", "days_between"] = Field(
        description="a/b, a-b, a*b, or days from a to b (both parsed as dates)"
    )
    name: str | None = Field(
        default=None, description="Output column (default '<a>_<op>_<b>')"
    )
    min_denominator: float = Field(
        default=1.0,
        gt=0,
        description="ratio only: |b| is clipped up to this value (no inf)",
    )


@transform("derive", params_model=DeriveParams, title="Derive column")
def derive(df: pd.DataFrame, params: DeriveParams, state: dict) -> pd.DataFrame:
    """Add a column combining two columns (ratio, difference, product, days_between)."""
    a, b = df[params.a], df[params.b]
    if params.op == "ratio":
        # Clip |b| up to min_denominator, keeping its sign (0 -> +min).
        sign = np.where(b < 0, -1.0, 1.0)
        denom = sign * b.abs().clip(lower=params.min_denominator)
        out = a / denom
    elif params.op == "difference":
        out = a - b
    elif params.op == "product":
        out = a * b
    else:
        delta = pd.to_datetime(b, errors="coerce") - pd.to_datetime(a, errors="coerce")
        out = delta / pd.Timedelta(days=1)
    return df.assign(**{params.name or f"{params.a}_{params.op}_{params.b}": out})


# --- datetime_parts ---------------------------------------------------------

_DT_PARTS = ("hour", "dayofweek", "month", "year", "is_weekend")


class DatetimePartsParams(TransformParams):
    column: str = column_field(..., "Datetime (or date-like) column", source="step")
    parts: list[Literal["hour", "dayofweek", "month", "year", "is_weekend"]] = Field(
        default_factory=lambda: list(_DT_PARTS),
        min_length=1,
        description="Parts to extract (each becomes '<column>_<part>')",
    )


@transform("datetime_parts", params_model=DatetimePartsParams, title="Datetime parts")
def datetime_parts(
    df: pd.DataFrame, params: DatetimePartsParams, state: dict
) -> pd.DataFrame:
    """Extract hour / dayofweek / month / year / is_weekend from a datetime column."""
    dt = pd.to_datetime(df[params.column], errors="coerce")
    new = {}
    for part in params.parts:
        if part == "is_weekend":
            # NaT -> NA, not False
            new[f"{params.column}_{part}"] = (
                (dt.dt.dayofweek >= 5).astype("Int64").where(dt.notna())
            )
        else:
            new[f"{params.column}_{part}"] = getattr(dt.dt, part)
    return df.assign(**new)


# --- cyclical ---------------------------------------------------------------


class CyclicalParams(TransformParams):
    column: str = column_field(
        ..., "Numeric column holding the cyclic value", source="step", dtype="numeric"
    )
    period: float = Field(gt=0, description="Cycle length (24 hour, 7 dow, 12 month)")


@transform("cyclical", params_model=CyclicalParams, title="Cyclical encoding")
def cyclical(df: pd.DataFrame, params: CyclicalParams, state: dict) -> pd.DataFrame:
    """Encode a cyclic column as '<col>_sin' and '<col>_cos'."""
    angle = 2 * np.pi * df[params.column] / params.period
    return df.assign(
        **{
            f"{params.column}_sin": np.sin(angle),
            f"{params.column}_cos": np.cos(angle),
        }
    )


# --- bin --------------------------------------------------------------------


class BinParams(TransformParams):
    column: str = column_field(
        ..., "Numeric column to bin", source="step", dtype="numeric"
    )
    mode: Literal["cut", "qcut"] = Field(
        description="cut: explicit edges; qcut: quantile edges fitted on train"
    )
    edges: list[float] | None = Field(
        default=None, description="cut only: increasing bin edges"
    )
    q: int | None = Field(default=None, ge=2, description="qcut only: number of bins")
    labels: list[str] | None = Field(
        default=None, description="Bin labels (default: integer bin codes)"
    )
    name: str | None = Field(
        default=None, description="Output column (default '<column>_bin')"
    )

    @model_validator(mode="after")
    def _mode_args(self) -> BinParams:
        if self.mode == "cut":
            if not self.edges or len(self.edges) < 2:
                raise ValueError("mode 'cut' needs `edges` (at least 2)")
            if self.q is not None:
                raise ValueError("`q` is for mode 'qcut'")
            if any(x >= y for x, y in zip(self.edges, self.edges[1:], strict=False)):
                raise ValueError("`edges` must be strictly increasing")
            if self.labels is not None and len(self.labels) != len(self.edges) - 1:
                raise ValueError("`labels` must have len(edges) - 1 entries")
        else:
            if self.q is None:
                raise ValueError("mode 'qcut' needs `q`")
            if self.edges is not None:
                raise ValueError("`edges` is for mode 'cut'")
            if self.labels is not None and len(self.labels) != self.q:
                raise ValueError("`labels` must have `q` entries")
        return self


def _bin_fit(df: pd.DataFrame, params: BinParams) -> dict:
    if params.mode == "cut":
        return {}
    quantiles = df[params.column].quantile([i / params.q for i in range(1, params.q)])
    return {"edges": sorted({float(v) for v in quantiles.dropna()})}


@transform(
    "bin",
    params_model=BinParams,
    fit=_bin_fit,
    title="Bin column",
    description="Bin a numeric column by explicit edges or train quantiles.",
)
def bin_column(df: pd.DataFrame, params: BinParams, state: dict) -> pd.DataFrame:
    """Bin a numeric column by explicit edges (cut) or train quantiles (qcut)."""
    x = df[params.column]
    # qcut: open-ended outer bins so values outside the train range still land somewhere.
    edges = params.edges if params.mode == "cut" else [-np.inf, *state["edges"], np.inf]
    labels = params.labels
    if labels is not None and len(labels) != len(edges) - 1:
        raise ValueError(
            f"qcut on {params.column!r} produced {len(edges) - 1} bins "
            f"(duplicate quantile edges) but {len(labels)} labels were given"
        )
    binned = pd.cut(x, bins=edges, labels=labels if labels is not None else False)
    if labels is None:
        binned = binned.astype("Int64")
    return df.assign(**{params.name or f"{params.column}_bin": binned})


# --- group_agg --------------------------------------------------------------

_ORDERED_AGGS = ("first", "last")


class GroupAggParams(TransformParams):
    group: str = column_field(..., "Group-by column", source="step")
    value: str = column_field(
        ..., "Numeric column to aggregate", source="step", dtype="numeric"
    )
    aggs: list[
        Literal["mean", "std", "count", "median", "min", "max", "first", "last"]
    ] = Field(
        min_length=1,
        description="Aggregations (column '<value>_<agg>_by_<group>'); first / last "
        "are the first / last observed value in `order`",
    )
    order: str | None = column_field(
        None,
        "first / last: order of the group's rows (e.g. age, a date); rows with a "
        "missing order are ignored, ties keep the frame order",
        source="step",
        extra=when(aggs=list(_ORDERED_AGGS)),
    )
    target: str | None = column_field(
        None,
        "Declared target column; aggregating it is refused (target leak)",
        source="step",
    )

    @model_validator(mode="after")
    def _check_order(self) -> GroupAggParams:
        if self.order is None and any(a in _ORDERED_AGGS for a in self.aggs):
            raise ValueError("group_agg: aggs first / last need an `order` column")
        return self

    @model_validator(mode="after")
    def _no_target_leak(self) -> GroupAggParams:
        if self.target is not None and self.value == self.target:
            raise ValueError(
                f"refusing to aggregate the target {self.target!r} by group: the "
                "group statistic would leak each row's own label into its feature. "
                "Use out-of-fold target encoding instead (planned)."
            )
        return self


def _group_agg_stats(df: pd.DataFrame, params: GroupAggParams) -> pd.DataFrame:
    """Per-group statistics, one column per agg (NaN where a group has no value)."""
    plain = [a for a in params.aggs if a not in _ORDERED_AGGS]
    ordered = [a for a in params.aggs if a in _ORDERED_AGGS]
    parts = []
    if plain:
        parts.append(df.groupby(params.group)[params.value].agg(plain))
    if ordered:  # first / last skip NaN values; a row without order is unusable
        rows = df[[params.group, params.value, params.order]]
        rows = rows.dropna(subset=[params.order]).sort_values(
            params.order, kind="stable"
        )
        parts.append(rows.groupby(params.group)[params.value].agg(ordered))
    stats = pd.concat(parts, axis=1)
    return stats.reindex(df[params.group].dropna().unique())[params.aggs]


def _group_agg_fit(df: pd.DataFrame, params: GroupAggParams) -> dict:
    stats = _group_agg_stats(df, params)
    return {
        "groups": [
            {"key": _py(key), **{a: _py(row[a]) for a in params.aggs}}
            for key, row in stats.iterrows()
        ]
    }


@transform(
    "group_agg",
    params_model=GroupAggParams,
    fit=_group_agg_fit,
    title="Group aggregate",
    description="Join per-group statistics (fitted on train) onto each row.",
)
def group_agg(df: pd.DataFrame, params: GroupAggParams, state: dict) -> pd.DataFrame:
    """Add per-group statistics of a column; unseen groups get NaN."""
    new = {}
    for agg in params.aggs:
        lookup = {g["key"]: g[agg] for g in state["groups"]}
        new[f"{params.value}_{agg}_by_{params.group}"] = (
            df[params.group].map(lookup).astype(float)
        )
    return df.assign(**new)


# --- interactions -----------------------------------------------------------


class InteractionsParams(TransformParams):
    columns: list[str] = columns_field(
        f"Numeric columns to combine (at most {MAX_INTERACTION_COLUMNS})",
        source="step",
        dtype="numeric",
        required=True,
        min_length=2,
        max_length=MAX_INTERACTION_COLUMNS,
    )
    interaction_only: bool = Field(
        default=True, description="Only products of distinct columns (no squares)"
    )


@transform("interactions", params_model=InteractionsParams, title="Interactions")
def interactions(
    df: pd.DataFrame, params: InteractionsParams, state: dict
) -> pd.DataFrame:
    """Add pairwise products '<a>*<b>' of the listed columns."""
    pairs = (
        combinations if params.interaction_only else combinations_with_replacement
    )(params.columns, 2)
    return df.assign(**{f"{a}*{b}": df[a] * df[b] for a, b in pairs})


# --- polynomial -------------------------------------------------------------


class PolynomialParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns to expand",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
        max_length=MAX_INTERACTION_COLUMNS,
    )
    degree: int = Field(default=2, ge=1, le=5, description="Polynomial degree")
    interaction_only: bool = Field(
        default=False,
        description="Only cross-products of distinct columns (no powers)",
    )
    include_bias: bool = Field(
        default=False, description="Add a constant column of ones"
    )
    max_output_columns: int = Field(
        default=MAX_POLYNOMIAL_OUTPUT,
        ge=1,
        description="Refuse if the expansion would produce more columns than this",
    )


def _fit_polynomial(df: pd.DataFrame, params: PolynomialParams) -> dict:
    feature_matrix(df, params.columns, "polynomial")  # missing -> clear error
    pf = PolynomialFeatures(
        degree=params.degree,
        interaction_only=params.interaction_only,
        include_bias=params.include_bias,
    )
    pf.fit(np.zeros((1, len(params.columns))))
    names = _poly_feature_names(pf.get_feature_names_out(params.columns))
    if len(names) > params.max_output_columns:
        raise ValueError(
            f"polynomial: expansion of {len(params.columns)} columns at degree "
            f"{params.degree} would produce {len(names)} columns "
            f"(cap is {params.max_output_columns}); lower degree, use "
            "interaction_only, or raise max_output_columns"
        )
    return {"columns": list(params.columns), "names": names}


@transform(
    "polynomial",
    params_model=PolynomialParams,
    fit=_fit_polynomial,
    title="Polynomial features",
    description=(
        "Replace numeric columns by their polynomial expansion "
        "(readable names like a^2, a*b; output capped)."
    ),
)
def polynomial(
    df: pd.DataFrame, params: PolynomialParams, state: dict
) -> pd.DataFrame:
    """Replace columns by PolynomialFeatures terms with readable names."""
    columns = state["columns"]
    X = feature_matrix(df, columns, "polynomial")
    pf = PolynomialFeatures(
        degree=params.degree,
        interaction_only=params.interaction_only,
        include_bias=params.include_bias,
    )
    pf.fit(X)
    names = state["names"]
    Y = pf.transform(X)
    return _replace_columns(df, columns, dict(zip(names, Y.T, strict=True)), "polynomial")


# --- power_transform --------------------------------------------------------


class PowerTransformParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns to transform in place",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )
    method: Literal["yeo-johnson", "box-cox"] = Field(
        default="yeo-johnson",
        description="yeo-johnson: any real values; box-cox: strictly positive only",
    )
    standardize: bool = Field(
        default=True, description="Zero-mean / unit-variance after the power map"
    )


def _fit_power_transform(df: pd.DataFrame, params: PowerTransformParams) -> dict:
    X = feature_matrix(df, params.columns, "power_transform")
    pt = PowerTransformer(method=params.method, standardize=params.standardize)
    pt.fit(X)
    state: dict = {
        "columns": list(params.columns),
        "lambdas": [float(v) for v in pt.lambdas_],
    }
    scaler = getattr(pt, "_scaler", None)
    if params.standardize and scaler is not None:
        state["mean"] = [float(v) for v in scaler.mean_]
        state["scale"] = [float(v) for v in scaler.scale_]
    return state


def _power_transformer(params: PowerTransformParams, state: dict) -> PowerTransformer:
    pt = PowerTransformer(method=params.method, standardize=params.standardize)
    # Dummy fit so sklearn sets n_features_in_ / optional _scaler skeleton.
    n = len(state["columns"])
    if params.method == "box-cox":
        dummy = np.arange(1, 1 + 2 * n, dtype=float).reshape(2, n)
    else:
        dummy = np.arange(2 * n, dtype=float).reshape(2, n)
    pt.fit(dummy)
    pt.lambdas_ = np.asarray(state["lambdas"], dtype=float)
    if params.standardize and "mean" in state:
        scaler = StandardScaler(copy=False)
        scaler.fit(dummy)
        scaler.mean_ = np.asarray(state["mean"], dtype=float)
        scaler.scale_ = np.asarray(state["scale"], dtype=float)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = n
        pt._scaler = scaler
    return pt


@transform(
    "power_transform",
    params_model=PowerTransformParams,
    fit=_fit_power_transform,
    title="Power transform",
    description="Power-map numeric columns (Yeo-Johnson / Box-Cox), fitted on train.",
)
def power_transform(
    df: pd.DataFrame, params: PowerTransformParams, state: dict
) -> pd.DataFrame:
    """Transform columns with a PowerTransformer fitted on train."""
    columns = state["columns"]
    X = feature_matrix(df, columns, "power_transform")
    Y = _power_transformer(params, state).transform(X)
    out = df.copy()
    out[columns] = Y
    return out


# --- quantile_transform -----------------------------------------------------


class QuantileTransformParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns to transform in place",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )
    output_distribution: Literal["uniform", "normal"] = Field(
        default="uniform", description="Map train quantiles to this distribution"
    )
    n_quantiles: int = Field(
        default=1000, ge=2, description="Number of quantiles learned on train"
    )
    random_state: int = Field(default=0, description="Seed for subsampled quantiles")


def _fit_quantile_transform(df: pd.DataFrame, params: QuantileTransformParams) -> dict:
    X = feature_matrix(df, params.columns, "quantile_transform")
    qt = QuantileTransformer(
        n_quantiles=params.n_quantiles,
        output_distribution=params.output_distribution,
        random_state=params.random_state,
        subsample=int(1e9),  # never subsample: state must match full train
    )
    qt.fit(X)
    return {
        "columns": list(params.columns),
        "n_quantiles": int(qt.n_quantiles_),
        "quantiles": [[float(v) for v in row] for row in qt.quantiles_],
        "references": [float(v) for v in qt.references_],
    }


def _quantile_transformer(
    params: QuantileTransformParams, state: dict
) -> QuantileTransformer:
    qt = QuantileTransformer(
        n_quantiles=state["n_quantiles"],
        output_distribution=params.output_distribution,
        random_state=params.random_state,
        subsample=int(1e9),
    )
    n = len(state["columns"])
    qt.fit(np.zeros((state["n_quantiles"], n)))
    qt.quantiles_ = np.asarray(state["quantiles"], dtype=float)
    qt.references_ = np.asarray(state["references"], dtype=float)
    qt.n_quantiles_ = state["n_quantiles"]
    return qt


@transform(
    "quantile_transform",
    params_model=QuantileTransformParams,
    fit=_fit_quantile_transform,
    title="Quantile transform",
    description="Map columns to a uniform/normal distribution via train quantiles.",
)
def quantile_transform(
    df: pd.DataFrame, params: QuantileTransformParams, state: dict
) -> pd.DataFrame:
    """Transform columns with a QuantileTransformer fitted on train."""
    columns = state["columns"]
    X = feature_matrix(df, columns, "quantile_transform")
    Y = _quantile_transformer(params, state).transform(X)
    out = df.copy()
    out[columns] = Y
    return out


# --- spline -----------------------------------------------------------------


class SplineParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns to expand into spline bases",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
        max_length=MAX_INTERACTION_COLUMNS,
    )
    n_knots: int = Field(default=5, ge=2, description="Number of knots per column")
    degree: int = Field(default=3, ge=0, le=5, description="B-spline degree")
    knots: Literal["uniform", "quantile"] = Field(
        default="quantile",
        description="Place knots evenly or on train quantiles",
    )
    include_bias: bool = Field(
        default=True, description="Keep the constant basis function per column"
    )
    max_output_columns: int = Field(
        default=MAX_SPLINE_OUTPUT,
        ge=1,
        description="Refuse if the expansion would produce more columns than this",
    )


def _fit_spline(df: pd.DataFrame, params: SplineParams) -> dict:
    X = feature_matrix(df, params.columns, "spline")
    st = SplineTransformer(
        n_knots=params.n_knots,
        degree=params.degree,
        knots=params.knots,
        include_bias=params.include_bias,
    )
    st.fit(X)
    names = [str(n) for n in st.get_feature_names_out(params.columns)]
    if len(names) > params.max_output_columns:
        raise ValueError(
            f"spline: expansion of {len(params.columns)} columns "
            f"(n_knots={params.n_knots}, degree={params.degree}) would produce "
            f"{len(names)} columns (cap is {params.max_output_columns}); "
            "lower n_knots/degree or raise max_output_columns"
        )
    # Base knot positions (without degree padding) — enough to rebuild at apply.
    base_knots = [
        [float(v) for v in b.t[params.degree : params.degree + params.n_knots]]
        for b in st.bsplines_
    ]
    return {
        "columns": list(params.columns),
        "names": names,
        "knots": list(map(list, zip(*base_knots, strict=True))),  # (n_knots, n_features)
    }


@transform(
    "spline",
    params_model=SplineParams,
    fit=_fit_spline,
    title="Spline features",
    description="Replace numeric columns by B-spline bases fitted on train knots.",
)
def spline(df: pd.DataFrame, params: SplineParams, state: dict) -> pd.DataFrame:
    """Replace columns by SplineTransformer bases using train knot positions."""
    columns = state["columns"]
    X = feature_matrix(df, columns, "spline")
    st = SplineTransformer(
        degree=params.degree,
        knots=np.asarray(state["knots"], dtype=float),
        include_bias=params.include_bias,
    )
    st.fit(X)
    Y = st.transform(X)
    names = state["names"]
    return _replace_columns(df, columns, dict(zip(names, Y.T, strict=True)), "spline")
