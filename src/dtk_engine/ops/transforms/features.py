"""Feature engineering ops (derived, datetime, cyclical, binned, aggregated columns)."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from dtk_engine.ops._util import json_scalar as _py
from dtk_engine.transform_registry import TransformParams, transform

MAX_INTERACTION_COLUMNS = 10


# --- derive ---------------------------------------------------------------


class DeriveParams(TransformParams):
    a: str = Field(description="Left column")
    b: str = Field(description="Right column")
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
    column: str = Field(description="Datetime (or date-like) column")
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
    column: str = Field(description="Numeric column holding the cyclic value")
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
    column: str = Field(description="Numeric column to bin")
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
    if params.mode == "cut":
        edges = params.edges
    else:
        # Open-ended outer bins so values outside the train range still land somewhere.
        edges = [-np.inf, *state["edges"], np.inf]
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

_AGGS = ("mean", "std", "count", "median")


class GroupAggParams(TransformParams):
    group: str = Field(description="Group-by column")
    value: str = Field(description="Numeric column to aggregate")
    aggs: list[Literal["mean", "std", "count", "median"]] = Field(
        min_length=1, description="Aggregations (column '<value>_<agg>_by_<group>')"
    )
    target: str | None = Field(
        default=None,
        description="Declared target column; aggregating it is refused (target leak)",
    )

    @model_validator(mode="after")
    def _no_target_leak(self) -> GroupAggParams:
        if self.target is not None and self.value == self.target:
            raise ValueError(
                f"refusing to aggregate the target {self.target!r} by group: the "
                "group statistic would leak each row's own label into its feature. "
                "Use out-of-fold target encoding instead (planned)."
            )
        return self


def _group_agg_fit(df: pd.DataFrame, params: GroupAggParams) -> dict:
    stats = df.groupby(params.group)[params.value].agg(params.aggs)
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
    columns: list[str] = Field(
        min_length=2,
        max_length=MAX_INTERACTION_COLUMNS,
        description=f"Numeric columns to combine (at most {MAX_INTERACTION_COLUMNS})",
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
