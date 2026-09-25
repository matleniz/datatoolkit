"""Cleaning ops: drop, rename, cast, dedupe, text, dates, sentinels, filter, clip."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dtk_engine.ops.profile import hashable_frame
from dtk_engine.transform_registry import TransformParams, transform


class DropColumnsParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Columns to drop")
    missing_ok: bool = Field(
        default=False, description="Ignore listed columns absent from the frame"
    )


@transform("drop_columns", params_model=DropColumnsParams, title="Drop columns")
def drop_columns(
    df: pd.DataFrame, params: DropColumnsParams, state: dict
) -> pd.DataFrame:
    """Remove the listed columns."""
    return df.drop(
        columns=params.columns, errors="ignore" if params.missing_ok else "raise"
    )


class RenameParams(TransformParams):
    mapping: dict[str, str] = Field(
        min_length=1, description="Old column name -> new column name"
    )
    missing_ok: bool = Field(
        default=False, description="Ignore old names absent from the frame"
    )


@transform("rename", params_model=RenameParams, title="Rename columns")
def rename(df: pd.DataFrame, params: RenameParams, state: dict) -> pd.DataFrame:
    """Rename columns via an old -> new mapping."""
    return df.rename(
        columns=params.mapping, errors="ignore" if params.missing_ok else "raise"
    )


class CastParams(TransformParams):
    dtypes: dict[str, str] = Field(
        min_length=1, description="Column -> target dtype (e.g. 'int64', 'category')"
    )

    @field_validator("dtypes")
    @classmethod
    def _valid_dtypes(cls, value: dict[str, str]) -> dict[str, str]:
        for col, dtype in value.items():
            try:
                pd.api.types.pandas_dtype(dtype)
            except TypeError as exc:
                raise ValueError(f"column {col!r}: unknown dtype {dtype!r}") from exc
        return value


@transform("cast", params_model=CastParams, title="Cast column types")
def cast(df: pd.DataFrame, params: CastParams, state: dict) -> pd.DataFrame:
    """Cast columns to the given dtypes; a failing conversion raises."""
    missing = [c for c in params.dtypes if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    return df.astype(params.dtypes, errors="raise")


class DropDuplicatesParams(TransformParams):
    subset: list[str] | None = Field(
        default=None, description="Columns defining a duplicate (default: all)"
    )
    keep: Literal["first", "last", "none"] = Field(
        default="last",
        description="Row kept per duplicate group by sort_by order "
        "(last = most recent); 'none' drops every duplicated row",
    )
    sort_by: list[str] | None = Field(
        default=None,
        description="Columns ordering rows before picking first/last (ascending, "
        "so last = greatest); required when keep is first or last",
    )

    @model_validator(mode="after")
    def _sort_by_required(self) -> DropDuplicatesParams:
        if self.keep != "none" and not self.sort_by:
            raise ValueError("sort_by is required when keep is 'first' or 'last'")
        return self


@transform(
    "drop_duplicates", params_model=DropDuplicatesParams, title="Drop duplicates"
)
def drop_duplicates(
    df: pd.DataFrame, params: DropDuplicatesParams, state: dict
) -> pd.DataFrame:
    """Drop duplicate rows, keeping first/last by an explicit sort order."""
    cols = list(params.subset or []) + list(params.sort_by or [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    # Lists / dicts / arrays compared by value (they are not hashable as is).
    pos = hashable_frame(pd.DataFrame(df.reset_index(drop=True)))
    order = (
        pos.sort_values(params.sort_by, kind="stable").index.to_numpy()
        if params.sort_by
        else np.arange(len(pos))
    )
    keep = False if params.keep == "none" else params.keep
    dup = pos.iloc[order].duplicated(subset=params.subset, keep=keep).to_numpy()
    mask = np.ones(len(pos), dtype=bool)
    mask[order[dup]] = False
    return df[mask]


class StandardizeTextParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Text columns to normalize")
    strip: bool = Field(default=True, description="Strip surrounding whitespace")
    lower: bool = Field(default=False, description="Lowercase")
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Variant -> canonical value, matched after strip / lower",
    )


@transform(
    "standardize_text", params_model=StandardizeTextParams, title="Standardize text"
)
def standardize_text(
    df: pd.DataFrame, params: StandardizeTextParams, state: dict
) -> pd.DataFrame:
    """Strip / lowercase text columns and map variants to canonical values."""
    out = df.copy()
    for col in params.columns:
        s = out[col]
        if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)):
            raise TypeError(f"column {col!r} is not text (dtype {s.dtype})")
        if params.strip:
            s = s.str.strip()
        if params.lower:
            s = s.str.lower()
        if params.mapping:
            s = s.mask(s.isin(list(params.mapping)), s.map(params.mapping))
        out[col] = s
    return out


class ParseDatesParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Columns to parse")
    format: str | None = Field(
        default=None, description="strptime format (default: pandas inference)"
    )


@transform("parse_dates", params_model=ParseDatesParams, title="Parse dates")
def parse_dates(
    df: pd.DataFrame, params: ParseDatesParams, state: dict
) -> pd.DataFrame:
    """Parse columns to datetime; unparseable values raise, never become NaT."""
    out = df.copy()
    for col in params.columns:
        out[col] = pd.to_datetime(out[col], format=params.format, errors="raise")
    return out


Scalar = str | int | float | bool


class ReplaceSentinelsParams(TransformParams):
    sentinels: dict[str, list[Scalar]] = Field(
        min_length=1, description="Column -> values to turn into NaN (e.g. -999)"
    )


@transform(
    "replace_sentinels",
    params_model=ReplaceSentinelsParams,
    title="Replace sentinels with NaN",
)
def replace_sentinels(
    df: pd.DataFrame, params: ReplaceSentinelsParams, state: dict
) -> pd.DataFrame:
    """Turn sentinel values (-999, 'N/A', ...) into NaN, per column."""
    out = df.copy()
    for col, values in params.sentinels.items():
        out[col] = out[col].mask(out[col].isin(values))
    return out


class DropMissingTargetParams(TransformParams):
    target: str = Field(description="Target column; rows missing it are dropped")


def _fit_drop_missing_target(df: pd.DataFrame, params: DropMissingTargetParams) -> dict:
    return {"dropped": int(df[params.target].isna().sum())}


@transform(
    "drop_missing_target",
    params_model=DropMissingTargetParams,
    fit=_fit_drop_missing_target,
    title="Drop rows with missing target",
)
def drop_missing_target(
    df: pd.DataFrame, params: DropMissingTargetParams, state: dict
) -> pd.DataFrame:
    """Drop rows whose target is missing (state: rows dropped at fit).

    A frame without the target column (e.g. an unlabelled test) is unchanged.
    """
    if params.target not in df.columns:
        return df.copy()
    return df[df[params.target].notna()]


class Condition(TransformParams):
    column: str
    op: Literal["eq", "ne", "gt", "ge", "lt", "le", "isin", "notin", "isna", "notna"]
    value: Any = None

    @model_validator(mode="after")
    def _value_matches_op(self) -> Condition:
        if self.op in ("isna", "notna"):
            return self
        if self.value is None:
            raise ValueError(f"op {self.op!r} needs a value")
        if self.op in ("isin", "notin") and not isinstance(self.value, list):
            raise ValueError(f"op {self.op!r} needs a list value")
        return self


class FilterRowsParams(TransformParams):
    conditions: list[Condition] = Field(
        min_length=1, description="Conditions (column, op, value) on rows to KEEP"
    )
    combine: Literal["and", "or"] = Field(
        default="and", description="How conditions combine"
    )


def _condition_mask(df: pd.DataFrame, c: Condition) -> pd.Series:
    s = df[c.column]
    match c.op:
        case "eq":
            return s == c.value
        case "ne":
            return s != c.value
        case "gt":
            return s > c.value
        case "ge":
            return s >= c.value
        case "lt":
            return s < c.value
        case "le":
            return s <= c.value
        case "isin":
            return s.isin(c.value)
        case "notin":
            return ~s.isin(c.value)
        case "isna":
            return s.isna()
        case _:
            return s.notna()


@transform("filter_rows", params_model=FilterRowsParams, title="Filter rows")
def filter_rows(
    df: pd.DataFrame, params: FilterRowsParams, state: dict
) -> pd.DataFrame:
    """Keep the rows matching the conditions (no free-form expressions)."""
    masks = [_condition_mask(df, c) for c in params.conditions]
    combine = np.logical_and if params.combine == "and" else np.logical_or
    return df[np.asarray(combine.reduce(masks), dtype=bool)]


class ClipParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Numeric columns to clip")
    lower: float = Field(default=1.0, ge=0, le=100, description="Lower percentile")
    upper: float = Field(default=99.0, ge=0, le=100, description="Upper percentile")

    @model_validator(mode="after")
    def _ordered(self) -> ClipParams:
        if self.lower >= self.upper:
            raise ValueError("lower percentile must be below upper")
        return self


def _fit_clip(df: pd.DataFrame, params: ClipParams) -> dict:
    bounds = {}
    for col in params.columns:
        s = pd.to_numeric(df[col], errors="raise")
        lo, hi = s.quantile([params.lower / 100, params.upper / 100])
        bounds[col] = [
            None if pd.isna(lo) else float(lo),
            None if pd.isna(hi) else float(hi),
        ]
    return {"bounds": bounds}


@transform("clip", params_model=ClipParams, fit=_fit_clip, title="Clip outliers")
def clip(df: pd.DataFrame, params: ClipParams, state: dict) -> pd.DataFrame:
    """Clip columns to percentile bounds learned on train (state: bounds)."""
    out = df.copy()
    for col in params.columns:
        lo, hi = state["bounds"][col]
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


class AlignToTrainParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Numeric columns to align")
    mode: Literal["shift_mean", "shift_median", "standardize", "robust", "quantile"] = (
        Field(
            default="shift_mean",
            description="shift_mean / shift_median: move the frame's mean / median "
            "onto train's; standardize: also rescale std; robust: median + IQR; "
            "quantile: map onto train's quantile function",
        )
    )
    group: str | None = Field(
        default=None,
        description="Column: align per group (statistics fitted on train); unseen "
        "or small groups fall back to the global statistics",
    )
    min_rows: int = Field(
        default=30,
        ge=1,
        description="Fewer non-null frame values than this leaves the column "
        "(or group) unchanged",
    )
    on_small: Literal["skip", "raise"] = Field(
        default="skip", description="A column with fewer than min_rows values"
    )
    n_quantiles: int = Field(
        default=101, ge=2, description="Train quantiles kept for mode 'quantile'"
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _mode_alias(cls, value: Any) -> Any:
        return "standardize" if value == "standardize_to_train" else value

    @model_validator(mode="after")
    def _group_not_aligned(self) -> AlignToTrainParams:
        if self.group is not None and self.group in self.columns:
            raise ValueError("group column cannot be one of the aligned columns")
        return self


def _num(x: float) -> float | None:
    return None if pd.isna(x) else float(x)


def _iqr(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return q3 - q1


def _stats(s: pd.Series, n_quantiles: int | None) -> dict:
    """Location / scale statistics of a numeric series (JSON-safe, NaN -> null)."""
    n = int(s.notna().sum())
    out = {
        "n": n,
        "mean": _num(s.mean()),
        "std": _num(s.std()),
        "median": _num(s.median()),
        "iqr": _num(_iqr(s)),
    }
    if n_quantiles is not None:
        qs = s.quantile(np.linspace(0, 1, n_quantiles)) if n else None
        out["quantiles"] = None if qs is None else [float(v) for v in qs]
    return out


def _fit_align(df: pd.DataFrame, params: AlignToTrainParams) -> dict:
    nq = params.n_quantiles if params.mode == "quantile" else None
    nums = {c: pd.to_numeric(df[c], errors="raise") for c in params.columns}
    state: dict = {"global": {c: _stats(s, nq) for c, s in nums.items()}, "groups": {}}
    if params.group is not None:
        gser = df[params.group]
        for c, s in nums.items():
            state["groups"][c] = {
                str(key): _stats(sub, nq) for key, sub in s.groupby(gser, dropna=True)
            }
    return state


def _positive(x: float | None) -> bool:
    return x is not None and not pd.isna(x) and x > 0


def _align_series(s: pd.Series, ref: dict, mode: str) -> pd.Series:
    """Align ``s`` onto the reference statistics ``ref`` (one column or group)."""
    if mode == "quantile":
        qs = ref.get("quantiles")
        if qs is None:
            return s
        n = int(s.notna().sum())
        rank = s.rank(method="average")
        q = (rank - 1) / (n - 1) if n > 1 else rank * 0 + 0.5
        vals = np.interp(q.to_numpy(dtype=float), np.linspace(0, 1, len(qs)), qs)
        return pd.Series(vals, index=s.index).where(s.notna())
    if mode in ("shift_median", "robust"):
        loc_f, scale_f = s.median(), _iqr(s)
        loc_t, scale_t = ref["median"], ref["iqr"]
    else:
        loc_f, scale_f = s.mean(), s.std()
        loc_t, scale_t = ref["mean"], ref["std"]
    if loc_t is None or pd.isna(loc_f):
        return s
    centered = s - loc_f
    if mode in ("standardize", "robust") and _positive(scale_f) and _positive(scale_t):
        centered = centered / scale_f * scale_t
    return centered + loc_t


@transform(
    "align_to_train",
    params_model=AlignToTrainParams,
    fit=_fit_align,
    title="Align to train statistics",
)
def align_to_train(
    df: pd.DataFrame, params: AlignToTrainParams, state: dict
) -> pd.DataFrame:
    """Realign a shifted / broken numeric column onto train's distribution.

    Modes: ``shift_mean`` (x - mean + train mean), ``shift_median``,
    ``standardize`` (also rescale std; ``standardize_to_train`` is an accepted
    alias), ``robust`` (median and IQR) and ``quantile`` (empirical quantile
    within the frame -> train's quantile function, ``n_quantiles`` train
    quantiles, average ranks for ties, NaN kept).

    State holds train's statistics (global and, with ``group``, per group);
    the frame's own statistics are measured at apply time, so applying to
    train itself is the identity (approximately so for ``quantile``). A zero
    scale (std / IQR) on either side means shift only. A column with fewer
    than ``min_rows`` non-null values is left unchanged (``on_small="raise"``
    raises). With ``group``, a group unseen in train or with fewer than
    ``min_rows`` rows on either side uses the global statistics.
    """
    missing = [c for c in [*params.columns, params.group] if c and c not in df]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    out = df.copy()
    gvals = None if params.group is None else out[params.group].reset_index(drop=True)
    for col in params.columns:
        s = pd.to_numeric(out[col], errors="raise").astype(float)
        n = int(s.notna().sum())
        if n < params.min_rows:
            if params.on_small == "raise":
                raise ValueError(
                    f"column {col!r}: {n} non-null values < min_rows={params.min_rows}"
                )
            continue
        s = s.reset_index(drop=True)
        aligned = _align_series(s, state["global"][col], params.mode)
        if gvals is not None:
            groups = state["groups"].get(col, {})
            for key, sub in s.groupby(gvals, dropna=True):
                ref = groups.get(str(key))
                if (
                    ref is None
                    or ref["n"] < params.min_rows
                    or int(sub.notna().sum()) < params.min_rows
                ):
                    continue
                aligned.loc[sub.index] = _align_series(sub, ref, params.mode)
        out[col] = aligned.to_numpy()
    return out
