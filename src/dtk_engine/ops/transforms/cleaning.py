"""Cleaning ops: drop, rename, cast, dedupe, text, dates, sentinels, filter, clip."""

from __future__ import annotations

import operator
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dtk_engine.ops.profile import hashable_frame
from dtk_engine.params import column_field, columns_field, when
from dtk_engine.transform_registry import TransformParams, transform


class DropColumnsParams(TransformParams):
    columns: list[str] = columns_field(
        "Columns to drop", source="step", required=True, min_length=1
    )
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


class SelectColumnsParams(TransformParams):
    columns: list[str] = columns_field(
        "Columns to keep, in this order", source="step", required=True, min_length=1
    )
    missing_ok: bool = Field(
        default=False, description="Ignore listed columns absent from the frame"
    )

    @field_validator("columns")
    @classmethod
    def _no_repeat(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("columns must not repeat")
        return v


@transform(
    "select_columns", params_model=SelectColumnsParams, title="Select columns"
)
def select_columns(
    df: pd.DataFrame, params: SelectColumnsParams, state: dict
) -> pd.DataFrame:
    """Keep only the listed columns, in the given order."""
    missing = [c for c in params.columns if c not in df.columns]
    if missing and not params.missing_ok:
        raise KeyError(f"columns not in frame: {missing}")
    return df[[c for c in params.columns if c in df.columns]]


class ReorderColumnsParams(TransformParams):
    columns: list[str] = columns_field(
        "Columns to move, kept in this order", source="step", required=True, min_length=1
    )
    position: Literal["first", "last", "before", "after"] = Field(
        description="Where to put them: at the start, at the end, or next to `anchor`"
    )
    anchor: str | None = column_field(
        None, "Column to sit before / after (required for 'before' / 'after')",
        source="step",
        extra=when(position=["before", "after"]),
    )
    missing_ok: bool = Field(
        default=False, description="Ignore listed columns absent from the frame"
    )

    @model_validator(mode="after")
    def _check(self) -> ReorderColumnsParams:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("columns must not repeat")
        needs_anchor = self.position in ("before", "after")
        if needs_anchor and self.anchor is None:
            raise ValueError(f"anchor is required when position is {self.position!r}")
        if not needs_anchor and self.anchor is not None:
            raise ValueError("anchor only applies to position 'before' / 'after'")
        if self.anchor in self.columns:
            raise ValueError("anchor must not be one of the moved columns")
        return self


@transform(
    "reorder_columns", params_model=ReorderColumnsParams, title="Reorder columns"
)
def reorder_columns(
    df: pd.DataFrame, params: ReorderColumnsParams, state: dict
) -> pd.DataFrame:
    """Move the listed columns to the start, the end, or next to an anchor column."""
    missing = [c for c in params.columns if c not in df.columns]
    if missing and not params.missing_ok:
        raise KeyError(f"columns not in frame: {missing}")
    if params.anchor is not None and params.anchor not in df.columns:
        raise KeyError(f"anchor not in frame: {params.anchor!r}")
    moved = [c for c in params.columns if c in df.columns]
    rest = [c for c in df.columns if c not in moved]
    if params.position == "first":
        order = moved + rest
    elif params.position == "last":
        order = rest + moved
    else:
        at = rest.index(params.anchor) + (params.position == "after")
        order = rest[:at] + moved + rest[at:]
    return df[order]


class CopyColumnParams(TransformParams):
    column: str = column_field(..., "Column to copy", source="step")
    name: str = Field(min_length=1, description="Name of the new column")
    position: Literal["first", "last", "before", "after"] | None = Field(
        default=None,
        description="Where to put the copy: at the start, at the end, or next to "
        "`anchor` (default: at the end)",
    )
    anchor: str | None = column_field(
        None, "Column to sit before / after (required for 'before' / 'after')",
        source="step",
        extra=when(position=["before", "after"]),
    )

    @model_validator(mode="after")
    def _check(self) -> CopyColumnParams:
        needs_anchor = self.position in ("before", "after")
        if needs_anchor and self.anchor is None:
            raise ValueError(f"anchor is required when position is {self.position!r}")
        if not needs_anchor and self.anchor is not None:
            raise ValueError("anchor only applies to position 'before' / 'after'")
        if self.anchor == self.name:
            raise ValueError("anchor must not be the new column")
        return self


@transform("copy_column", params_model=CopyColumnParams, title="Copy a column")
def copy_column(
    df: pd.DataFrame, params: CopyColumnParams, state: dict
) -> pd.DataFrame:
    """Duplicate a column under a new name, optionally placed like reorder_columns."""
    if params.column not in df.columns:
        raise KeyError(f"column not in frame: {params.column!r}")
    if params.name in df.columns:
        raise ValueError(f"column already exists: {params.name!r}")
    out = df.assign(**{params.name: df[params.column]})
    if params.position is None:
        return out
    move = ReorderColumnsParams(
        columns=[params.name], position=params.position, anchor=params.anchor
    )
    return reorder_columns(out, move, {})


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
    subset: list[str] | None = columns_field(
        "Columns defining a duplicate (default: all)", source="step", nullable=True
    )
    keep: Literal["first", "last", "none"] = Field(
        default="last",
        description="Row kept per duplicate group by sort_by order "
        "(last = most recent); 'none' drops every duplicated row",
    )
    sort_by: list[str] | None = columns_field(
        "Columns ordering rows before picking first/last (ascending, "
        "so last = greatest); required when keep is first or last",
        source="step",
        nullable=True,
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


_SEPARATOR_RE = re.compile(r"[-_.]+")
_WHITESPACE_RE = re.compile(r"\s+")


class StandardizeTextParams(TransformParams):
    columns: list[str] = columns_field(
        "Text columns to normalize", source="step", required=True, min_length=1
    )
    strip: bool = Field(default=True, description="Strip surrounding whitespace")
    lower: bool = Field(default=False, description="Lowercase")
    unify_separators: bool = Field(
        default=False,
        description="Turn '-', '_', '.' and repeated spaces into a single space "
        "(then strip), so 'site-a' / 'site_a' / 'Site  A' collapse together",
    )
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Variant -> canonical value, matched after strip / lower / "
        "unify_separators",
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
        if params.unify_separators:
            s = s.str.replace(_SEPARATOR_RE, " ", regex=True)
            s = s.str.replace(_WHITESPACE_RE, " ", regex=True).str.strip()
        if params.lower:
            s = s.str.lower()
        if params.mapping:
            s = s.mask(s.isin(list(params.mapping)), s.map(params.mapping))
        out[col] = s
    return out


# Currency symbol or ISO code (stripped unconditionally, wherever it sits).
_CURRENCY_TOKEN_RE = re.compile(r"(?i)[$€£]|USD|EUR|GBP")
# Whitespace-like thousands separators (regular space, NBSP, narrow NBSP): a
# `thousands=" "` param strips all three, not just the plain space.
_THOUSANDS_WS_RE = re.compile(r"[   ]")
# Trailing whole-unit notation ('990,-' / '990.-'): no cents, drop the marker.
_WHOLE_UNIT_RE = re.compile(r"[.,]-$")


class ToNumericParams(TransformParams):
    columns: list[str] = columns_field(
        "Text columns to parse as numbers", source="step", required=True, min_length=1
    )
    decimal: Literal[".", ","] = Field(
        default=".", description="Decimal separator"
    )
    thousands: Literal[",", ".", " "] | None = Field(
        default=None, description="Thousands separator to drop (must differ from decimal)"
    )
    percent: bool = Field(
        default=False,
        description="Divide by 100 when the value carries a '%' sign "
        "(else the sign is stripped without scaling)",
    )
    errors: Literal["coerce", "raise"] = Field(
        default="raise",
        description="coerce: unparseable values become NaN; raise: they raise",
    )

    @model_validator(mode="after")
    def _sep_differ(self) -> ToNumericParams:
        if self.thousands is not None and self.thousands == self.decimal:
            raise ValueError("thousands and decimal separators must differ")
        return self


def _parse_numeric_cell(
    value: object, params: ToNumericParams
) -> float:
    if not isinstance(value, str):
        raise TypeError(f"not a string: {value!r}")
    s = value.strip()
    has_percent = "%" in s
    s = s.replace("%", "")
    s = _CURRENCY_TOKEN_RE.sub("", s).strip()
    s = _WHOLE_UNIT_RE.sub("", s).strip()
    if params.thousands == " ":
        s = _THOUSANDS_WS_RE.sub("", s)
    elif params.thousands:
        s = s.replace(params.thousands, "")
    if params.decimal != ".":
        s = s.replace(params.decimal, ".")
    s = s.strip()
    number = float(s)
    if params.percent and has_percent:
        number /= 100
    return number


@transform("to_numeric", params_model=ToNumericParams, title="Parse numeric text")
def to_numeric(df: pd.DataFrame, params: ToNumericParams, state: dict) -> pd.DataFrame:
    """Parse text numbers (currency symbols, thousands / decimal separators
    including NBSP-style spaces, percent signs, trailing ',-' / '.-' whole-unit
    notation) to float; `errors` controls unparseable values."""
    out = df.copy()
    for col in params.columns:

        def _parse(v, col=col):
            if pd.isna(v):
                return np.nan
            try:
                return _parse_numeric_cell(v, params)
            except (ValueError, TypeError):
                if params.errors == "coerce":
                    return np.nan
                raise ValueError(f"column {col!r}: cannot parse {v!r} as a number") from None

        out[col] = out[col].map(_parse).astype(float)
    return out


class ParseDatesParams(TransformParams):
    columns: list[str] = columns_field(
        "Columns to parse", source="step", required=True, min_length=1
    )
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


# Cap user-supplied patterns so a catastrophic-backtracking regex cannot hang
# the engine (stdlib ``re`` has no match timeout). Documented on the param.
MAX_EXTRACT_PATTERN_LENGTH = 256

_NUMERIC_GROUP_RE = re.compile(
    r"""
    ^\s*
    [+-]?
    (?:
        \d+(?:[._]\d+)?   # 1950, 2013.5, 1_000
        |\.\d+            # .5
    )
    \s*$
    """,
    re.VERBOSE,
)


def _compile_extract_pattern(pattern: str) -> re.Pattern[str]:
    """Compile ``pattern``; refuse invalid regex or patterns without named groups."""
    if len(pattern) > MAX_EXTRACT_PATTERN_LENGTH:
        raise ValueError(
            f"extract: pattern longer than {MAX_EXTRACT_PATTERN_LENGTH} characters "
            "(ReDoS guard; shorten the pattern)"
        )
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"extract: invalid pattern ({exc})") from exc
    if not compiled.groupindex:
        raise ValueError(
            "extract: pattern must include at least one named group "
            "(?P<name>...)"
        )
    return compiled


def _group_looks_numeric(values: pd.Series) -> bool:
    """True when every non-null capture looks like a number (altitude, year…)."""
    non_null = values.dropna()
    if non_null.empty:
        return False
    return all(_NUMERIC_GROUP_RE.match(str(v)) is not None for v in non_null)


class ExtractParams(TransformParams):
    column: str = column_field(
        ..., "Text column to match against the pattern", source="step"
    )
    pattern: str = Field(
        min_length=1,
        max_length=MAX_EXTRACT_PATTERN_LENGTH,
        description=(
            "Python regex with named capture groups (?P<name>...); each group "
            "becomes a column '<prefix>_<name>' (or '<column>_<name>'). "
            f"At most {MAX_EXTRACT_PATTERN_LENGTH} characters (stdlib re has no "
            "match timeout; the length cap is the ReDoS guard). "
            "Groups whose captures all look numeric are cast to float."
        ),
    )
    prefix: str | None = Field(
        default=None,
        description="Prefix for output columns (default: the source column name)",
    )
    errors: Literal["raise", "coerce"] = Field(
        default="raise",
        description="Non-null cell with no match: raise, or fill group columns with NaN",
    )

    @field_validator("pattern")
    @classmethod
    def _valid_pattern(cls, value: str) -> str:
        _compile_extract_pattern(value)
        return value


@transform("extract", params_model=ExtractParams, title="Extract by regex")
def extract(df: pd.DataFrame, params: ExtractParams, state: dict) -> pd.DataFrame:
    """Pull named regex groups from a text column into new typed columns.

    Numeric-looking groups (e.g. altitude low/high, harvest years) are cast to
    float. Pattern length is capped at ``MAX_EXTRACT_PATTERN_LENGTH`` (stdlib
    ``re`` has no match timeout).
    """
    if params.column not in df.columns:
        raise KeyError(f"extract: column {params.column!r} not in the frame")
    compiled = _compile_extract_pattern(params.pattern)
    group_names = list(compiled.groupindex)
    prefix = params.prefix if params.prefix is not None else params.column
    out_names = [f"{prefix}_{g}" for g in group_names]
    clash = [n for n in out_names if n in df.columns]
    if clash:
        raise ValueError(f"extract: output columns already exist {clash}")

    source = df[params.column]
    # StringDtype keeps NA (avoids matching the literals "nan" / "None").
    text = source.astype("string")
    captured = text.str.extract(compiled, expand=True)
    # pandas names columns from groupindex; enforce our declared order.
    captured = captured.reindex(columns=group_names)
    captured.columns = out_names

    unmatched = text.notna() & captured.isna().all(axis=1)
    if unmatched.any() and params.errors == "raise":
        first = source[unmatched].iloc[0]
        raise ValueError(
            f"extract: no match for {first!r} in column {params.column!r} "
            f"with pattern {params.pattern!r}"
        )

    out = df.copy()
    for name in out_names:
        col = captured[name]
        if _group_looks_numeric(col):
            out[name] = pd.to_numeric(col, errors="coerce").astype(float)
        else:
            out[name] = col.astype(object).where(col.notna(), other=np.nan)
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


class ReplaceValuesParams(TransformParams):
    mapping: dict[str, dict[str, Scalar | None]] = Field(
        min_length=1,
        description=(
            "Column -> {old value: new value}; a null new value gives NaN, values "
            "absent from the mapping are untouched. JSON keys are strings: on a "
            "numeric column a key matches the numbers equal to it ('1' matches 1 "
            "and 1.0; a non-numeric key matches nothing), on a boolean column "
            "'true' / 'false', otherwise the exact text of the value"
        ),
    )

    @field_validator("mapping")
    @classmethod
    def _non_empty(cls, v: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        empty = [c for c, m in v.items() if not m]
        if empty:
            raise ValueError(f"empty mapping for columns: {empty}")
        return v


def _key_matches(col: pd.Series, key: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(col):
        return col.astype(str).str.lower().eq(key.strip().lower())
    if pd.api.types.is_numeric_dtype(col):
        try:
            return col.eq(float(key))
        except ValueError:
            return pd.Series(False, index=col.index)
    return col.astype(object).map(lambda v: isinstance(v, str) and v == key).astype(bool)


@transform("replace_values", params_model=ReplaceValuesParams, title="Replace values")
def replace_values(
    df: pd.DataFrame, params: ReplaceValuesParams, state: dict
) -> pd.DataFrame:
    """Recode explicit values to others, per column (old -> new, new may be null)."""
    missing = [c for c in params.mapping if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    out = df.copy()
    for name, recode in params.mapping.items():
        col = df[name]
        hits = {k: _key_matches(col, k) for k in recode}
        for a, b in [(a, b) for a in hits for b in hits if a < b]:
            if (hits[a] & hits[b]).any():
                raise ValueError(f"keys {a!r} and {b!r} match the same value in {name!r}")
        if not any(m.any() for m in hits.values()):
            continue
        new = col.astype(object)
        for key, mask in hits.items():
            value = recode[key]
            new = new.mask(mask, np.nan if value is None else value)
        out[name] = new.infer_objects()
    return out


class DropMissingTargetParams(TransformParams):
    target: str = column_field(
        ..., "Target column; rows missing it are dropped", source="step"
    )


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


class DropHighMissingParams(TransformParams):
    threshold: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Drop columns whose train missing fraction exceeds this",
    )
    exclude: list[str] | None = columns_field(
        "Columns never dropped", source="step", nullable=True
    )
    target: str | None = column_field(None, "Target column: never dropped", source="step")


def _fit_drop_high_missing(df: pd.DataFrame, params: DropHighMissingParams) -> dict:
    exclude = set(params.exclude or [])
    if params.target is not None:
        exclude.add(params.target)
    rates = df.isna().mean()
    dropped = [
        str(c) for c in df.columns if c not in exclude and rates[c] > params.threshold
    ]
    return {"dropped": dropped}


@transform(
    "drop_high_missing",
    params_model=DropHighMissingParams,
    fit=_fit_drop_high_missing,
    title="Drop high-missing columns",
)
def drop_high_missing(
    df: pd.DataFrame, params: DropHighMissingParams, state: dict
) -> pd.DataFrame:
    """Drop columns whose train missing fraction exceeded the threshold
    (state: dropped), same columns dropped on train and test."""
    missing = [c for c in state["dropped"] if c not in df.columns]
    if missing:
        raise KeyError(f"drop_high_missing: fitted columns not in the frame {missing}")
    return df.drop(columns=state["dropped"])


class Condition(TransformParams):
    column: str = column_field(..., "Column the condition tests", source="step")
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


_CONDITION_MASKS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "isin": lambda s, v: s.isin(v),
    "notin": lambda s, v: ~s.isin(v),
    "isna": lambda s, v: s.isna(),
}


def _condition_mask(df: pd.DataFrame, c: Condition) -> pd.Series:
    s = df[c.column]
    mask = _CONDITION_MASKS.get(c.op)
    return mask(s, c.value) if mask else s.notna()  # notna: the last op


@transform("filter_rows", params_model=FilterRowsParams, title="Filter rows")
def filter_rows(
    df: pd.DataFrame, params: FilterRowsParams, state: dict
) -> pd.DataFrame:
    """Keep the rows matching the conditions (no free-form expressions)."""
    masks = [_condition_mask(df, c) for c in params.conditions]
    combine = np.logical_and if params.combine == "and" else np.logical_or
    return df[np.asarray(combine.reduce(masks), dtype=bool)]


class SortRowsParams(TransformParams):
    by: list[str] = columns_field(
        "Columns to sort by, most significant first",
        source="step",
        required=True,
        min_length=1,
    )
    ascending: bool | list[bool] = Field(
        default=True,
        description="Ascending order: one bool for all columns, or one per column",
    )
    na_position: Literal["first", "last"] = Field(
        default="last", description="Where missing values go"
    )

    @model_validator(mode="after")
    def _check(self) -> SortRowsParams:
        if len(set(self.by)) != len(self.by):
            raise ValueError("by must not repeat")
        if isinstance(self.ascending, list) and len(self.ascending) != len(self.by):
            raise ValueError("ascending list must have one entry per column in by")
        return self


@transform("sort_rows", params_model=SortRowsParams, title="Sort rows")
def sort_rows(df: pd.DataFrame, params: SortRowsParams, state: dict) -> pd.DataFrame:
    """Stable sort by one or more columns; each row keeps its index (row id)."""
    missing = [c for c in params.by if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    return df.sort_values(
        params.by,
        ascending=params.ascending,
        na_position=params.na_position,
        kind="stable",
    )


class ClipParams(TransformParams):
    columns: list[str] = columns_field(
        "Numeric columns to clip",
        source="step",
        dtype="numeric",
        required=True,
        min_length=1,
    )
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


class SampleRowsParams(TransformParams):
    mode: Literal["random", "head"] = Field(
        default="random", description="Random subsample, or the first rows"
    )
    n: int | None = Field(
        default=None, ge=1, description="Number of rows to keep (or give `frac`)"
    )
    frac: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Fraction of the rows to keep (or give `n`)",
    )
    random_state: int | None = Field(
        default=None,
        description="Seed of the random draw (required: replay must be deterministic)",
        json_schema_extra=when(mode="random"),
    )

    @model_validator(mode="after")
    def _check(self) -> SampleRowsParams:
        if (self.n is None) == (self.frac is None):
            raise ValueError("give exactly one of n or frac")
        if self.mode == "random" and self.random_state is None:
            raise ValueError("random_state is required when mode is 'random'")
        if self.mode == "head" and self.random_state is not None:
            raise ValueError("random_state only applies to mode 'random'")
        return self


@transform("sample_rows", params_model=SampleRowsParams, title="Sample rows")
def sample_rows(
    df: pd.DataFrame, params: SampleRowsParams, state: dict
) -> pd.DataFrame:
    """Keep the first n rows, or a seeded random subsample (original order).

    Stateless; rows kept keep their index (row id). Allowed on any target:
    sampling the test frame changes what is scored, so prefer target ``train``.
    """
    total = len(df)
    k = params.n if params.n is not None else round(total * params.frac)
    k = min(k, total)
    if params.mode == "head":
        return df.iloc[:k]
    rng = np.random.default_rng(params.random_state)
    return df.iloc[np.sort(rng.choice(total, size=k, replace=False))]
