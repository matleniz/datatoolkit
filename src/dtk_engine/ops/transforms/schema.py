"""``expected_schema``: check the frame against a declared schema (data unchanged).

The step is a data contract: it passes the frame through untouched and reports
every violation at once (``ops/schema_check.py``). With ``on_violation="report"``
the findings are recorded on the step (``checks`` of a preview, ``checks`` of the
replay's fitted entry) and the replay goes on; with ``"error"`` an error finding
fails the step like any other failing step.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from dtk_engine.ops.schema_check import check_schema, effective_dtype, summarize
from dtk_engine.transform_registry import TransformParams, transform

MAX_COLUMN_RULES = 500
MAX_ALLOWED = 1000
MAX_EXAMPLES = 50

Bound = StrictInt | StrictFloat | StrictStr
Scalar = StrictBool | StrictInt | StrictFloat | StrictStr


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field} {value!r} is not a number or a date") from exc


class ColumnRule(BaseModel):
    """What one column must satisfy (every constraint but ``name`` is optional)."""

    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, description="Column name (compared as text)")
    required: bool = Field(default=True, description="The column must be present")
    dtype: Literal["numeric", "string", "bool", "datetime", "category"] | None = Field(
        default=None,
        description="Expected kind; category accepts a categorical or a string column",
    )
    nullable: bool = Field(default=True, description="False: no missing value at all")
    max_missing_pct: float | None = Field(
        default=None, ge=0, le=100, description="Largest share of missing values, in %"
    )
    min: Bound | None = Field(
        default=None, description="Smallest value (number, or ISO date for a datetime)"
    )
    max: Bound | None = Field(
        default=None, description="Largest value (number, or ISO date for a datetime)"
    )
    allowed: list[Scalar] | None = Field(
        default=None, max_length=MAX_ALLOWED, description="The only values permitted"
    )
    unique: bool = Field(default=False, description="No repeated (non-missing) value")

    @field_validator("min", "max")
    @classmethod
    def _finite(cls, value: Any, info) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.allowed is not None and not self.allowed:
            raise ValueError(f"column {self.name!r}: allowed must not be empty")
        if not self.nullable and self.max_missing_pct is not None:
            raise ValueError(
                f"column {self.name!r}: nullable=false and max_missing_pct are contradictory"
            )
        self._check_bounds()
        self._check_allowed()
        return self

    def _check_bounds(self) -> None:
        bounds = [b for b in (self.min, self.max) if b is not None]
        if not bounds:
            return
        dtype = effective_dtype(self)
        if dtype not in ("numeric", "datetime"):
            raise ValueError(
                f"column {self.name!r}: min / max need dtype numeric or datetime, got {dtype}"
            )
        if dtype == "numeric":
            if not all(_is_number(b) for b in bounds):
                raise ValueError(f"column {self.name!r}: min / max must be numbers")
        else:
            bounds = [_timestamp(b, f"column {self.name!r}: bound") for b in bounds]
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            raise ValueError(f"column {self.name!r}: min {self.min!r} is above max {self.max!r}")

    def _check_allowed(self) -> None:
        if self.allowed is None:
            return
        ok = {
            None: lambda v: True,
            "numeric": _is_number,
            "string": lambda v: isinstance(v, str),
            "category": lambda v: isinstance(v, str),
            "bool": lambda v: isinstance(v, bool),
        }
        if self.dtype == "datetime":
            raise ValueError(f"column {self.name!r}: allowed is not supported for datetime")
        bad = [v for v in self.allowed if not ok[self.dtype](v)]
        if bad:
            raise ValueError(
                f"column {self.name!r}: allowed values {bad[:3]} do not fit dtype {self.dtype}"
            )


class ExpectedSchemaParams(TransformParams):
    schema_version: Literal[1] = Field(default=1, description="Version of this schema format")
    on_violation: Literal["report", "error"] = Field(
        default="report",
        description="report: record the violations and go on; error: fail the step "
        "when there is an error-level violation",
    )
    max_examples: int = Field(
        default=5, ge=0, le=MAX_EXAMPLES, description="Offending rows quoted per finding"
    )
    extra_columns: Literal["ignore", "warn"] = Field(
        default="warn", description="What to do with columns the schema does not declare"
    )
    min_rows: int | None = Field(default=None, ge=0, description="Fewest rows expected")
    max_rows: int | None = Field(default=None, ge=0, description="Most rows expected")
    columns: list[ColumnRule] = Field(
        default_factory=list, max_length=MAX_COLUMN_RULES, description="One rule per column"
    )

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [r.name for r in self.columns]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate column rules for {dupes}")
        if None not in (self.min_rows, self.max_rows) and self.min_rows > self.max_rows:
            raise ValueError(f"min_rows {self.min_rows} is above max_rows {self.max_rows}")
        return self


def _fit(df: pd.DataFrame, params: ExpectedSchemaParams) -> dict:
    """Counts only, so the fit panel shows the contract's verdict on the fit frame."""
    return {"n_rows": len(df), "violations": summarize(check_schema(df, params))}


def _check(df: pd.DataFrame, params: ExpectedSchemaParams, state: dict) -> list[dict]:
    found = check_schema(df, params)
    errors = [f for f in found if f["severity"] == "error"]
    if errors and params.on_violation == "error":
        lines = "; ".join(
            f"{f['column']}: {f['message']}" if f["column"] else f["message"]
            for f in errors[:10]
        )
        more = f" (+{len(errors) - 10} more)" if len(errors) > 10 else ""
        raise ValueError(f"expected schema violated ({len(errors)} errors): {lines}{more}")
    return found


@transform(
    "expected_schema",
    params_model=ExpectedSchemaParams,
    fit=_fit,
    check=_check,
    title="Expected schema",
)
def expected_schema(df: pd.DataFrame, params: ExpectedSchemaParams, state: dict) -> pd.DataFrame:
    """Check the frame against a declared schema; the data passes through unchanged."""
    return df.copy(deep=False)
