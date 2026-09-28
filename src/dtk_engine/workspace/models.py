"""Workspace model: which files make train / test, how labels join, the step log."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dtk_engine.sources.spec import FileSourceSpec

# A workspace name is also its file name: no path separators, no leading dot.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
VARIABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
VARIABLE_STATS = ("mean", "median", "std", "min", "max", "q25", "q75", "count")
VariableStat = Literal["mean", "median", "std", "min", "max", "q25", "q75", "count"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(_Strict):
    """One dataset (train or test): X, and its labels as a y file or an X column."""

    x: FileSourceSpec
    y: FileSourceSpec | None = Field(default=None, description="Separate label file")
    target_column: str | None = Field(
        default=None, description="Label column already in X (instead of a y file)"
    )

    @model_validator(mode="after")
    def _one_label_source(self) -> DatasetSpec:
        if self.y is not None and self.target_column is not None:
            raise ValueError("give either `y` or `target_column`, not both")
        return self


class Datasets(_Strict):
    train: DatasetSpec
    test: DatasetSpec | None = None


class LabelJoin(_Strict):
    """How y joins X: by row order, or on a common key column."""

    mode: Literal["order", "key"] = "order"
    key: str | None = Field(
        default=None,
        description='Join column (required for mode "key"; for "order", an optional '
        "id column of y checked against X)",
    )

    @model_validator(mode="after")
    def _key_mode_needs_key(self) -> LabelJoin:
        if self.mode == "key" and not self.key:
            raise ValueError('label mode "key" needs a `key` column')
        return self


class Step(_Strict):
    """One logged transform: ``op`` (a registered @transform) applied to ``target``."""

    op: str
    target: Literal["train", "test", "both"]
    params: dict[str, Any] = Field(default_factory=dict)


class MergeSpec(_Strict):
    """Extra columns joined onto train and/or test on a key column."""

    source: FileSourceSpec
    key: str = Field(min_length=1)
    apply_to: Literal["train", "both"] = "both"
    columns: list[str] | None = Field(
        default=None, description="null = all non-key columns"
    )


class VariableSpec(_Strict):
    """A statistic computed on a train column, frozen for expressions."""

    name: str = Field(pattern=VARIABLE_NAME_PATTERN)
    stat: VariableStat
    column: str = Field(min_length=1)


class ChartSpec(_Strict):
    """A named chart saved on the workspace (chart-key params minus source)."""

    name: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class Workspace(_Strict):
    name: str = Field(pattern=NAME_PATTERN)
    datasets: Datasets
    label: LabelJoin = Field(default_factory=LabelJoin)
    merges: list[MergeSpec] = Field(default_factory=list)
    variables: list[VariableSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)

    @field_validator("variables")
    @classmethod
    def _unique_variable_names(cls, vars: list[VariableSpec]) -> list[VariableSpec]:
        seen = set()
        for v in vars:
            if v.name in seen:
                raise ValueError(f"duplicate variable name {v.name!r}")
            seen.add(v.name)
        return vars

    @field_validator("charts")
    @classmethod
    def _unique_chart_names(cls, charts: list[ChartSpec]) -> list[ChartSpec]:
        seen = set()
        for c in charts:
            if c.name in seen:
                raise ValueError(f"duplicate chart name {c.name!r}")
            seen.add(c.name)
        return charts

    @field_validator("merges")
    @classmethod
    def _validate_merges(cls, merges: list[MergeSpec]) -> list[MergeSpec]:
        from dtk_engine.errors import SourceError
        from dtk_engine.sources.registry import load

        for m in merges:
            try:
                df = load(m.source)
            except (SourceError, FileNotFoundError, OSError):
                df = None
            if df is not None and m.key in df.columns:
                null_keys = int(df[m.key].isna().sum())
                if null_keys:
                    raise ValueError(
                        f"merge source key {m.key!r} contains {null_keys} null value(s)"
                    )
                dup_mask = df[m.key].duplicated(keep=False)
                if dup_mask.any():
                    dups = df.loc[dup_mask, m.key].dropna().unique().tolist()
                    raise ValueError(
                        f"merge source key {m.key!r} has duplicated keys: {dups}"
                    )
        return merges

