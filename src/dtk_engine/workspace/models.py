"""Workspace model: which files make train / test, how labels join, the step log."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dtk_engine.sources.spec import FileSourceSpec

# A workspace name is also its file name: no path separators, no leading dot.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


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


class Workspace(_Strict):
    name: str = Field(pattern=NAME_PATTERN)
    datasets: Datasets
    label: LabelJoin = Field(default_factory=LabelJoin)
    steps: list[Step] = Field(default_factory=list)
