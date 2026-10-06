"""Workspace model: which files make train / test, how labels join, the step log."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dtk_engine.sources.spec import FileSourceSpec

# A workspace name is also its file name: no path separators, no leading dot.
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
VARIABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
# A step id names a pipeline slot (kept when the step is replaced), opaque.
STEP_ID_PATTERN = r"^s[0-9a-z-]{1,32}$"
NOTE_MAX = 4000  # characters per note
DOCUMENT_ID_PATTERN = r"^d[0-9a-z-]{1,32}$"
DOCUMENTS_MAX = 50
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
    """One logged transform: ``op`` (a registered @transform) applied to ``target``.

    ``id`` is stable across edits; a workspace fills the missing ones
    (``fill_step_ids``). It never feeds a frame (not in cache keys).
    """

    id: str | None = Field(default=None, pattern=STEP_ID_PATTERN)
    op: str
    target: Literal["train", "test", "both"]
    params: dict[str, Any] = Field(default_factory=dict)
    note: str | None = Field(default=None, max_length=NOTE_MAX)


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


class WorkspaceNotes(_Strict):
    """Free-text notes: on the workspace, and per column keyed by the column's
    origin name (its source name, or the name a step created it with), so a
    note survives ``rename`` steps (``inspect.column_notes`` resolves names)."""

    workspace: str | None = Field(default=None, max_length=NOTE_MAX)
    columns: dict[str, str] = Field(default_factory=dict)

    @field_validator("columns")
    @classmethod
    def _column_notes(cls, notes: dict[str, str]) -> dict[str, str]:
        long = [k for k, v in notes.items() if len(v) > NOTE_MAX]
        if long:
            raise ValueError(f"column notes over {NOTE_MAX} characters: {long}")
        return {k: v for k, v in notes.items() if v}


class WorkspaceDocument(_Strict):
    """A reference file kept with the workspace (datatoolkit-issues#178): an
    upload ref (``path``, content-addressed, never copied) and how to read it.
    ``id`` is filled like step ids (``fill_document_ids``)."""

    id: str | None = Field(default=None, pattern=DOCUMENT_ID_PATTERN)
    name: str = Field(min_length=1, max_length=255)
    path: str = Field(min_length=1, description="Upload ref (PUT /api/uploads answer)")
    mime: str = "application/octet-stream"
    size: int = Field(default=0, ge=0)
    kind: Literal["text", "pdf", "table", "other"] = "other"
    added_at: str | None = None
    note: str | None = Field(default=None, max_length=NOTE_MAX)


class Workspace(_Strict):
    name: str = Field(pattern=NAME_PATTERN)
    datasets: Datasets
    label: LabelJoin = Field(default_factory=LabelJoin)
    merges: list[MergeSpec] = Field(default_factory=list)
    variables: list[VariableSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    notes: WorkspaceNotes = Field(default_factory=WorkspaceNotes)
    documents: list[WorkspaceDocument] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def _step_ids(cls, steps: list[Step]) -> list[Step]:
        return fill_step_ids(steps)

    @field_validator("documents")
    @classmethod
    def _document_ids(cls, docs: list[WorkspaceDocument]) -> list[WorkspaceDocument]:
        if len(docs) > DOCUMENTS_MAX:
            raise ValueError(f"at most {DOCUMENTS_MAX} documents per workspace")
        return fill_document_ids(docs)

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


def fill_step_ids(steps: list[Step]) -> list[Step]:
    """Give each step without an id ``s<position>`` (1-based; ``-2``, ``-3``...
    on a clash): deterministic, so a workspace stored before step ids reads
    the same ids until it is saved with them. Duplicate ids -> ValueError."""
    taken: set[str] = set()
    for step in steps:
        if step.id is None:
            continue
        if step.id in taken:
            raise ValueError(f"duplicate step id {step.id!r}")
        taken.add(step.id)
    for pos, step in enumerate(steps, start=1):
        if step.id is not None:
            continue
        candidate, n = f"s{pos}", 1
        while candidate in taken:
            n += 1
            candidate = f"s{pos}-{n}"
        step.id = candidate
        taken.add(candidate)
    return steps


def fill_document_ids(docs: list[WorkspaceDocument]) -> list[WorkspaceDocument]:
    """Give each document without an id the next free ``d<n>`` (above every
    numbered id, so a removed document's id is not reused while a higher one
    exists). Duplicate ids -> ValueError."""
    taken: set[str] = set()
    for doc in docs:
        if doc.id is None:
            continue
        if doc.id in taken:
            raise ValueError(f"duplicate document id {doc.id!r}")
        taken.add(doc.id)
    n = max((int(i[1:]) for i in taken if i[1:].isdigit()), default=0)
    for doc in docs:
        if doc.id is None:
            n += 1
            doc.id = f"d{n}"
    return docs
