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
# A group id names a run of steps (datatoolkit-issues#249), opaque.
GROUP_ID_PATTERN = r"^g[0-9a-z-]{1,32}$"
GROUP_NAME_MAX = 80
GROUP_INTENT_MAX = 400
DOCUMENT_ID_PATTERN = r"^d[0-9a-z-]{1,32}$"
DOCUMENTS_MAX = 50
MEMORY_ID_PATTERN = r"^m[0-9a-z-]{1,32}$"
MEMORY_ENTRY_MAX = 500  # characters per memory entry
MEMORY_ENTRIES_MAX = 100
MEMORY_CHARS_MAX = 8000  # all entries' text (~2k tokens: injected in a first turn)
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


class Group(_Strict):
    """A contiguous run of steps that answers one intention ("Impute ledd").

    Groups partition the workspace's steps in step order (``Workspace``
    checks it), so versions, replay and export positions keep their meaning.
    """

    id: str = Field(pattern=GROUP_ID_PATTERN)
    name: str = Field(min_length=1, max_length=GROUP_NAME_MAX)
    intent: str = Field(default="", max_length=GROUP_INTENT_MAX)
    step_ids: list[str] = Field(min_length=1)
    origin: Literal["auto", "manual"] = "auto"
    produced_by: str | None = Field(
        default=None, description='Informational label, e.g. "agent", "ui:add_column"'
    )


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


class MemoryEntry(_Strict):
    """One short fact the in-Studio agent keeps across chat sessions
    (datatoolkit-issues#179); ``id`` is filled like document ids."""

    id: str | None = Field(default=None, pattern=MEMORY_ID_PATTERN)
    text: str = Field(min_length=1, max_length=MEMORY_ENTRY_MAX)
    kind: Literal["fact", "decision", "preference", "todo"] = "fact"
    updated_at: str | None = None


class Workspace(_Strict):
    name: str = Field(pattern=NAME_PATTERN)
    datasets: Datasets
    label: LabelJoin = Field(default_factory=LabelJoin)
    # Keep the label column last through replay / export (datatoolkit-issues#196).
    label_last: bool = False
    merges: list[MergeSpec] = Field(default_factory=list)
    variables: list[VariableSpec] = Field(default_factory=list)
    charts: list[ChartSpec] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    groups: list[Group] = Field(
        default_factory=list,
        description="Contiguous runs of steps, in step order; steps without a "
        "group get a singleton auto group, empty groups are dropped",
    )
    notes: WorkspaceNotes = Field(default_factory=WorkspaceNotes)
    documents: list[WorkspaceDocument] = Field(default_factory=list)
    memory: list[MemoryEntry] = Field(default_factory=list)
    # Highest memory (``m``) / document (``d``) number ever handed out, so a
    # forgotten or removed id is never reused (datatoolkit-issues#180).
    id_counters: dict[Literal["m", "d"], int] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def _step_ids(cls, steps: list[Step]) -> list[Step]:
        return fill_step_ids(steps)

    @field_validator("groups", mode="before")
    @classmethod
    def _drop_empty_groups(cls, groups: Any) -> Any:
        if not isinstance(groups, list):
            return groups
        return [g for g in groups if not _is_empty_group(g)]

    @model_validator(mode="after")
    def _normalise_groups(self) -> Workspace:
        self.groups = normalise_groups(self.steps, self.groups)
        return self

    @field_validator("documents")
    @classmethod
    def _document_ids(cls, docs: list[WorkspaceDocument]) -> list[WorkspaceDocument]:
        if len(docs) > DOCUMENTS_MAX:
            raise ValueError(f"at most {DOCUMENTS_MAX} documents per workspace")
        return docs

    @field_validator("memory")
    @classmethod
    def _memory_caps(cls, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        if len(entries) > MEMORY_ENTRIES_MAX:
            raise ValueError(f"at most {MEMORY_ENTRIES_MAX} memory entries")
        chars = sum(len(e.text) for e in entries)
        if chars > MEMORY_CHARS_MAX:
            raise ValueError(f"memory is {chars} characters, over {MEMORY_CHARS_MAX}")
        return entries

    @model_validator(mode="after")
    def _fill_entry_ids(self) -> Workspace:
        self.id_counters = {
            "d": _fill_ids(self.documents, "d", "document", self.id_counters.get("d", 0)),
            "m": _fill_ids(self.memory, "m", "memory", self.id_counters.get("m", 0)),
        }
        return self

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


def _is_empty_group(group: Any) -> bool:
    ids = group.get("step_ids") if isinstance(group, dict) else getattr(group, "step_ids", None)
    return isinstance(ids, list) and not ids


def _singleton_group_id(step: Step, taken: set[str]) -> str:
    """``g<rest of the step id>``, ``-2``, ``-3``... on a clash with ``taken``."""
    base = "g" + (step.id or "")[1:]
    candidate, n = base, 1
    while candidate in taken:
        n += 1
        suffix = f"-{n}"
        candidate = base[: 33 - len(suffix)] + suffix
    return candidate


def _check_group_slice(group: Group, pos: dict[str, int], at: int) -> None:
    """``group`` must be the steps ``at``, ``at + 1``... in order (else ValueError)."""
    for offset, sid in enumerate(group.step_ids):
        if sid not in pos:
            raise ValueError(f"group {group.id!r} names unknown step id {sid!r}")
        if pos[sid] != at + offset:
            raise ValueError(
                f"group {group.id!r} is not a contiguous slice in step order: "
                f"step {sid!r} is at position {pos[sid] + 1}, expected {at + offset + 1}"
            )


def _check_partition(groups: list[Group]) -> dict[str, str]:
    """``{step id: group id}``; duplicate group ids / shared steps -> ValueError."""
    ids: set[str] = set()
    seen: dict[str, str] = {}
    for g in groups:
        if g.id in ids:
            raise ValueError(f"duplicate group id {g.id!r}")
        ids.add(g.id)
        for sid in g.step_ids:
            if sid in seen:
                raise ValueError(
                    f"step {sid!r} is in groups {seen[sid]!r} and {g.id!r}: "
                    "a step belongs to exactly one group"
                )
            seen[sid] = g.id
    return seen


def normalise_groups(steps: list[Step], groups: list[Group]) -> list[Group]:
    """Validate ``groups`` against ``steps`` and wrap uncovered steps.

    Invariants (ValueError naming the group / step): group ids are unique, each
    step is in at most one group, and the groups are contiguous slices in step
    order. Steps in no group get a singleton ``auto`` group at their position
    (name ``"<op> <target>"``, deterministic id that avoids every existing one).
    """
    seen = _check_partition(groups)
    pos = {s.id: i for i, s in enumerate(steps)}
    taken = {g.id for g in groups}
    out: list[Group] = []

    def wrap_until(end: int, nxt: str) -> None:
        for step in steps[len(_flat(out)) : end]:
            if step.id in seen:
                raise ValueError(
                    f"group {nxt!r} is out of step order: step {step.id!r} "
                    f"(group {seen[step.id]!r}) comes before it"
                )
            out.append(_wrap(step, taken))

    for g in groups:
        first = g.step_ids[0]
        if first not in pos:
            raise ValueError(f"group {g.id!r} names unknown step id {first!r}")
        wrap_until(pos[first], g.id)
        _check_group_slice(g, pos, pos[first])
        out.append(g)
    wrap_until(len(steps), "(end)")
    return out


def _flat(groups: list[Group]) -> list[str]:
    return [sid for g in groups for sid in g.step_ids]


def _wrap(step: Step, taken: set[str]) -> Group:
    gid = _singleton_group_id(step, taken)
    taken.add(gid)
    name = f"{step.op} {step.target}"[:GROUP_NAME_MAX]
    return Group(id=gid, name=name, step_ids=[step.id or ""], origin="auto")


def _fill_ids(items: list, prefix: str, what: str, floor: int = 0) -> int:
    """Give each item without an id the next ``<prefix><n>``, above every id in
    ``items`` and above ``floor`` (the highest number ever handed out), so an id
    is never reused. Duplicate ids -> ValueError. Returns the new highest number."""
    taken: set[str] = set()
    for item in items:
        if item.id is None:
            continue
        if item.id in taken:
            raise ValueError(f"duplicate {what} id {item.id!r}")
        taken.add(item.id)
    n = max((int(i[1:]) for i in taken if i[1:].isdigit()), default=0)
    n = max(n, floor)
    for item in items:
        if item.id is None:
            n += 1
            item.id = f"{prefix}{n}"
    return n


def fill_document_ids(docs: list[WorkspaceDocument]) -> list[WorkspaceDocument]:
    _fill_ids(docs, "d", "document")
    return docs


def fill_memory_ids(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    _fill_ids(entries, "m", "memory")
    return entries
