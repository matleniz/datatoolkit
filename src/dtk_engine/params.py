"""KeyParams: strict base for every key's Params, plus the column-selector fields.

Column-selector schema convention (read by a front to offer the columns of a
source instead of a free-text field). A field built by ``columns_field`` /
``column_field`` carries, next to its usual JSON Schema:

- ``x-dtk-widget``: ``"columns"`` (multiselect, ``list[str]``) or ``"column"``
  (single choice, ``str``);
- ``x-dtk-source``: name of the sibling param (a ``SourceSpec``) whose columns
  are the options (``contract.source_columns(spec)`` lists them);
- ``x-dtk-dtype``: ``"any"`` or ``"numeric"`` (offer only numeric columns).

An empty ``columns`` list means "every eligible column" (each key caps it).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ColumnDtype = Literal["any", "numeric"]


class KeyParams(BaseModel):
    """Unknown or misspelled params raise instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")


def _hints(widget: str, source: str, dtype: ColumnDtype) -> dict[str, str]:
    return {"x-dtk-widget": widget, "x-dtk-source": source, "x-dtk-dtype": dtype}


def columns_field(
    description: str,
    *,
    source: str = "source",
    dtype: ColumnDtype = "any",
    nullable: bool = False,
    min_length: int | None = None,
) -> Any:
    """A ``list[str]`` param picking columns of the ``source`` param (default: []).

    ``nullable=True`` (default None, type ``list[str] | None``) keeps a key's "null = auto" semantics.
    """
    return Field(
        default=None
        if nullable
        else [],  # pydantic copies it; a factory would hide it from the schema
        min_length=min_length,
        description=description,
        json_schema_extra=_hints("columns", source, dtype),
    )


def column_field(
    default: str | None,
    description: str,
    *,
    source: str = "source",
    dtype: ColumnDtype = "any",
) -> Any:
    """A single-column param (e.g. a target) picking a column of ``source``."""
    return Field(
        default=default,
        description=description,
        json_schema_extra=_hints("column", source, dtype),
    )
