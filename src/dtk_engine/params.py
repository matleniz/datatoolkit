"""KeyParams: strict base for every key's Params, plus the column-selector fields.

Column-selector schema convention (read by a front to offer the columns of a
source instead of a free-text field). A field built by ``columns_field`` /
``column_field`` carries, next to its usual JSON Schema:

- ``x-dtk-widget``: ``"columns"`` (multiselect, ``list[str]``) or ``"column"``
  (single choice, ``str``);
- ``x-dtk-source``: name of the sibling param (a ``SourceSpec``) whose columns
  are the options (``contract.source_columns(spec)`` lists them), or the special
  value ``"step"`` (transform ops, which have no sibling ``SourceSpec``): "the
  frame this step applies to", i.e. the workspace ``dataset`` source of the
  step's role (train for ``train`` / ``both``, test for ``test``), in its state
  before the step;
- ``x-dtk-dtype``: ``"any"`` or ``"numeric"`` (offer only numeric columns).

An empty ``columns`` list means "every eligible column" (each key caps it).

Any param may also carry ``x-dtk-when``: ``{sibling param: value}``, i.e. the
param only matters (a front shows it) when every listed sibling has that value,
e.g. ``impute.expr`` -> ``{"strategy": "formula"}``.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.sources.spec import CsvSource, SourceSpec

ColumnDtype = Literal["any", "numeric"]


class KeyParams(BaseModel):
    """Unknown or misspelled params raise instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")


class SourceParams(KeyParams):
    """Params of a key analysing one ``source`` (default: the demo train CSV)."""

    source: SourceSpec = CsvSource(path=TRAIN_CSV)


def _hints(widget: str, source: str, dtype: ColumnDtype) -> dict[str, str]:
    return {"x-dtk-widget": widget, "x-dtk-source": source, "x-dtk-dtype": dtype}


def columns_field(
    description: str,
    *,
    source: str = "source",
    dtype: ColumnDtype = "any",
    nullable: bool = False,
    required: bool = False,
    min_length: int | None = None,
    max_length: int | None = None,
) -> Any:
    """A ``list[str]`` param picking columns of the ``source`` param (default: []).

    ``nullable=True`` (default None, type ``list[str] | None``) keeps a key's "null = auto" semantics.
    ``required=True`` drops the default (the param must be given).
    """
    default = ... if required else None if nullable else []  # pydantic copies []
    return Field(
        default=default,
        min_length=min_length,
        max_length=max_length,
        description=description,
        json_schema_extra=_hints("columns", source, dtype),
    )


def column_field(
    default: Any,
    description: str,
    *,
    source: str = "source",
    dtype: ColumnDtype = "any",
) -> Any:
    """A single-column param (e.g. a target) picking a column of ``source``.

    ``default=...`` makes the param required."""
    return Field(
        default=default,
        description=description,
        json_schema_extra=_hints("column", source, dtype),
    )
