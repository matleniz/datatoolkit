"""SourceSpec: pydantic union of source descriptions, discriminated on `kind`."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class CsvSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["csv"] = "csv"
    path: str = Field(description="Local path to the CSV file")
    sep: str = Field(default="auto", description='Separator; "auto" sniffs it')
    encoding: str = "utf-8"
    decimal: str = Field(default=".", description="Decimal mark")
    header: int | None = Field(
        default=0, description="Row number of the header; null = no header"
    )
    na_values: list[str] | None = Field(
        default=None, description="Extra strings read as missing"
    )
    dtype: dict[str, str] | None = Field(
        default=None, description="Column -> dtype (e.g. {'zip': 'str'})"
    )
    parse_dates: list[str] | None = Field(
        default=None, description="Columns parsed as datetimes"
    )
    usecols: list[str] | None = Field(default=None, description="Columns to keep")


class DatasetSource(BaseModel):
    """The current state of a workspace dataset: files + label join + replayed steps."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dataset"] = "dataset"
    workspace: str = Field(description="Workspace name")
    role: Literal["train", "test"] = "train"
    labeled: bool = Field(
        default=True, description="Join / keep the labels (y) when the role has some"
    )


class ParquetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["parquet"] = "parquet"
    path: str = Field(description="Parquet file or partitioned directory")
    columns: list[str] | None = Field(default=None, description="Columns to keep")
    filters: list[tuple[str, str, Any]] | None = Field(
        default=None,
        description="Row filters [column, op, value], ANDed; op in ==, !=, <, <=, >, >=, in, not in",
    )


class ExcelSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["excel"] = "excel"
    path: str = Field(description="Local path to the .xlsx file")
    sheet: str | int = Field(default=0, description="Sheet name or 0-based index")
    header: int | None = Field(
        default=0,
        description="0-based row of the header; null = no header. Sheets with a "
        "title block above the header need it (file_inspect suggests one)",
    )
    usecols: list[str] | None = Field(default=None, description="Columns to keep")


class JsonSource(BaseModel):
    """JSON or JSON lines. Nested objects are flattened (``a_b``); lists stay as-is."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["json"] = "json"
    path: str = Field(description="Local path to the JSON / JSONL file")
    lines: bool = Field(default=False, description="One JSON record per line (jsonl)")
    encoding: str = Field(
        default="utf-8-sig",
        description="Text encoding; utf-8-sig also reads plain utf-8",
    )
    record_path: str | None = Field(
        default=None, description="Dotted path to the list of records, e.g. data.items"
    )


class SqlSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["sql"] = "sql"
    url_env: str = Field(
        description="NAME of the environment variable holding the SQLAlchemy URL "
        "(never the URL itself)"
    )
    query: str = Field(description="SQL run on the server")


# Sources that read a file. A workspace's X / y must be one of these (never a
# `dataset`, which would let a workspace point at itself).
FileSourceSpec = Annotated[
    Union[CsvSource, ParquetSource, ExcelSource, JsonSource],  # noqa: UP007
    Field(discriminator="kind"),
]

# Add new readers' specs to this union (file readers also to FileSourceSpec).
SourceSpec = Annotated[
    CsvSource | DatasetSource | ParquetSource | ExcelSource | JsonSource | SqlSource,
    Field(discriminator="kind"),
]
