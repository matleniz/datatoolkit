"""SourceSpec: pydantic union of source descriptions, discriminated on `kind`."""

from __future__ import annotations

from typing import Annotated, Literal, Union

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


# Add new readers' specs to this union.
SourceSpec = Annotated[Union[CsvSource], Field(discriminator="kind")]  # noqa: UP007
