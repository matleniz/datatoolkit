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


class DatasetSource(BaseModel):
    """The current state of a workspace dataset: files + label join + replayed steps."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["dataset"] = "dataset"
    workspace: str = Field(description="Workspace name")
    role: Literal["train", "test"] = "train"
    labeled: bool = Field(
        default=True, description="Join / keep the labels (y) when the role has some"
    )


# Sources that read a file. A workspace's X / y must be one of these (never a
# `dataset`, which would let a workspace point at itself).
FileSourceSpec = Annotated[Union[CsvSource], Field(discriminator="kind")]  # noqa: UP007

# Add new readers' specs to this union (file readers also to FileSourceSpec).
SourceSpec = Annotated[
    Union[CsvSource, DatasetSource],  # noqa: UP007
    Field(discriminator="kind"),
]
