"""Result: the JSON-serializable output of every key."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class Table(BaseModel):
    title: str
    records: list[dict[str, Any]]


class Figure(BaseModel):
    title: str
    plotly: dict[str, Any]


class Result(BaseModel):
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    text: str = ""

    def add_figure(self, title: str, fig: Any) -> None:
        """Attach a Plotly figure as JSON."""
        self.figures.append(Figure(title=title, plotly=json.loads(fig.to_json())))

    def add_table(self, title: str, df: pd.DataFrame) -> None:
        """Attach a DataFrame as JSON-safe records."""
        records = json.loads(df.to_json(orient="records"))
        self.tables.append(Table(title=title, records=records))

    def show(self) -> None:
        """Render in a notebook: print metrics, show each figure."""
        import plotly.io

        for name, value in self.metrics.items():
            print(f"{name}: {value}")
        for figure in self.figures:
            plotly.io.from_json(json.dumps(figure.plotly)).show()
