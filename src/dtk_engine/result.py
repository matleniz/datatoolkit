"""Result: the JSON-serializable output of every key."""

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

# Rows of each table shown by _repr_html_ (the Result keeps them all).
HTML_TABLE_ROWS = 10


class Table(BaseModel):
    title: str
    records: list[dict[str, Any]]
    group: str | None = None


class Figure(BaseModel):
    title: str
    plotly: dict[str, Any]
    group: str | None = None


class Result(BaseModel):
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    text: str = ""

    def add_figure(self, title: str, fig: Any, group: str | None = None) -> None:
        """Attach a Plotly figure as JSON; `group` lets a front bucket it in a tab."""
        self.figures.append(
            Figure(title=title, plotly=json.loads(fig.to_json()), group=group)
        )

    def add_table(self, title: str, df: pd.DataFrame, group: str | None = None) -> None:
        """Attach a DataFrame as JSON-safe records; `group` buckets it in a tab."""
        records = json.loads(df.to_json(orient="records"))
        self.tables.append(Table(title=title, records=records, group=group))

    def show(self) -> None:
        """Render in a notebook: print metrics, show each figure."""
        import plotly.io

        for name, value in self.metrics.items():
            print(f"{name}: {value}")
        for figure in self.figures:
            plotly.io.from_json(json.dumps(figure.plotly)).show()

    def _repr_html_(self) -> str:
        """Jupyter display: metrics, head of each table, figures, text."""
        import plotly.io

        parts = []
        if self.metrics:
            metrics = pd.DataFrame(
                {"metric": list(self.metrics), "value": list(self.metrics.values())}
            )
            parts.append(metrics.to_html(index=False))
        for table in self.tables:
            frame = pd.DataFrame.from_records(table.records)
            parts.append(f"<h4>{html.escape(_label(table))}</h4>")
            parts.append(frame.head(HTML_TABLE_ROWS).to_html(index=False))
            if len(frame) > HTML_TABLE_ROWS:
                parts.append(f"<p><i>{HTML_TABLE_ROWS} of {len(frame)} rows</i></p>")
        for i, figure in enumerate(self.figures):
            fig = plotly.io.from_json(json.dumps(figure.plotly))
            parts.append(f"<h4>{html.escape(_label(figure))}</h4>")
            parts.append(
                fig.to_html(
                    full_html=False, include_plotlyjs="cdn" if i == 0 else False
                )
            )
        if self.text:
            parts.append(f"<pre>{html.escape(self.text)}</pre>")
        return "\n".join(parts)


def _label(item: Table | Figure) -> str:
    return f"{item.group} / {item.title}" if item.group else item.title
