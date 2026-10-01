"""Result: the JSON-serializable output of every key."""

from __future__ import annotations

import contextlib
import html
import json
import threading
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from dtk_engine.ops.profile import as_text, object_kind


class _PlotlyLock(contextlib.ContextDecorator):
    """Process-wide re-entrant lock protecting Plotly figure construction & serialization.

    Plotly's figure construction and template cascade are not thread-safe:
    concurrent callers share global template objects and validator caches,
    causing intermittent ``ValueError: Invalid value`` in ``_index_is``.
    """

    _lock = threading.RLock()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


plotly_lock = _PlotlyLock()

# Rows of each table shown by _repr_html_ (the Result keeps them all).
HTML_TABLE_ROWS = 10


# Table.kind values. "steps": each record is a workspace step (``op``,
# ``target``, ``params``, plus key-specific fields), ready for a front to apply.
TableKind = Literal["steps"]


class Table(BaseModel):
    title: str
    records: list[dict[str, Any]]
    group: str | None = None
    kind: TableKind | None = None


class Figure(BaseModel):
    title: str
    plotly: dict[str, Any]
    group: str | None = None
    main: bool = False


class Result(BaseModel):
    headline: str = ""
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    tables: list[Table] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    text: str = ""

    @model_validator(mode="after")
    def _check_single_main_figure(self) -> Result:
        main_count = sum(1 for f in self.figures if f.main)
        if main_count > 1:
            raise ValueError(f"At most one figure can have main=True, got {main_count}")
        return self

    def add_figure(
        self,
        title: str,
        fig: Any | Callable[[], Any],
        group: str | None = None,
        main: bool = False,
    ) -> None:
        """Attach a Plotly figure as JSON; `group` lets a front bucket it in a tab.

        ``fig`` may be a Plotly Figure, a dict, or a callable returning either.
        Construction (if callable) and serialization are protected by ``plotly_lock``.
        At most one figure per Result can have ``main=True``.
        """
        with plotly_lock:
            if callable(fig):
                fig = fig()
            data = fig if isinstance(fig, dict) else json.loads(fig.to_json())
            if main:
                for existing in self.figures:
                    existing.main = False
            self.figures.append(
                Figure(title=title, plotly=data, group=group, main=main)
            )

    def add_table(
        self,
        title: str,
        df: pd.DataFrame,
        group: str | None = None,
        kind: TableKind | None = None,
    ) -> None:
        """Attach a DataFrame as JSON-safe records; `group` buckets it in a tab,
        `kind` tells a front how to read the rows (see ``TableKind``)."""
        # Bytes cells (WKB geometry, blobs) are not JSON: shown as their repr.
        binary = [
            i for i in range(df.shape[1]) if object_kind(df.iloc[:, i]) == "binary"
        ]
        if binary:
            df = df.copy()
            for i in binary:
                df.isetitem(i, as_text(df.iloc[:, i]).where(df.iloc[:, i].notna()))
        records = json.loads(df.to_json(orient="records"))
        self.tables.append(Table(title=title, records=records, group=group, kind=kind))

    def show(self) -> None:
        """Render in a notebook: print metrics, show each figure."""
        import plotly.io

        for name, value in self.metrics.items():
            print(f"{name}: {value}")
        for figure in self.figures:
            with plotly_lock:
                fig = plotly.io.from_json(json.dumps(figure.plotly))
            fig.show()

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
            with plotly_lock:
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
