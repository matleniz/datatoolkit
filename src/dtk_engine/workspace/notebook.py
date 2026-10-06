"""A workspace as a runnable notebook (``.ipynb``) or Python script.

The code replays the pipeline through the notebook door only
(``dtk_engine.api.workspace_frames`` + ``DtkTransformer``), so running it
reproduces the exported frames; the notes (workspace, columns, steps) become
markdown cells (comments in the script). The ``.ipynb`` JSON is built by hand
(nbformat 4.5): no ``nbformat`` dependency (datatoolkit-issues#156).
"""

from __future__ import annotations

import json
import pprint

from dtk_engine.workspace.models import Step, Workspace

_SIDES = {
    "both": ("train = step.fit_transform(train)", "test = step.transform(test)"),
    "train": ("train = step.fit_transform(train)", None),
    "test": (None, "test = step.fit_transform(test)"),
}


def _literal(value: object) -> str:
    """A Python literal of a JSON value (``true`` -> ``True``, ...)."""
    return pprint.pformat(value, width=88, sort_dicts=False)


def _sources(ws: Workspace) -> dict:
    return {
        "datasets": ws.datasets.model_dump(mode="json"),
        "label": ws.label.model_dump(mode="json"),
        "merges": [m.model_dump(mode="json") for m in ws.merges],
    }


def _step_code(step: Step, has_test: bool) -> str | None:
    fit_train, fit_test = _SIDES[step.target]
    if not has_test:
        fit_test = None
    if fit_train is None and fit_test is None:
        return None  # a test-only step without a test set: replay skips it too
    make = f"step = DtkTransformer({step.op!r}, **{_literal(step.params)})"
    return "\n".join([make, *(line for line in (fit_train, fit_test) if line)])


def _intro(ws: Workspace, exported_at: str) -> str:
    lines = [f"# Workspace `{ws.name}`", "", f"Exported {exported_at} by dtk_engine."]
    if ws.notes.workspace:
        lines += ["", ws.notes.workspace]
    if ws.notes.columns:
        lines += ["", "| column | note |", "|---|---|"]
        lines += [
            f"| `{c}` | {t.replace(chr(10), ' ').replace('|', '/')} |"
            for c, t in ws.notes.columns.items()
        ]
    return "\n".join(lines)


def cells(ws: Workspace, exported_at: str) -> list[tuple[str, str]]:
    """``[(kind, text)]`` with kind ``markdown`` | ``code``, in order."""
    has_test = ws.datasets.test is not None
    load = [
        "from dtk_engine import DtkTransformer",
        "from dtk_engine.api import workspace_frames",
        "",
        f"SOURCES = {_literal(_sources(ws))}",
        "train, test = workspace_frames(SOURCES)",
    ]
    out = [("markdown", _intro(ws, exported_at)), ("code", "\n".join(load))]
    for step in ws.steps:
        code = _step_code(step, has_test)
        if code is None:
            continue
        title = f"**{step.id} · {step.op}** ({step.target})"
        out.append(("markdown", f"{title}\n\n{step.note}" if step.note else title))
        out.append(("code", code))
    shapes = "train.shape, (test.shape if test is not None else None)"
    out.append(("code", shapes))
    return out


def notebook_json(ws: Workspace, exported_at: str) -> str:
    """The ``.ipynb`` (nbformat 4.5) text."""
    nb_cells = []
    for i, (kind, text) in enumerate(cells(ws, exported_at)):
        cell: dict = {
            "cell_type": kind,
            "id": f"cell-{i}",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def script_text(ws: Workspace, exported_at: str) -> str:
    """The same code as a plain script; markdown becomes ``#`` comments."""
    parts = []
    for kind, text in cells(ws, exported_at):
        if kind == "markdown":
            parts.append("\n".join(f"# {line}".rstrip() for line in text.splitlines()))
        else:
            parts.append(text)
    parts[-1] = f"print({parts[-1]})"
    return "\n\n".join(parts) + "\n"
