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

from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace.dataset import replay_label
from dtk_engine.workspace.models import Group, Step, Workspace

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


def _step_code(step: Step, has_test: bool, label: str | None = None) -> str | None:
    fit_train, fit_test = _SIDES[step.target]
    if not has_test:
        fit_test = None
    if fit_train is None and fit_test is None:
        return None  # a test-only step without a test set: replay skips it too
    make = f"step = DtkTransformer({step.op!r}, **{_literal(step.params)})"
    lines = [make, *(line for line in (fit_train, fit_test) if line)]
    if label is not None:  # the replay moves the label back to the end after each step
        lines.append("train = keep_label_last(train, LABEL)")
        lines.append("test = keep_label_last(test, LABEL)")
    return "\n".join(lines)


def _intro(ws: Workspace, exported_at: str) -> str:
    lines = [f"# Workspace `{ws.name}`", "", f"Exported {exported_at} by dtk_engine."]
    if ws.notes.workspace:
        lines += ["", ws.notes.workspace]
    if ws.notes.columns:
        lines += ["", "| column | note |", "|---|---|"]
        lines += [
            f"| `{c}` | {_one_line(t).replace('|', '/')} |"
            for c, t in ws.notes.columns.items()
        ]
    return "\n".join(lines)


def _step_block(step: Step, code: str) -> str:
    """One step's code under its ``# s12 · op target`` comment (note kept as comments)."""
    head = [f"# {step.id} · {_one_line(step.op)} {step.target}"]
    head += [f"# {line}".rstrip() for line in (step.note or "").splitlines()]
    return "\n".join([*head, code])


def _ids(step_ids: list[str]) -> str:
    return ", ".join(step_ids) if len(step_ids) <= 3 else f"{step_ids[0]} … {step_ids[-1]}"


def _group_sections(ws: Workspace) -> list[tuple[Group, str]]:
    """``[(group, code)]`` in order; the code of a group holds its steps in order.

    A group whose steps all have no code (test-only steps without a test set,
    skipped by replay too) is left out.
    """
    has_test = ws.datasets.test is not None
    label = replay_label(ws)
    by_id = {s.id: s for s in ws.steps}
    out = []
    for group in ws.groups:
        blocks = []
        for sid in group.step_ids:
            step = by_id[sid]
            code = _step_code(step, has_test, label)
            if code is not None:
                blocks.append(_step_block(step, code))
        if blocks:
            out.append((group, "\n\n".join(blocks)))
    return out


def _one_line(text: str) -> str:
    """Whitespace (any line break included) collapsed: a newline would end a ``#`` comment."""
    return " ".join(text.split())


def _name(group: Group) -> str:
    return _one_line(group.name)


def _group_title(group: Group) -> str:
    lines = [f"## {_name(group)}"]
    if group.intent:
        lines.append(group.intent)
    lines.append(f"Steps: {_ids(group.step_ids)}")
    return "\n".join(lines)


def _load_code(ws: Workspace) -> str:
    label = replay_label(ws)
    return "\n".join(
        [
            "from dtk_engine import DtkTransformer",
            "from dtk_engine.api import workspace_frames"
            + (", keep_label_last" if label is not None else ""),
            "",
            f"SOURCES = {_literal(_sources(ws))}",
            *([f"LABEL = {label!r}"] if label is not None else []),
            "train, test = workspace_frames(SOURCES)",
        ]
    )


_SHAPES = "train.shape, (test.shape if test is not None else None)"


def cells(ws: Workspace, exported_at: str) -> list[tuple[str, str]]:
    """``[(kind, text)]`` with kind ``markdown`` | ``code``, in order.

    Intro and loading cells, then a markdown + code pair per step group.
    """
    out = [("markdown", _intro(ws, exported_at)), ("code", _load_code(ws))]
    for group, code in _group_sections(ws):
        out.append(("markdown", _group_title(group)))
        out.append(("code", code))
    out.append(("code", _SHAPES))
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


# Ops that drop, duplicate or reorder rows: a sklearn Pipeline cannot keep X and y aligned
# through them, so PIPELINE leaves them out (with a comment).
_ROW_OPS = frozenset(
    {"filter_rows", "sample_rows", "sort_rows", "drop_duplicates", "drop_missing_target"}
)


def _pipeline_entry(i: int, step: Step) -> list[str]:
    """Lines of the ``Pipeline`` list for one step: a ``# skipped`` comment, or the
    ``DtkTransformer`` (with a caveat comment when it reads the label)."""
    what = f"{step.id} {_one_line(step.op)} ({step.target})"
    if step.target != "both":
        return [f"    # skipped: {what}: {step.target}-only"]
    if step.op in _ROW_OPS:
        return [
            f"    # skipped: {what}: changes the rows, which a Pipeline cannot keep aligned",
            "    #   with y; apply it to your data yourself before fitting.",
        ]
    lines = []
    if get_transform(step.op).needs_target:
        col = step.params.get("target")
        lines += [
            f"    # {step.id} reads the label column {col!r}: it must be in the frame passed to",
            "    #   fit (and then stays in the output); a separate X without it fails.",
        ]
    make = f"DtkTransformer({step.op!r}, **{_literal(step.params)})"
    return [*lines, f"    ({f'{i}_{step.op}'!r}, {make}),"]


def _pipeline_code(ws: Workspace) -> str:
    """The "both" steps as an unfitted sklearn ``Pipeline``, standalone: built from the
    steps' literals, no stored workspace needed. Same names / steps as
    ``dtk_engine.workspace_pipeline``, except that row-changing steps are left out;
    every step not in it is listed in a comment."""
    label = replay_label(ws)
    entries = [line for i, s in enumerate(ws.steps) for line in _pipeline_entry(i, s)]
    if not any(not line.lstrip().startswith("#") for line in entries):
        entries.append("    ('passthrough', 'passthrough'),")
    head = [
        "# --- sklearn pipeline (optional) ---",
        "# The steps applied to train AND test, as an unfitted Pipeline. It starts from the",
        "# frame workspace_frames(SOURCES) returns (merges and label joins are not in it),",
        "# so fit it on a frame of that shape: PIPELINE.fit(train), PIPELINE.transform(new).",
        "# Train-only / test-only and row-changing steps are skipped (listed below).",
    ]
    if ws.merges:
        head.append(f"# This workspace has {len(ws.merges)} merge(s): run them on your data first.")
    if label is not None:
        head += [
            f"# label_last: the replay moves {label!r} to the end after each step; PIPELINE does",
            "# not, so the column order can differ: keep_label_last(frame, LABEL) restores it.",
        ]
    return "\n".join(
        [*head, "from sklearn.pipeline import Pipeline", "", "PIPELINE = Pipeline([", *entries, "])"]
    )


def script_text(ws: Workspace, exported_at: str, pipeline: bool = False) -> str:
    """The same code as a plain script; notes and group headers become ``#`` comments.
    ``pipeline`` appends the optional trailing sklearn ``Pipeline`` section."""

    def comment(text: str) -> str:
        return "\n".join(f"# {line}".rstrip() for line in text.splitlines())

    parts = [comment(_intro(ws, exported_at)), _load_code(ws)]
    for group, code in _group_sections(ws):
        header = [f"# --- {_name(group)} ---"]
        if group.intent:
            header.append(comment(group.intent))
        parts.append("\n".join([*header, code]))
    parts.append(f"print({_SHAPES})")
    if pipeline:
        parts.append(_pipeline_code(ws))
    return "\n\n".join(parts) + "\n"
