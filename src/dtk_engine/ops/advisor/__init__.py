"""Preprocessing advisor: per-column recommendations, each one a concrete step.

Every recommendation names a registered transform op, the params to give it and
the target (train / test / both), so a front can "apply" it as a workspace step.
Recommendations are ordered like a preprocessing pipeline: row fixes, drops,
cleaning (sentinels, variants, types), imputation, skew, scaling, encoding.

The cleaning recommendations are applied (with the real ops) to a copy of the
frames before the later checks, so e.g. a column whose test side holds
"unknown" among numbers is seen as numeric, and its missing rate counts the
replaced sentinels. Nothing here mutates the caller's frames.
"""

from __future__ import annotations

import pandas as pd

from dtk_engine.cache import memoize
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.advisor.cleaning import sentinel_rec, type_rec, variant_rec
from dtk_engine.ops.advisor.common import (
    COLUMN_FIELDS,
    MODEL_FAMILIES,
    REC_FIELDS,
    STAGES,
    Rec,
    apply_rec,
    summarize,
)
from dtk_engine.ops.advisor.drops import (
    column_drop_rec,
    drop_columns_rec,
    drop_missing_rec,
)
from dtk_engine.ops.advisor.encode import encoding_recs
from dtk_engine.ops.advisor.missing import missing_rec
from dtk_engine.ops.advisor.numeric import numeric_recs
from dtk_engine.ops.advisor.rows import row_recs
from dtk_engine.ops.consistency import text_columns
from dtk_engine.ops.missing import DROP_PCT
from dtk_engine.ops.profile import semantic_types

__all__ = [
    "COLUMN_FIELDS",
    "MODEL_FAMILIES",
    "REC_FIELDS",
    "Rec",
    "advise",
    "as_steps",
]


def _drop_phase(train, test, target, features, semantic, recs, infos) -> list[str]:
    """Drops first: a dropped column gets no other advice. Returns the kept columns."""
    kept = []
    for col in features:
        rec = column_drop_rec(col, train, test, semantic[col], target)
        if rec is None:
            kept.append(col)
        else:
            recs.append(rec)
            infos[col].action = "drop" if rec.op == "drop_columns" else rec.op
    if test is not None:
        for col in (str(c) for c in test.columns if c not in train.columns):
            recs.append(
                Rec(
                    col,
                    "drop",
                    "warning",
                    "only in test: the model never saw it; drop it on test",
                    "drop_columns",
                    "test",
                    {"columns": [col]},
                )
            )
    return kept


def _cleaning_phase(
    train, test, kept, recs
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cleaning, applied to working copies so later checks see clean columns."""
    work_train, work_test = train, test
    frames = [train] if test is None else [train, test]
    text = set(text_columns(train))
    for col in kept:
        cleaning = [sentinel_rec(col, frames)]
        if col in text:
            cleaning.append(variant_rec(col, frames))
        for rec in filter(None, cleaning):
            recs.append(rec)
            work_train, work_test = apply_rec(rec, work_train, work_test)
        rec = type_rec(col, [f for f in (work_train, work_test) if f is not None])
        if rec is not None:
            recs.append(rec)
            work_train, work_test = apply_rec(rec, work_train, work_test)
    return work_train, work_test


def _column_phase(
    train, test, kept, semantic, family, recs, infos
) -> dict[str, Rec | None]:
    """Per-column drops once cleaned, missing and numeric advice. Returns the live
    columns with their missing rec."""
    live: dict[str, Rec | None] = {}
    for col in kept:
        info = infos[col] = summarize(train, col, semantic[col])
        if semantic[col] == "constant":
            recs.append(drop_columns_rec(col, "drop", "info", "constant once cleaned"))
        elif info.pct_missing >= DROP_PCT:
            recs.append(drop_missing_rec(col, info.pct_missing))
        else:
            live[col] = missing = missing_rec(col, semantic[col], train, test, info)
            if missing is not None:
                recs.append(missing)
            if semantic[col] == "numeric":
                recs += numeric_recs(col, train, test, family, info)
            continue
        info.action = "drop"
    return live


def _encoding_phase(train, test, live, semantic, family, recs, infos) -> None:
    for col, missing in live.items():
        info = infos[col]
        recs += encoding_recs(col, semantic[col], train, test, family, missing, info)
        ops = [r.op for r in recs if r.column == col and r.op != "drop_columns"]
        info.action = ", ".join(ops) if ops else "keep"


@memoize(max_entries=16)
def advise(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    model_family: str | None = None,
    target: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(recommendations, per-column summary) for ``train`` (and ``test``).

    Recommendations: one row per suggested step, fields ``REC_FIELDS``; ``op`` /
    ``target`` / ``params`` form a workspace step, ``order`` the suggested
    application order. ``model_family`` (tree | linear | distance | neural)
    decides scaling, skew and high-cardinality encoding; None = advice for all.
    An unknown ``model_family`` or ``target`` raises KeyParamsError.
    """
    if model_family is not None and model_family not in MODEL_FAMILIES:
        raise KeyParamsError(
            f"unknown model_family {model_family!r} (known: {list(MODEL_FAMILIES)})"
        )
    if target is not None and target not in train.columns:
        raise KeyParamsError(f"target {target!r} is not a column of the source")

    recs = row_recs(train, target)
    semantic = semantic_types(train)
    features = [str(c) for c in train.columns if c != target]
    infos = {c: summarize(train, c, semantic[c]) for c in features}

    kept = _drop_phase(train, test, target, features, semantic, recs, infos)
    work_train, work_test = _cleaning_phase(train, test, kept, recs)
    semantic = semantic_types(work_train[kept])
    live = _column_phase(
        work_train, work_test, kept, semantic, model_family, recs, infos
    )
    _encoding_phase(work_train, work_test, live, semantic, model_family, recs, infos)

    rows = sorted(enumerate(recs), key=lambda ir: (STAGES[ir[1].category], ir[0]))
    table = pd.DataFrame(
        [{"order": i + 1, **vars(r)} for i, (_, r) in enumerate(rows)],
        columns=REC_FIELDS,
    )
    columns = pd.DataFrame([vars(infos[c]) for c in features], columns=COLUMN_FIELDS)
    return table, columns


def as_steps(recommendations: pd.DataFrame) -> list[dict]:
    """The recommendations as workspace step dicts ``{op, target, params}``, in order."""
    return [
        {"op": r.op, "target": r.target, "params": r.params}
        for r in recommendations.itertuples()
    ]
