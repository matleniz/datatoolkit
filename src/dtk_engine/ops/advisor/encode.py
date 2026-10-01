"""Encoding recommendations: datetime parts, booleans, ordinal / one-hot."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops._util import py as _py
from dtk_engine.ops.advisor.common import ColumnInfo, Rec
from dtk_engine.ops.advisor.drops import drop_columns_rec

# Course "Categorical Encoding": min_frequency=10 folds rare categories; past a
# few dozen categories one-hot is mostly zeros.
MIN_FREQUENCY = 10
MIN_FREQUENCY_FROM_UNIQUE = 10
HIGH_CARDINALITY = 50

# Known orders of ordinal vocabularies (lower-case), lowest first.
ORDINAL_VOCABULARIES = (
    ("low", "medium", "high"),
    ("low", "mid", "high"),
    ("very low", "low", "medium", "high", "very high"),
    ("xs", "s", "m", "l", "xl", "xxl"),
    ("small", "medium", "large"),
    ("poor", "fair", "good", "very good", "excellent"),
    ("bad", "average", "good"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("cold", "warm", "hot"),
    ("first", "second", "third"),
    ("primary", "secondary", "tertiary"),
    (
        "strongly disagree",
        "disagree",
        "neutral",
        "agree",
        "strongly agree",
    ),
)


def _ordinal_order(values: list) -> list | None:
    """The values in a known ordinal order, or None when they match no vocabulary."""
    by_key: dict[str, Any] = {}
    for v in values:
        key = " ".join(str(v).split()).lower()
        if key in by_key:
            return None  # two spellings of one level: clean the variants first
        by_key[key] = v
    for vocab in ORDINAL_VOCABULARIES:
        if len(by_key) >= 2 and set(by_key) <= set(vocab):
            return [by_key[k] for k in vocab if k in by_key]
    return None


def _datetime_recs(col: str) -> list[Rec]:
    return [
        Rec(
            col, "encoding", "info",
            "date/time: models need numbers, extract calendar parts "
            "(then drop the raw column)",
            "datetime_parts", "both", {"column": col},
        ),
        drop_columns_rec(col, "encoding", "info", "raw date/time replaced by its parts"),
    ]  # fmt: skip


def _onehot_rec(col, counts, n_unique, unseen, unseen_note, family, info) -> Rec:
    params: dict[str, Any] = {"columns": [col]}
    n_rare = int((counts < MIN_FREQUENCY).sum())
    produced = n_unique
    notes = []
    if n_unique == 2 and not unseen:
        # With unseen test values, drop_first would alias them to the dropped level.
        params["drop_first"] = True
        produced = 1
        notes.append("binary: one 0/1 column (drop_first)")
    elif n_rare and n_unique > MIN_FREQUENCY_FROM_UNIQUE and n_rare < len(counts):
        params["min_frequency"] = MIN_FREQUENCY
        produced = n_unique - n_rare + 1
        notes.append(
            f"{n_rare} categories seen < {MIN_FREQUENCY} times share one "
            f"{col}_infrequent column (min_frequency={MIN_FREQUENCY})"
        )
    info.onehot_columns = produced
    high = n_unique > HIGH_CARDINALITY
    advice = f"nominal: one-hot costs {produced} column{'s' if produced > 1 else ''}"
    if notes:
        advice += "; " + "; ".join(notes)
    if high:
        advice += (
            f"; high cardinality ({n_unique} categories): the matrix is mostly "
            "zeros, consider grouping or dropping"
        )
    if family == "linear" and n_unique > 2:
        advice += "; drop_first only matters for an unregularised linear model"
    if unseen:
        advice += unseen_note + " (unseen -> all-zero row)"
    severity = "warning" if high else "info"
    return Rec(col, "encoding", severity, advice, "onehot", "both", params)


def _categorical_recs(col, train, test, family, missing_rec, info) -> list[Rec]:
    counts = train[col].value_counts()
    fills = missing_rec is not None and missing_rec.params["strategy"] == "constant"
    n_unique = len(counts) + (1 if fills else 0)
    unseen = []
    if test is not None and col in test.columns:
        unseen = sorted(set(test[col].dropna().unique()) - set(counts.index), key=str)
    note = (
        f"; {len(unseen)} test categories unseen in train (e.g. {unseen[:3]})"
        if unseen
        else ""
    )
    order = _ordinal_order(list(counts.index))
    if order is not None:
        info.onehot_columns = 1
        return [Rec(
            col, "encoding", "info",
            f"ordinal hint: levels {order} have a natural order: code them "
            f"0..{len(order) - 1} (check the order; unknown -> -1){note}",
            "ordinal", "both", {"categories": {col: order}},
        )]  # fmt: skip
    if family == "tree" and n_unique > HIGH_CARDINALITY:
        info.onehot_columns = 1
        return [Rec(
            col, "encoding", "warning",
            f"high cardinality ({n_unique} categories) for a tree model: an "
            "arbitrary integer code (by frequency) avoids a mostly-zero one-hot "
            f"matrix; trees can carve the codes apart{note}",
            "ordinal", "both", {"categories": {col: [_py(v) for v in counts.index]}},
        )]  # fmt: skip
    return [_onehot_rec(col, counts, n_unique, unseen, note, family, info)]


def encoding_recs(
    col: str,
    semantic: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    family: str | None,
    missing_rec: Rec | None,
    info: ColumnInfo,
) -> list[Rec]:
    if semantic == "datetime":
        return _datetime_recs(col)
    if semantic == "boolean" and pdt.is_bool_dtype(train[col]):
        return [Rec(
            col, "encoding", "info", "boolean: cast to 0/1",
            "cast", "both", {"dtypes": {col: "int64"}},
        )]  # fmt: skip
    if semantic == "categorical":
        return _categorical_recs(col, train, test, family, missing_rec, info)
    return []
