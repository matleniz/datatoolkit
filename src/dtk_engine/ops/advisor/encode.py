"""Encoding recommendations: datetime parts, booleans, ordinal / one-hot."""

from __future__ import annotations

from functools import cached_property
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


class _Col:
    """One column's inputs to the encoding rules; the categorical facts are lazy."""

    def __init__(self, col, semantic, train, test, family, missing_rec, info):
        self.col, self.semantic, self.train, self.test = col, semantic, train, test
        self.family, self.missing_rec, self.info = family, missing_rec, info

    @cached_property
    def counts(self) -> pd.Series:
        return self.train[self.col].value_counts()

    @cached_property
    def n_unique(self) -> int:
        fills = self.missing_rec is not None and (
            self.missing_rec.params["strategy"] == "constant"
        )
        return len(self.counts) + (1 if fills else 0)

    @cached_property
    def unseen(self) -> list:
        if self.test is None or self.col not in self.test.columns:
            return []
        seen = set(self.counts.index)
        return sorted(set(self.test[self.col].dropna().unique()) - seen, key=str)

    @cached_property
    def unseen_note(self) -> str:
        if not self.unseen:
            return ""
        return (
            f"; {len(self.unseen)} test categories unseen in train "
            f"(e.g. {self.unseen[:3]})"
        )

    @cached_property
    def order(self) -> list | None:
        return _ordinal_order(list(self.counts.index))

    @property
    def high(self) -> bool:
        return self.n_unique > HIGH_CARDINALITY


def _datetime_recs(c: _Col) -> list[Rec]:
    return [
        Rec(
            c.col,
            "encoding",
            "info",
            "date/time: models need numbers, extract calendar parts "
            "(then drop the raw column)",
            "datetime_parts",
            "both",
            {"column": c.col},
        ),
        drop_columns_rec(
            c.col, "encoding", "info", "raw date/time replaced by its parts"
        ),
    ]


def _bool_cast_recs(c: _Col) -> list[Rec]:
    return [
        Rec(
            c.col,
            "encoding",
            "info",
            "boolean: cast to 0/1",
            "cast",
            "both",
            {"dtypes": {c.col: "int64"}},
        )
    ]


def _ordinal_hint_recs(c: _Col) -> list[Rec]:
    order = c.order
    c.info.onehot_columns = 1
    return [
        Rec(
            c.col,
            "encoding",
            "info",
            f"ordinal hint: levels {order} have a natural order: code them "
            f"0..{len(order) - 1} (check the order; unknown -> -1){c.unseen_note}",
            "ordinal",
            "both",
            {"categories": {c.col: order}},
        )
    ]


def _tree_ordinal_recs(c: _Col) -> list[Rec]:
    by_freq = [_py(v) for v in c.counts.index]
    c.info.onehot_columns = 1
    return [
        Rec(
            c.col,
            "encoding",
            "warning",
            f"high cardinality ({c.n_unique} categories) for a tree model: an "
            "arbitrary integer code (by frequency) avoids a mostly-zero one-hot "
            f"matrix; trees can carve the codes apart{c.unseen_note}",
            "ordinal",
            "both",
            {"categories": {c.col: by_freq}},
        )
    ]


def _onehot_recs(c: _Col) -> list[Rec]:
    n_unique, unseen, counts = c.n_unique, c.unseen, c.counts
    params: dict[str, Any] = {"columns": [c.col]}
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
            f"{c.col}_infrequent column (min_frequency={MIN_FREQUENCY})"
        )
    c.info.onehot_columns = produced
    severity = "warning" if c.high else "info"
    advice = f"nominal: one-hot costs {produced} column{'s' if produced > 1 else ''}"
    if notes:
        advice += "; " + "; ".join(notes)
    if c.high:
        advice += (
            f"; high cardinality ({n_unique} categories): the matrix is mostly "
            "zeros, consider grouping or dropping"
        )
    if c.family == "linear" and n_unique > 2:
        advice += "; drop_first only matters for an unregularised linear model"
    if unseen:
        advice += c.unseen_note + " (unseen -> all-zero row)"
    return [Rec(c.col, "encoding", severity, advice, "onehot", "both", params)]


# (predicate, recs builder), in priority order: the first predicate that holds wins;
# no match = no encoding advice.
_ENCODING_RULES = (
    (lambda c: c.semantic == "datetime", _datetime_recs),
    (
        lambda c: c.semantic == "boolean" and pdt.is_bool_dtype(c.train[c.col]),
        _bool_cast_recs,
    ),
    (lambda c: c.semantic == "categorical" and c.order is not None, _ordinal_hint_recs),
    (
        lambda c: c.semantic == "categorical" and c.family == "tree" and c.high,
        _tree_ordinal_recs,
    ),
    (lambda c: c.semantic == "categorical", _onehot_recs),
)


def encoding_recs(
    col: str,
    semantic: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    family: str | None,
    missing_rec: Rec | None,
    info: ColumnInfo,
) -> list[Rec]:
    c = _Col(col, semantic, train, test, family, missing_rec, info)
    return next((build(c) for when, build in _ENCODING_RULES if when(c)), [])
