"""Shared vocabulary of the advisor stages: recommendation record, stage order, fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from dtk_engine.ops.profile import hashable
from dtk_engine.transform_registry import get_transform

MODEL_FAMILIES = ("tree", "linear", "distance", "neural")

# Stage of each category: the recommendation table is sorted by it.
STAGES = {
    "rows": 0,
    "leak": 1,
    "drop": 1,
    "sentinels": 2,
    "consistency": 3,
    "type": 4,
    "missing": 5,
    "skew": 6,
    "scaling": 7,
    "encoding": 8,
}

REC_FIELDS = [
    "order",
    "column",
    "category",
    "severity",
    "advice",
    "op",
    "target",
    "params",
]
COLUMN_FIELDS = [
    "column",
    "semantic_type",
    "pct_missing",
    "n_unique",
    "skew",
    "pct_outliers_iqr",
    "onehot_columns",
    "action",
]
ROWS = "(rows)"


@dataclass
class Rec:
    column: str
    category: str
    severity: str  # "info" | "warning"
    advice: str
    op: str
    target: str  # "train" | "test" | "both"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnInfo:
    """Per-column summary row."""

    column: str
    semantic_type: str = ""
    pct_missing: float = 0.0
    n_unique: int = 0
    skew: float | None = None
    pct_outliers_iqr: float | None = None
    onehot_columns: int = 0
    action: str = "keep"


def summarize(df: pd.DataFrame, col: str, semantic: str) -> ColumnInfo:
    n = len(df)
    n_missing = int(df[col].isna().sum())
    return ColumnInfo(
        col,
        semantic_type=semantic,
        pct_missing=round(100 * n_missing / n, 2) if n else 0.0,
        n_unique=int(hashable(df[col]).nunique()),
    )


def apply_rec(rec: Rec, train: pd.DataFrame, test: pd.DataFrame | None):
    """Run a cleaning recommendation on the working copies, like replay would."""
    t = get_transform(rec.op)
    params = t.parse(rec.params)
    if rec.target == "both":
        state = t.fit(train, params)
        train = t.apply(train, params, state)
        if test is not None:
            test = t.apply(test, params, state)
    elif rec.target == "train":
        train = t.fit_apply(train, params)
    elif test is not None:
        test = t.fit_apply(test, params)
    return train, test
