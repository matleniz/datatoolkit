"""Column drops and leaks: ids, target leaks, too-missing, constants, text."""

from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops.advisor.common import Rec
from dtk_engine.ops.missing import DROP_PCT

# |corr| with the target above this: the column is a near-copy of the label.
TARGET_CORR = 0.95


def drop_columns_rec(column: str, category: str, severity: str, advice: str) -> Rec:
    return Rec(
        column,
        category,
        severity,
        advice,
        "drop_columns",
        "both",
        # missing_ok: the column may exist on one side only.
        {"columns": [column], "missing_ok": True},
    )


def drop_missing_rec(column: str, pct: float) -> Rec:
    return drop_columns_rec(
        column,
        "drop",
        "warning",
        f"{pct}% missing (>= {DROP_PCT:.0f}%): drop it, unless missingness itself "
        "is predictive (then impute with an indicator instead)",
    )


def column_drop_rec(
    col: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    semantic: str,
    target: str | None,
) -> Rec | None:
    if test is not None and col not in test.columns:
        return drop_columns_rec(
            col,
            "leak",
            "warning",
            "only in train: not available at prediction time (a label, or "
            "something computed after the fact); drop it",
        )
    if semantic == "id_like":
        return drop_columns_rec(
            col,
            "leak",
            "warning",
            "identifier: unique per row, a model can only memorise it (and it may "
            "encode collection order); drop it",
        )
    if semantic in ("nested", "binary"):
        advice = (
            "lists / dicts per cell (nested JSON or Parquet): no op reads it; drop "
            "it, or flatten it into scalar columns first (pandas.json_normalize, "
            "explode)"
            if semantic == "nested"
            else "raw bytes per cell (geometry WKB, blob): no op reads it; drop it, "
            "or decode it into scalar columns first (e.g. x / y of a geometry)"
        )
        return drop_columns_rec(col, "drop", "warning", advice)
    if target is not None and _names_target(col, target):
        return drop_columns_rec(
            col,
            "leak",
            "warning",
            f"name derives from the target {target!r} (e.g. a group aggregate of "
            "the label): it leaks each row's own label; drop it",
        )
    if target is not None:
        corr = _target_corr(train, col, target)
        if corr is not None and abs(corr) >= TARGET_CORR:
            return drop_columns_rec(
                col,
                "leak",
                "warning",
                f"correlation {corr:.3f} with the target {target!r}: a near-copy of "
                "the label (derived from it?); drop it unless it is truly known "
                "before the outcome",
            )
    pct = round(100 * float(train[col].isna().mean()), 2) if len(train) else 0.0
    if pct >= DROP_PCT:
        return drop_missing_rec(col, pct)
    if semantic == "constant":
        return drop_columns_rec(col, "drop", "info", "constant: carries no information")
    if semantic == "group_id":
        return drop_columns_rec(
            col,
            "leak",
            "warning",
            "entity key repeated over rows: use it as `groups` for GroupKFold "
            "rather than as a feature (rows of one entity in train and "
            "validation inflate the score)",
        )
    if semantic == "text":
        return drop_columns_rec(
            col,
            "drop",
            "info",
            "free text: no text-feature op yet; drop it or engineer features first",
        )
    return None


def _names_target(col: str, target: str) -> bool:
    """``target`` appears as a whole token of ``col`` (``y`` in ``y_mean_by_g``,
    not in ``city``)."""
    token = re.escape(target.lower())
    return re.search(rf"(^|[^0-9a-z]){token}($|[^0-9a-z])", col.lower()) is not None


def _target_corr(train: pd.DataFrame, col: str, target: str) -> float | None:
    x, y = train[col], train[target]
    if not (pdt.is_numeric_dtype(x) and pdt.is_numeric_dtype(y)):
        return None
    if pdt.is_bool_dtype(x) or pdt.is_bool_dtype(y):
        x, y = x.astype(float), y.astype(float)
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 3 or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return None
    return float(both.iloc[:, 0].corr(both.iloc[:, 1]))
