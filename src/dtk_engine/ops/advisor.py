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

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops.consistency import text_columns, variants
from dtk_engine.ops.duplicates import exact_duplicates
from dtk_engine.ops.missing import (
    DATE_SENTINELS,
    DROP_PCT,
    NUMERIC_SENTINELS,
    REVIEW_PCT,
    STRING_SENTINELS,
    sentinel_counts,
)
from dtk_engine.ops.outliers import univariate_outliers
from dtk_engine.ops.profile import pct_numeric_parsable, semantic_types
from dtk_engine.ops.transforms.impute import INDICATOR_SUFFIX
from dtk_engine.transform_registry import get_transform

MODEL_FAMILIES = ("tree", "linear", "distance", "neural")
# Course "Scaling and Normalization", who needs it: trees split on order (no);
# regularised linear models (penalty on raw coefficients), distance-based
# models and neural networks (yes).
SCALING_NEEDED = {"tree": False, "linear": True, "distance": True, "neural": True}
# Right skew above this (and no negative value) -> log1p.
SKEW_THRESHOLD = 1.0
# More IQR outliers than this (% of non-null values, after log1p) -> robust scaler.
ROBUST_OUTLIER_PCT = 1.0
# Course "Categorical Encoding": min_frequency=10 folds rare categories; past a
# few dozen categories one-hot is mostly zeros.
MIN_FREQUENCY = 10
MIN_FREQUENCY_FROM_UNIQUE = 10
HIGH_CARDINALITY = 50
# |corr| with the target above this: the column is a near-copy of the label.
TARGET_CORR = 0.95
# Hits of the "implausible 0" sentinel heuristic are only reported, not replaced.
UNSAFE_SENTINELS = frozenset({"0"})

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
class _Col:
    """Per-column summary row."""

    column: str
    semantic_type: str = ""
    pct_missing: float = 0.0
    n_unique: int = 0
    skew: float | None = None
    pct_outliers_iqr: float | None = None
    onehot_columns: int = 0
    action: str = "keep"


def _summary(df: pd.DataFrame, col: str, semantic: str) -> _Col:
    n = len(df)
    n_missing = int(df[col].isna().sum())
    return _Col(
        col,
        semantic_type=semantic,
        pct_missing=round(100 * n_missing / n, 2) if n else 0.0,
        n_unique=int(df[col].nunique()),
    )


def _py(value):
    return value.item() if isinstance(value, np.generic) else value


def _apply(rec: Rec, train: pd.DataFrame, test: pd.DataFrame | None):
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


# --- rows ---------------------------------------------------------------------


def _row_recs(train: pd.DataFrame, target: str | None) -> list[Rec]:
    recs = []
    if target is not None and train[target].isna().any():
        n = int(train[target].isna().sum())
        recs.append(
            Rec(
                ROWS,
                "rows",
                "warning",
                f"{n} train rows have no target {target!r}: drop them first",
                "drop_missing_target",
                "train",
                {"target": target},
            )
        )
    n_dup, _ = exact_duplicates(train)
    if n_dup:
        recs.append(
            Rec(
                ROWS,
                "rows",
                "warning",
                f"{n_dup} exact duplicate train rows: keep one copy each "
                "(identical rows, so which one is irrelevant)",
                "drop_duplicates",
                "train",
                {"keep": "first", "sort_by": [str(train.columns[0])]},
            )
        )
    return recs


# --- drops and leaks ----------------------------------------------------------


def _drop(column: str, category: str, severity: str, advice: str) -> Rec:
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


def _drop_missing(column: str, pct: float) -> Rec:
    return _drop(
        column,
        "drop",
        "warning",
        f"{pct}% missing (>= {DROP_PCT:.0f}%): drop it, unless missingness itself "
        "is predictive (then impute with an indicator instead)",
    )


def _drop_rec(
    col: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    semantic: str,
    target: str | None,
) -> Rec | None:
    if test is not None and col not in test.columns:
        return _drop(
            col,
            "leak",
            "warning",
            "only in train: not available at prediction time (a label, or "
            "something computed after the fact); drop it",
        )
    if semantic == "id_like":
        return _drop(
            col,
            "leak",
            "warning",
            "identifier: unique per row, a model can only memorise it (and it may "
            "encode collection order); drop it",
        )
    if target is not None and _names_target(col, target):
        return _drop(
            col,
            "leak",
            "warning",
            f"name derives from the target {target!r} (e.g. a group aggregate of "
            "the label): it leaks each row's own label; drop it",
        )
    if target is not None:
        corr = _target_corr(train, col, target)
        if corr is not None and abs(corr) >= TARGET_CORR:
            return _drop(
                col,
                "leak",
                "warning",
                f"correlation {corr:.3f} with the target {target!r}: a near-copy of "
                "the label (derived from it?); drop it unless it is truly known "
                "before the outcome",
            )
    pct = round(100 * float(train[col].isna().mean()), 2) if len(train) else 0.0
    if pct >= DROP_PCT:
        return _drop_missing(col, pct)
    if semantic == "constant":
        return _drop(col, "drop", "info", "constant: carries no information")
    if semantic == "group_id":
        return _drop(
            col,
            "leak",
            "warning",
            "entity key repeated over rows: use it as `groups` for GroupKFold "
            "rather than as a feature (rows of one entity in train and "
            "validation inflate the score)",
        )
    if semantic == "text":
        return _drop(
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


# --- cleaning: sentinels, variants, types ---------------------------------------


def _sentinel_values(series: pd.Series) -> list:
    """Raw cell values ``sentinel_counts`` flags in ``series`` (same rules, per value).

    A numeric sentinel (-1, 999, ...) counts only outside the range of the other
    values: -1 in a column of temperatures is data, -999 in a column of ages is not.
    """
    if pdt.is_datetime64_any_dtype(series) or pdt.is_bool_dtype(series):
        return []  # replace_sentinels takes scalars, not timestamps
    table = sentinel_counts(series.to_frame("v"))
    if not (set(table["sentinel"]) - UNSAFE_SENTINELS):
        return []
    values = series.dropna()
    if pdt.is_numeric_dtype(series):
        codes = [float(h) for h in table["sentinel"] if h not in UNSAFE_SENTINELS]
        others = values[~values.isin(codes)]
        return sorted(
            _py(v)
            for v in values[values.isin(codes)].unique()
            if others.empty or v < others.min() or v > others.max()
        )
    # Text: the rules of ops.missing, vectorised over the distinct values.
    uniques = pd.Series(values.unique())
    text = uniques.astype(str).str.strip()
    hit = (
        text.str.lower().isin(STRING_SENTINELS)
        | text.str.startswith(DATE_SENTINELS)
        | pd.to_numeric(text, errors="coerce").isin(NUMERIC_SENTINELS)
    )
    return sorted((_py(v) for v in uniques[hit]), key=str)


def _sentinel_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    values: list = []
    for df in frames:
        if col not in df.columns:
            continue
        for v in _sentinel_values(df[col]):
            if v not in values:
                values.append(v)
    if not values:
        return None
    return Rec(
        col,
        "sentinels",
        "warning",
        f"disguised missing values {values}: turn them into NaN before imputing "
        "(else they skew the statistics and the model learns the code)",
        "replace_sentinels",
        "both",
        {"sentinels": {col: values}},
    )


def _variant_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    series = pd.concat([df[col] for df in frames if col in df.columns])
    _, mapping = variants(series.to_frame(col), [col])
    pairs = {
        str(r.variant).strip(): str(r.canonical).strip()
        for r in mapping.itertuples()
        if str(r.variant).strip() != str(r.canonical).strip()
    }
    if mapping.empty:
        return None
    return Rec(
        col,
        "consistency",
        "warning",
        f"spelling variants of the same value ({len(mapping)} forms): merge them "
        "before encoding, else each becomes its own category",
        "standardize_text",
        "both",
        {"columns": [col], "strip": True, "mapping": pairs},
    )


def _type_rec(col: str, frames: list[pd.DataFrame]) -> Rec | None:
    present = [df[col] for df in frames if col in df.columns]
    if all(pdt.is_numeric_dtype(s) for s in present):
        return None
    if any(pdt.is_bool_dtype(s) or pdt.is_datetime64_any_dtype(s) for s in present):
        return None
    if any(s.notna().any() and pct_numeric_parsable(s) < 100 for s in present):
        return None
    if not any(s.notna().any() for s in present):
        return None
    return Rec(
        col,
        "type",
        "warning",
        "numbers stored as text (on at least one side): cast to float so train "
        "and test share a numeric type",
        "cast",
        "both",
        {"dtypes": {col: "float64"}},
    )


# --- missing, skew, scaling -----------------------------------------------------


def _missing_rec(
    col: str,
    semantic: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    info: _Col,
) -> Rec | None:
    test_missing = test is not None and col in test.columns and test[col].isna().any()
    if semantic == "datetime" or not (train[col].isna().any() or test_missing):
        return None  # datetime: datetime_parts keeps NaT as missing parts
    pct = info.pct_missing
    numeric = semantic == "numeric"
    if numeric:
        strategy, why = "median", "median (robust to skew and outliers)"
    elif pct >= REVIEW_PCT:
        strategy, why = "constant", 'a "MISSING" category (missingness kept as a value)'
    else:
        strategy, why = "most_frequent", "the most frequent value"
    indicator = pct >= REVIEW_PCT and numeric
    advice = f"{pct}% missing in train: impute with {why}, fitted on train"
    if indicator:
        advice += (
            f", plus a {col}{INDICATOR_SUFFIX} flag (missingness may be predictive)"
        )
    if not pct:
        advice = f"missing only in test: impute with {why} learned on train"
    params: dict[str, Any] = {"columns": [col], "strategy": strategy}
    if indicator:
        params["add_indicator"] = True
    return Rec(col, "missing", "info", advice, "impute", "both", params)


def _numeric_recs(
    col: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    family: str | None,
    info: _Col,
) -> list[Rec]:
    recs = []
    values = train[col].dropna().astype(float)
    if len(values) < 3:
        return recs
    skew = float(values.skew())
    info.skew = round(skew, 3) if not np.isnan(skew) else None
    mins = [values.min()]
    if test is not None and col in test.columns and test[col].notna().any():
        mins.append(float(test[col].min()))
    screened = train[[col]].astype(float)
    if family != "tree" and skew > SKEW_THRESHOLD and min(mins) >= 0:
        recs.append(
            Rec(
                col,
                "skew",
                "info",
                f"right-skewed (skew {skew:.2f}, no negative value): log1p compresses "
                "the tail; scaling alone would not reshape it",
                "log1p",
                "both",
                {"columns": [col]},
            )
        )
        screened = np.log1p(screened)
    outliers = univariate_outliers(screened, [col])
    pct_out = float(outliers["pct_iqr"].iloc[0])
    info.pct_outliers_iqr = pct_out
    needed = SCALING_NEEDED.get(family) if family else None
    if needed is False:
        return recs
    method = "robust" if pct_out > ROBUST_OUTLIER_PCT else "standard"
    why = (
        f"robust scaler (median / IQR): {pct_out}% IQR outliers, kept on purpose, "
        "would move a mean and a std"
        if method == "robust"
        else "standard scaler (mean 0, std 1)"
    )
    if needed:
        advice = f"{family} models need scaled inputs: {why}, fitted on train"
    else:
        advice = (
            f"scale for linear (regularised), distance-based or neural models, "
            f"not for trees: {why}"
        )
    recs.append(
        Rec(
            col,
            "scaling",
            "info",
            advice,
            "scale",
            "both",
            {"columns": [col], "method": method},
        )
    )
    return recs


# --- encoding -------------------------------------------------------------------


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


def _encoding_recs(
    col: str,
    semantic: str,
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    family: str | None,
    missing_rec: Rec | None,
    info: _Col,
) -> list[Rec]:
    if semantic == "datetime":
        return [
            Rec(
                col,
                "encoding",
                "info",
                "date/time: models need numbers, extract calendar parts "
                "(then drop the raw column)",
                "datetime_parts",
                "both",
                {"column": col},
            ),
            _drop(col, "encoding", "info", "raw date/time replaced by its parts"),
        ]
    if semantic == "boolean":
        if pdt.is_bool_dtype(train[col]):
            return [
                Rec(
                    col,
                    "encoding",
                    "info",
                    "boolean: cast to 0/1",
                    "cast",
                    "both",
                    {"dtypes": {col: "int64"}},
                )
            ]
        return []
    if semantic != "categorical":
        return []

    counts = train[col].value_counts()
    fills_missing = (
        missing_rec is not None and missing_rec.params["strategy"] == "constant"
    )
    n_unique = len(counts) + (1 if fills_missing else 0)
    unseen: list = []
    if test is not None and col in test.columns:
        unseen = sorted(set(test[col].dropna().unique()) - set(counts.index), key=str)
    unseen_note = (
        f"; {len(unseen)} test categories unseen in train (e.g. {unseen[:3]})"
        if unseen
        else ""
    )

    order = _ordinal_order(list(counts.index))
    if order is not None:
        info.onehot_columns = 1
        return [
            Rec(
                col,
                "encoding",
                "info",
                f"ordinal hint: levels {order} have a natural order: code them "
                f"0..{len(order) - 1} (check the order; unknown -> -1){unseen_note}",
                "ordinal",
                "both",
                {"categories": {col: order}},
            )
        ]

    high = n_unique > HIGH_CARDINALITY
    if family == "tree" and high:
        by_freq = [_py(v) for v in counts.index]
        info.onehot_columns = 1
        return [
            Rec(
                col,
                "encoding",
                "warning",
                f"high cardinality ({n_unique} categories) for a tree model: an "
                "arbitrary integer code (by frequency) avoids a mostly-zero one-hot "
                f"matrix; trees can carve the codes apart{unseen_note}",
                "ordinal",
                "both",
                {"categories": {col: by_freq}},
            )
        ]

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
    severity = "warning" if high else "info"
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
    return [Rec(col, "encoding", severity, advice, "onehot", "both", params)]


# --- entry point ----------------------------------------------------------------


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
    """
    if model_family is not None and model_family not in MODEL_FAMILIES:
        raise ValueError(
            f"unknown model_family {model_family!r} (known: {list(MODEL_FAMILIES)})"
        )
    if target is not None and target not in train.columns:
        raise ValueError(f"target {target!r} is not a column of the source")

    recs = _row_recs(train, target)
    semantic = semantic_types(train)
    features = [str(c) for c in train.columns if c != target]
    infos = {c: _summary(train, c, semantic[c]) for c in features}
    frames = [train] if test is None else [train, test]

    # Drops first: a dropped column gets no other advice.
    kept = []
    for col in features:
        rec = _drop_rec(col, train, test, semantic[col], target)
        if rec is None:
            kept.append(col)
        else:
            recs.append(rec)
            infos[col].action = "drop"
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

    # Cleaning, applied to working copies so later checks see clean columns.
    work_train, work_test = train, test
    text = set(text_columns(train))
    for col in kept:
        cleaning = [_sentinel_rec(col, frames)]
        if col in text:
            cleaning.append(_variant_rec(col, frames))
        for rec in filter(None, cleaning):
            recs.append(rec)
            work_train, work_test = _apply(rec, work_train, work_test)
        rec = _type_rec(col, [f for f in (work_train, work_test) if f is not None])
        if rec is not None:
            recs.append(rec)
            work_train, work_test = _apply(rec, work_train, work_test)

    semantic = semantic_types(work_train[kept])
    for col in kept:
        info = infos[col] = _summary(work_train, col, semantic[col])
        if semantic[col] == "constant":
            recs.append(_drop(col, "drop", "info", "constant once cleaned"))
            info.action = "drop"
            continue
        if info.pct_missing >= DROP_PCT:
            recs.append(_drop_missing(col, info.pct_missing))
            info.action = "drop"
            continue
        missing = _missing_rec(col, semantic[col], work_train, work_test, info)
        if missing is not None:
            recs.append(missing)
        if semantic[col] == "numeric":
            recs += _numeric_recs(col, work_train, work_test, model_family, info)
        recs += _encoding_recs(
            col, semantic[col], work_train, work_test, model_family, missing, info
        )
        ops = [r.op for r in recs if r.column == col and r.op != "drop_columns"]
        info.action = ", ".join(ops) if ops else "keep"

    rows = sorted(enumerate(recs), key=lambda ir: (STAGES[ir[1].category], ir[0]))
    table = pd.DataFrame(
        [
            {
                "order": i + 1,
                "column": r.column,
                "category": r.category,
                "severity": r.severity,
                "advice": r.advice,
                "op": r.op,
                "target": r.target,
                "params": r.params,
            }
            for i, (_, r) in enumerate(rows)
        ],
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
