"""Encoding ops (categorical -> numeric, fitted on train).

``onehot`` learns the vocabulary on train (state: kept and infrequent
categories per column); ``ordinal`` takes its order from the params, never
from the data (alphabetical order is rarely the real one).
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator

from dtk_engine.ops._util import py as _py
from dtk_engine.transform_registry import TransformParams, transform

INFREQUENT_SUFFIX = "infrequent"
UNKNOWN_CODE = -1

Category = str | int | float | bool


# --- onehot -------------------------------------------------------------------


class OnehotParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Categorical columns")
    min_frequency: int | float | None = Field(
        default=None,
        gt=0,
        description="Categories seen fewer times on train (count, or fraction of "
        "non-missing rows if < 1) share one <col>_infrequent column",
    )
    drop_first: bool = Field(
        default=False,
        description="Drop the first category's column (linear models without "
        "regularization); an unknown value then looks like that category",
    )
    handle_unknown: Literal["ignore", "error"] = Field(
        default="ignore",
        description="Category unseen on train: all-zero row (ignore) or raise",
    )

    @field_validator("min_frequency")
    @classmethod
    def _fraction_below_one(cls, v):
        if isinstance(v, float) and v >= 1:
            raise ValueError("a float min_frequency is a fraction, it must be < 1")
        return v


def _fit_onehot(df: pd.DataFrame, params: OnehotParams) -> dict:
    state = {}
    for col in params.columns:
        counts = df[col].value_counts(dropna=True)
        threshold = 0
        if params.min_frequency is not None:
            threshold = params.min_frequency
            if isinstance(threshold, float):
                threshold = math.ceil(threshold * counts.sum())
        # Sorted by their text so the output columns do not depend on row order.
        cats = sorted(counts.index, key=str)
        state[col] = {
            "categories": [_py(c) for c in cats if counts[c] >= threshold],
            "infrequent": [_py(c) for c in cats if counts[c] < threshold],
        }
    return state


def _dummies(s: pd.Series, col: str, learned: dict, params: OnehotParams):
    kept, infrequent = learned["categories"], learned["infrequent"]
    known = s.isin(kept + infrequent) | s.isna()
    if params.handle_unknown == "error" and not known.all():
        unseen = sorted(map(str, s[~known].unique()))
        raise ValueError(f"onehot: unknown categories in {col!r}: {unseen}")
    columns = {f"{col}_{c}": (s == c) for c in kept[1 if params.drop_first else 0 :]}
    if infrequent:
        columns[f"{col}_{INFREQUENT_SUFFIX}"] = s.isin(infrequent)
    return pd.DataFrame(columns, index=s.index).astype(int)


@transform("onehot", params_model=OnehotParams, fit=_fit_onehot, title="One-hot encode")
def onehot(df: pd.DataFrame, params: OnehotParams, state: dict) -> pd.DataFrame:
    """One 0/1 column per train category, in place of the column.

    Output names are ``<col>_<category>`` (categories sorted as text), plus
    ``<col>_infrequent`` when ``min_frequency`` grouped some. Missing and
    unknown values give an all-zero row (impute first to keep missingness).
    """
    missing = [c for c in params.columns if c not in df.columns]
    if missing:
        raise KeyError(f"onehot: columns not in frame {missing}")
    pieces = []
    for name in df.columns:
        if name in params.columns:
            pieces.append(_dummies(df[name], name, state[name], params))
        else:
            pieces.append(df[[name]])
    out = pd.concat(pieces, axis=1)
    if out.columns.duplicated().any():
        dupes = sorted(set(out.columns[out.columns.duplicated()]))
        raise ValueError(f"onehot: output column names collide {dupes}")
    return out


# --- ordinal ------------------------------------------------------------------


class OrdinalParams(TransformParams):
    categories: dict[str, list[Category]] = Field(
        min_length=1,
        description="Column -> its categories from lowest to highest "
        '(e.g. {"size": ["S", "M", "L"]}); coded 0, 1, 2, …',
    )

    @field_validator("categories")
    @classmethod
    def _ordered_unique(cls, v):
        for col, cats in v.items():
            if not cats:
                raise ValueError(f"ordinal: no categories given for {col!r}")
            if len(set(cats)) != len(cats):
                raise ValueError(f"ordinal: duplicate categories for {col!r}")
        return v


@transform("ordinal", params_model=OrdinalParams, title="Ordinal encode")
def ordinal(df: pd.DataFrame, params: OrdinalParams, state: dict) -> pd.DataFrame:
    """Code each category by its rank in an explicit order (unknown -> -1).

    Missing values stay missing.
    """
    out = df.copy()
    for col, cats in params.categories.items():
        s = out[col]
        codes = s.map({c: i for i, c in enumerate(cats)})
        codes[s.notna() & codes.isna()] = UNKNOWN_CODE
        out[col] = codes if codes.isna().any() else codes.astype(int)
    return out
