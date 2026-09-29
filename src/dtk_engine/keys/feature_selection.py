"""Feature selection: filter / wrapper / embedded scores, collinearity, PCA variance."""

from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.advisor import advise
from dtk_engine.ops.selection import (
    COLLINEAR_CORR,
    FAMILIES,
    FAMILIES_TEXT,
    candidate_columns,
    collinear_pairs,
    feature_scores,
    near_constant,
    pca_variance,
    resolve_task,
    suggested_steps,
)
from dtk_engine.params import KeyParams, column_field, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result, plotly_lock
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    target: str = column_field("Survived", "Target column of `source`")
    task: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="auto: classification for a non-numeric target or a few "
        "integer values, regression otherwise",
    )
    columns: list[str] | None = columns_field(
        "Numeric features to score (default: every numeric column but the target)",
        dtype="numeric",
        nullable=True,
        min_length=1,
    )
    wrapper: bool = Field(
        default=False,
        description="Also run RFECV (wrapper family: one model fit per step, slow)",
    )
    random_state: int = Field(default=0, description="Seed of the models")


@key(
    id="feature_selection",
    title="Feature selection",
    category="analysis",
    description="Per numeric feature: variance, missing %, filter scores (mutual "
    "information, ANOVA F / f_regression, |corr| with the target), embedded "
    "scores (L1 coefficient, random-forest importance), optional RFECV rank and "
    "a combined rank; collinear pairs, near-constant columns, PCA explained "
    "variance, and suggested selection steps.",
)
def run(params: Params) -> Result:
    return selection_result(
        load(params.source),
        params.target,
        params.task,
        params.columns,
        params.wrapper,
        params.random_state,
    )


def selection_result(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    columns: list[str] | None = None,
    wrapper: bool = False,
    random_state: int = 0,
) -> Result:
    """The key's Result on a DataFrame (shared with ``dtk_engine.api.select_features``)."""
    if target not in df.columns:
        raise KeyParamsError(f"feature_selection: target {target!r} not in the frame")
    if df[target].notna().sum() == 0:
        raise KeyParamsError(f"feature_selection: target {target!r} is all missing")
    task = resolve_task(df[target], task)
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise KeyParamsError(
            f"feature_selection: regression needs a numeric target, {target!r} is not"
        )
    if columns is None and not _numeric_features(df, target):
        return _encode_first_result(df, target, task)
    try:
        columns = candidate_columns(df, target, columns, "feature_selection")
    except (KeyError, ValueError) as exc:
        raise KeyParamsError(exc.args[0]) from exc
    scores = feature_scores(df, target, task, columns, wrapper, random_state)
    pairs = collinear_pairs(df, columns)
    constant = near_constant(df, columns)
    pca_table, needed = pca_variance(df, columns)
    steps = suggested_steps(
        target, scores, pairs, constant, needed["pca_components_95pct"]
    )
    top_feat = str(scores["column"].iloc[0])
    pca95 = needed["pca_components_95pct"]
    headline = (
        f"Top feature: {top_feat}; {pca95} PCA components explain 95 % variance "
        f"({len(columns)} features scored)"
    )
    result = Result(
        headline=headline,
        metrics={
            "task": task,
            "n_rows": len(df),
            "n_features": len(columns),
            "n_collinear_pairs": len(pairs),
            "n_near_constant": len(constant),
            "top_feature": top_feat,
            **needed,
        },
        text=FAMILIES_TEXT
        + "\n\nSuggested steps (each a 'both' step, fitted on train):\n"
        + "\n".join(f"{r.order}. {r.op}: {r.why}" for r in steps.itertuples())
        + _missing_hint(scores),
    )
    result.add_table("feature_scores", scores)
    result.add_table(f"collinear_pairs (|corr| >= {COLLINEAR_CORR})", pairs)
    result.add_table("near_constant", constant)
    result.add_table("pca_explained_variance", pca_table)
    result.add_table("families", FAMILIES)
    result.add_table("suggested_steps", steps, kind="steps")
    with plotly_lock:
        result.add_figure(
            "Mutual information per feature (color: combined rank, 1 = best)",
            px.bar(scores, x="column", y="mutual_info", color="combined_rank"),
            main=True,
        )
        result.add_figure(
            "PCA cumulative explained variance (standardized features)",
            px.line(pca_table, x="component", y="cumulative", markers=True),
        )
    return result


def _numeric_features(df: pd.DataFrame, target: str) -> list[str]:
    return [
        str(c)
        for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]


def _encode_first_result(df: pd.DataFrame, target: str, task: str) -> Result:
    """No numeric feature to score: the advisor's preparation steps (encode the
    categories, parse dates, drop what cannot be a feature), then re-run."""
    steps, summary = advise(df, target=target)
    features = summary[summary["column"] != target]
    result = Result(
        headline=f"No numeric features to score for target {target!r}; encode first",
        metrics={
            "task": task,
            "n_rows": len(df),
            "n_features": 0,
            "n_non_numeric_columns": len(features),
            "n_encode_steps": len(steps),
        },
        text=(
            f"No numeric feature besides the target {target!r}: selection scores "
            "need numbers. Apply the encode-first steps (preprocessing_advisor: "
            "onehot / ordinal for categories, datetime_parts for dates, drops for "
            "ids / free text), then re-run feature_selection on the encoded table."
        ),
    )
    result.add_table("non_numeric_columns", features)
    result.add_table("encode_first_steps", steps, kind="steps")
    result.add_table("families", FAMILIES)
    return result


def _missing_hint(scores: pd.DataFrame) -> str:
    missing = scores.loc[scores["pct_missing"] > 0, "column"].tolist()
    if not missing:
        return ""
    return (
        f"\n\nMissing values in {missing}: scores here use a median fill; the "
        "selection ops refuse missing values, add an impute step before them."
    )
