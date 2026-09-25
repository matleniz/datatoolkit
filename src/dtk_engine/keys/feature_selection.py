"""Feature selection: filter / wrapper / embedded scores, collinearity, PCA variance."""

from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
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
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load


class Params(KeyParams):
    source: SourceSpec = CsvSource(path=TRAIN_CSV)
    target: str = Field(default="Survived", description="Target column of `source`")
    task: Literal["auto", "classification", "regression"] = Field(
        default="auto",
        description="auto: classification for a non-numeric target or a few "
        "integer values, regression otherwise",
    )
    columns: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Numeric features to score (default: every numeric column "
        "but the target)",
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
        raise ValueError(f"feature_selection: target {target!r} not in the frame")
    columns = candidate_columns(df, target, columns, "feature_selection")
    task = resolve_task(df[target], task)
    scores = feature_scores(df, target, task, columns, wrapper, random_state)
    pairs = collinear_pairs(df, columns)
    constant = near_constant(df, columns)
    pca_table, needed = pca_variance(df, columns)
    steps = suggested_steps(
        target, scores, pairs, constant, needed["pca_components_95pct"]
    )
    result = Result(
        metrics={
            "task": task,
            "n_rows": len(df),
            "n_features": len(columns),
            "n_collinear_pairs": len(pairs),
            "n_near_constant": len(constant),
            "top_feature": str(scores["column"].iloc[0]),
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
    result.add_table("suggested_steps", steps)
    result.add_figure(
        "Mutual information per feature (color: combined rank, 1 = best)",
        px.bar(scores, x="column", y="mutual_info", color="combined_rank"),
    )
    result.add_figure(
        "PCA cumulative explained variance (standardized features)",
        px.line(pca_table, x="component", y="cumulative", markers=True),
    )
    return result


def _missing_hint(scores: pd.DataFrame) -> str:
    missing = scores.loc[scores["pct_missing"] > 0, "column"].tolist()
    if not missing:
        return ""
    return (
        f"\n\nMissing values in {missing}: scores here use a median fill; the "
        "selection ops refuse missing values, add an impute step before them."
    )
