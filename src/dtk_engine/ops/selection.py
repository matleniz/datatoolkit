"""Feature selection analysis (course "Feature Selection"): filter, wrapper and
embedded scores per feature, collinear pairs, near-constant columns, PCA variance.

Pure pandas / sklearn, no ``Result``. The scoring helpers (task inference,
candidate columns, filter scores, embedded models) are shared with the
selection transform ops in ``dtk_engine.ops.transforms.selection``.

Scores need complete data: the analysis drops rows with a missing target and
fills missing feature values with the column median (the ``pct_missing`` column
says where); the transform ops refuse missing values instead (impute first).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFECV,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression

from dtk_engine.cache import memo_frame

# task "auto": an integer-valued numeric target with at most this many distinct
# values is a class label, not a quantity.
MAX_CLASSES = 10
# |corr| between two features at or above this: collinear pair (analysis).
COLLINEAR_CORR = 0.9
# A column whose most frequent value covers this share of rows is near-constant.
NEAR_CONSTANT_SHARE = 0.95
PCA_TARGETS = (0.90, 0.95, 0.99)
N_ESTIMATORS = 100
CV_FOLDS = 5
# The analysis scores (MI, L1, forest, RFECV) are fitted on at most this many
# rows (a seeded sample of the rows with a target); larger frames are sampled.
SCORE_SAMPLE_SIZE = 10_000

FAMILIES = pd.DataFrame(
    [
        {
            "family": "filter",
            "how it decides": "a statistic per feature against the target "
            "(mutual information, ANOVA F, |corr|)",
            "cost": "one pass",
            "blind spot": "interactions: a feature useless alone but useful "
            "with another scores zero",
            "op": "select_k_best",
        },
        {
            "family": "wrapper",
            "how it decides": "train the model on candidate subsets (RFECV)",
            "cost": "one fit per step",
            "blind spot": "overfits the selection on small data",
            "op": None,
        },
        {
            "family": "embedded",
            "how it decides": "selection happens during fitting (L1 zeroes "
            "coefficients, tree importances)",
            "cost": "nearly free",
            "blind spot": "tied to that model: another model may need other columns",
            "op": "select_from_model",
        },
    ]
)
FAMILIES_TEXT = """Three families of feature selection:
- filter (select_k_best): a statistic per feature against the target, one pass. Blind spot: interactions.
- wrapper (RFECV): trains the model on candidate subsets, one fit per step. Blind spot: overfits on small data.
- embedded (select_from_model): selection during fitting (L1, tree importance), nearly free. Blind spot: tied to that model.
Every selector is a fitted step: fit it on train (a "both" step), never on train + test.
PCA needs scaled inputs (variance has units) and is fitted on train like anything else."""


# --- shared helpers -----------------------------------------------------------


def infer_task(y: pd.Series) -> str:
    """classification for a non-numeric / bool target or a few integer values,
    regression otherwise."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    values = y.dropna()
    if values.nunique() <= MAX_CLASSES and (values == values.round()).all():
        return "classification"
    return "regression"


def resolve_task(y: pd.Series, task: str) -> str:
    return infer_task(y) if task == "auto" else task


def candidate_columns(
    df: pd.DataFrame, target: str | None, columns: list[str] | None, op: str
) -> list[str]:
    """``columns`` (checked numeric, target refused) or every numeric column but
    the target."""
    if columns is None:
        columns = [
            c
            for c in df.columns
            if c != target and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not columns:
            raise ValueError(f"{op}: no numeric candidate column (encode first)")
        return columns
    if target is not None and target in columns:
        raise ValueError(f"{op}: the target {target!r} cannot be a candidate column")
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyError(f"{op}: columns not in the frame {absent}")
    bad = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(
            f"{op} needs numeric columns, got non-numeric {bad} (encode first)"
        )
    return list(columns)


def require_target(df: pd.DataFrame, target: str, op: str) -> pd.Series:
    if target not in df.columns:
        raise KeyError(
            f"{op}: target column {target!r} not in the fit frame "
            "(in sklearn, pass y to fit)"
        )
    return df[target]


def target_vector(df: pd.DataFrame, target: str, task: str, op: str) -> np.ndarray:
    """The target as a model-ready vector (classes factorized); missing -> error."""
    y = require_target(df, target, op)
    if y.isna().any():
        raise ValueError(
            f"{op}: target {target!r} has missing values (drop those rows)"
        )
    if task == "classification":
        return pd.factorize(y, sort=True)[0]
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"{op}: regression needs a numeric target, {target!r} is not")
    return y.astype(float).to_numpy()


def feature_matrix(df: pd.DataFrame, columns: list[str], op: str) -> np.ndarray:
    """The candidate columns as floats; missing values -> clear error."""
    missing = [c for c in columns if df[c].isna().any()]
    if missing:
        raise ValueError(
            f"{op}: missing values in {missing}; add an impute step before this one"
        )
    return df[columns].astype(float).to_numpy()


def standardize(X: np.ndarray) -> np.ndarray:
    """Mean 0, std 1 per column (a constant column is only centered)."""
    std = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(std == 0, 1.0, std)


def filter_scores(
    X: np.ndarray, y: np.ndarray, task: str, random_state: int = 0
) -> dict[str, np.ndarray]:
    """mutual_info, f_score, f_pvalue per column (constant column: 0, p 1)."""
    if task == "classification":
        mi = mutual_info_classif(X, y, random_state=random_state)
    else:
        mi = mutual_info_regression(X, y, random_state=random_state)
    constant = X.std(axis=0) == 0
    f, p = np.zeros(X.shape[1]), np.ones(X.shape[1])
    if (~constant).any():
        score_fn = f_classif if task == "classification" else f_regression
        with np.errstate(divide="ignore", invalid="ignore"):
            f[~constant], p[~constant] = score_fn(X[:, ~constant], y)
    return {
        "mutual_info": mi,
        "f_score": np.nan_to_num(f, nan=0.0, posinf=np.finfo(float).max),
        "f_pvalue": np.nan_to_num(p, nan=1.0),
    }


def l1_model(task: str, random_state: int = 0, n_classes: int = 2):
    """L1 logistic regression (classification) or LassoCV (regression); expects
    standardized inputs."""
    if task == "regression":
        return LassoCV(cv=CV_FOLDS, random_state=random_state)
    # liblinear is exact for binary; it refuses 3+ classes, saga does not.
    solver = "liblinear" if n_classes <= 2 else "saga"
    return LogisticRegression(
        l1_ratio=1.0, solver=solver, max_iter=5000, random_state=random_state
    )


def tree_model(task: str, random_state: int = 0):
    cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    return cls(n_estimators=N_ESTIMATORS, random_state=random_state)


def model_importance(model) -> np.ndarray:
    """|coef| (summed over classes) or feature_importances_."""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    coef = np.atleast_2d(model.coef_)
    return np.abs(coef).sum(axis=0)


# --- analysis -----------------------------------------------------------------


def _analysis_frame(
    df: pd.DataFrame, target: str, columns: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = df[df[target].notna()]
    X = rows[columns].astype(float)
    return rows, X.fillna(X.median()).fillna(0.0).to_numpy()


def score_sample_rows(n_rows: int) -> int:
    """Rows the analysis scores are fitted on, out of ``n_rows`` with a target."""
    return min(n_rows, SCORE_SAMPLE_SIZE)


def feature_scores(
    df: pd.DataFrame,
    target: str,
    task: str,
    columns: list[str],
    wrapper: bool = False,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per feature: variance, pct_missing, filter scores, embedded scores,
    optional RFECV rank and a combined rank (1 = most useful), best first.
    Above ``SCORE_SAMPLE_SIZE`` rows the scores are fitted on a seeded row sample
    (variance and pct_missing still use every row). Memoized on the content of ``df[[target, *columns]]`` and the params."""
    return memo_frame(
        "feature_scores",
        df[list(dict.fromkeys([target, *columns]))],
        (target, task, list(columns), wrapper, random_state, SCORE_SAMPLE_SIZE),
        lambda: _feature_scores(df, target, task, columns, wrapper, random_state),
    )


def _feature_scores(
    df: pd.DataFrame,
    target: str,
    task: str,
    columns: list[str],
    wrapper: bool,
    random_state: int,
) -> pd.DataFrame:
    rows, X = _analysis_frame(df, target, columns)
    y = target_vector(rows, target, task, "feature_selection")
    if len(y) > SCORE_SAMPLE_SIZE:
        keep = np.sort(
            np.random.default_rng(random_state).choice(
                len(y), SCORE_SAMPLE_SIZE, replace=False
            )
        )
        X, y = X[keep], y[keep]
    table = pd.DataFrame(
        {
            "column": columns,
            "variance": df[columns].astype(float).var(ddof=0).to_numpy(),
            "pct_missing": (df[columns].isna().mean() * 100).round(2).to_numpy(),
        }
    )
    for name, values in filter_scores(X, y, task, random_state).items():
        table[name] = values
    y_num = y.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = [abs(np.corrcoef(X[:, i], y_num)[0, 1]) for i in range(X.shape[1])]
    table["abs_corr_target"] = np.nan_to_num(corr, nan=0.0)
    Z = standardize(X)
    n_classes = len(np.unique(y)) if task == "classification" else 0
    table["l1_coef"] = model_importance(
        l1_model(task, random_state, n_classes).fit(Z, y)
    )
    table["tree_importance"] = model_importance(
        tree_model(task, random_state).set_params(n_jobs=-1).fit(X, y)
    )
    score_columns = [
        "mutual_info",
        "f_score",
        "abs_corr_target",
        "l1_coef",
        "tree_importance",
    ]
    ranks = table[score_columns].rank(ascending=False, method="average")
    if wrapper:
        estimator = (
            LogisticRegression(max_iter=1000)
            if task == "classification"
            else LinearRegression()
        )
        cv = min(CV_FOLDS, _max_folds(y, task))
        table["rfecv_rank"] = RFECV(estimator, cv=cv).fit(Z, y).ranking_
        ranks["rfecv_rank"] = table["rfecv_rank"]
    table["combined_rank"] = (
        ranks.mean(axis=1).rank(method="min").astype(int).to_numpy()
    )
    return table.sort_values(["combined_rank", "column"], kind="stable").reset_index(
        drop=True
    )


def _max_folds(y: np.ndarray, task: str) -> int:
    if task == "classification":
        return max(2, int(np.bincount(y).min()))
    return max(2, len(y) // 2)


def collinear_pairs(
    df: pd.DataFrame, columns: list[str], threshold: float = COLLINEAR_CORR
) -> pd.DataFrame:
    """Feature pairs with |Pearson corr| >= threshold, strongest first."""
    corr = df[columns].astype(float).corr()
    pairs = [
        {"a": a, "b": b, "corr": float(corr.loc[a, b])}
        for i, a in enumerate(columns)
        for b in columns[i + 1 :]
        if abs(corr.loc[a, b]) >= threshold
    ]
    out = pd.DataFrame(pairs, columns=["a", "b", "corr"])
    return out.reindex(out["corr"].abs().sort_values(ascending=False).index)


def near_constant(
    df: pd.DataFrame, columns: list[str], share: float = NEAR_CONSTANT_SHARE
) -> pd.DataFrame:
    """Columns whose most frequent value covers >= ``share`` of non-null rows."""
    rows = []
    for col in columns:
        s = df[col].dropna()
        top = float(s.value_counts(normalize=True).iloc[0]) if len(s) else 1.0
        if top >= share:
            rows.append(
                {
                    "column": col,
                    "top_value_share": round(top, 4),
                    "variance": float(s.astype(float).var(ddof=0)) if len(s) else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=["column", "top_value_share", "variance"])


def pca_variance(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """PCA on the standardized (median-filled) features: per component explained
    variance ratio and cumulative, plus the components needed for 90 / 95 / 99 %."""
    X = df[columns].astype(float)
    Z = standardize(X.fillna(X.median()).fillna(0.0).to_numpy())
    ratio = PCA(svd_solver="full").fit(Z).explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    table = pd.DataFrame(
        {
            "component": [f"pc{i + 1}" for i in range(len(ratio))],
            "explained_variance_ratio": ratio,
            "cumulative": cumulative,
        }
    )
    needed = {
        f"pca_components_{round(p * 100)}pct": int(
            min(np.searchsorted(cumulative, p - 1e-12) + 1, len(ratio))
        )
        for p in PCA_TARGETS
    }
    return table, needed


def suggested_steps(
    target: str,
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
    constant: pd.DataFrame,
    n_components_95: int,
) -> pd.DataFrame:
    """Selection steps in pipeline order (op, target, params), each fitted on train."""
    steps = []
    if len(constant):
        steps.append(
            (
                "drop_low_variance",
                {"threshold": 0.0, "target": target},
                f"drop constant columns ({len(constant)} near-constant found)",
            )
        )
    if len(pairs):
        steps.append(
            (
                "drop_correlated",
                {"threshold": 0.95, "target": target},
                (
                    f"drop one of each collinear pair ({len(pairs)} pairs >= "
                    f"{COLLINEAR_CORR})"
                ),
            )
        )
    n = len(scores)
    if n > 1:
        k = max(1, math.ceil(n / 2))
        steps.append(
            (
                "select_k_best",
                {"target": target, "score": "mutual_info", "k": k},
                f"filter: keep the {k} best features by mutual information",
            )
        )
        steps.append(
            (
                "select_from_model",
                {"target": target, "model": "tree", "threshold": "median"},
                "embedded (alternative): keep features above the median importance",
            )
        )
    if n > 2 and n_components_95 < n:
        steps.append(
            (
                "pca",
                {"n_components": 0.95, "standardize": True, "target": target},
                (
                    f"compress to {n_components_95} components (95 % variance) "
                    "for a distance-based model"
                ),
            )
        )
    return pd.DataFrame(
        [
            {"order": i + 1, "op": op, "target": "both", "params": p, "why": why}
            for i, (op, p, why) in enumerate(steps)
        ],
        columns=["order", "op", "target", "params", "why"],
    )
