"""sklearn door: any registered transform as a fit / transform estimator.

``DtkTransformer(op, **params)`` wraps one op (pandas in, pandas out) so it runs
inside ``Pipeline`` / ``cross_val_score``: ``fit`` learns the op's state on the
training fold, ``transform`` applies it. A supervised op (``needs_target``, e.g.
``select_k_best``) gets ``y`` joined to X under its ``target`` param name for
fit only; unsupervised ops ignore ``y``. ``workspace_pipeline(name)`` chains the
workspace's steps into a ``Pipeline``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from dtk_engine.errors import UnknownTransformError
from dtk_engine.ops import transforms  # noqa: F401  (registers every transform op)
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import JsonWorkspaceStore


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


class DtkTransformer(TransformerMixin, BaseEstimator):
    """One transform op as an sklearn transformer, e.g.
    ``DtkTransformer("drop_columns", columns=["Name"])``.

    Fitted attributes: ``state_`` (the op's JSON-safe state),
    ``feature_names_in_``, ``n_features_in_``, ``feature_names_out_``.
    Params are validated at ``fit`` (invalid -> KeyParamsError).
    """

    def __init__(self, op: str, **params):
        self.op = op
        self.params = params

    # The op's params are the estimator's params, so clone / grid search see them.
    def get_params(self, deep: bool = True) -> dict:
        return {"op": self.op, **self.params}

    def set_params(self, **params) -> DtkTransformer:
        # A new op keeps only the current params it declares (e.g. `columns`);
        # the old op's own params (e.g. `missing_ok`) are dropped.
        if "op" in params and (op := params.pop("op")) != self.op:
            try:
                fields = get_transform(op).params_model.model_fields
            except UnknownTransformError:
                fields = {}  # unknown op: fails at fit, carry nothing over
            self.op = op
            self.params = {k: v for k, v in self.params.items() if k in fields}
        self.params = {**self.params, **params}
        return self

    def _fit(self, X, y=None) -> pd.DataFrame:
        X = _as_frame(X)
        t = get_transform(self.op)
        params = t.parse(self.params)
        fit_frame = X
        target = params.target if t.needs_target else None
        if target is not None and y is not None and target not in X.columns:
            fit_frame = X.assign(**{target: np.asarray(y)})
        self.state_ = t.fit(fit_frame, params)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        out = t.apply(X, params, self.state_)
        self.feature_names_out_ = np.asarray(out.columns, dtype=object)
        return out

    def fit(self, X, y=None) -> DtkTransformer:
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params) -> pd.DataFrame:
        return self._fit(X, y)

    def transform(self, X) -> pd.DataFrame:
        check_is_fitted(self, "state_")
        t = get_transform(self.op)
        return t.apply(_as_frame(X), t.parse(self.params), self.state_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def workspace_pipeline(name: str, store=None) -> Pipeline:
    """The workspace's "both" steps, in order, as an unfitted ``Pipeline``.

    "both" steps are the ones fitted on train and applied to new data. "train" /
    "test"-only steps (row cleaning of one side) are not part of a model
    pipeline and are skipped. Ops that add or drop rows break the X / y
    alignment sklearn expects; keep them out of "both" steps you train on.
    """
    ws = (store if store is not None else JsonWorkspaceStore()).get(name)
    steps = [
        (f"{i}_{step.op}", DtkTransformer(step.op, **step.params))
        for i, step in enumerate(ws.steps)
        if step.target == "both"
    ]
    for _, est in steps:  # unknown op / invalid params fail here, not at fit
        get_transform(est.op).parse(est.params)
    if not steps:
        steps = [("passthrough", "passthrough")]
    return Pipeline(steps)
