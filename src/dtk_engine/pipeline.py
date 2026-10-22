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
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_array, check_is_fitted

from dtk_engine.errors import UnknownTransformError
from dtk_engine.ops import transforms  # noqa: F401  (registers every transform op)
from dtk_engine.transform_registry import get_transform
from dtk_engine.workspace import JsonWorkspaceStore


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    # Array-likes: sklearn's validation gives the standard errors (1d, empty,
    # sparse, complex); NaN / inf and any dtype stay the op's business.
    X = check_array(X, dtype=None, ensure_all_finite=False)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _check_feature_names(expected: np.ndarray, X: pd.DataFrame) -> None:
    """sklearn's ``_check_feature_names`` rule and wording (same names, same order)."""
    got = np.asarray(X.columns, dtype=object)
    if len(got) == len(expected) and bool(np.all(got == expected)):
        return
    unexpected = sorted(set(got) - set(expected), key=str)
    missing = sorted(set(expected) - set(got), key=str)

    def names(items) -> str:
        out = "".join(f"- {n}\n" for n in items[:5])
        return out + ("- ...\n" if len(items) > 5 else "")

    message = "The feature names should match those that were passed during fit.\n"
    if unexpected:
        message += "Feature names unseen at fit time:\n" + names(unexpected)
    if missing:
        message += "Feature names seen at fit time, yet now missing:\n" + names(missing)
    if not unexpected and not missing:
        message += "Feature names must be in the same order as they were in fit.\n"
    raise ValueError(message)


_ESC = {"op": "param_op"}
_UNESC = {v: k for k, v in _ESC.items()}


class DtkTransformer(TransformerMixin, BaseEstimator):
    """One transform op as an sklearn transformer, e.g.
    ``DtkTransformer("drop_columns", columns=["Name"])``.

    Fitted attributes: ``state_`` (the op's JSON-safe state),
    ``feature_names_in_``, ``n_features_in_``, ``feature_names_out_``.
    Params are validated at ``fit`` (invalid -> KeyParamsError).

    ``strict`` (keyword-only, default ``False``) is an estimator hyperparameter,
    not an op param. With ``strict=False`` a DataFrame given to ``transform`` is
    matched by column name and is not checked against the columns seen at fit:
    the label is typically present at fit and legitimately absent from test, and
    exported scripts rely on it. With ``strict=True`` the DataFrame's columns must
    equal ``feature_names_in_`` (same names, same order, as scikit-learn's own
    estimators require); otherwise ``transform`` raises a ``ValueError`` listing
    the unseen and the missing names. Use it when the estimator sits in a model
    pipeline fed only features, to catch a schema drift early. Array input is
    unaffected: only the width is checked. Fitted on an array (names ``x0``...),
    a named DataFrame then raises under ``strict=True``, where sklearn only warns.
    """

    # The op identity is positional-only so an op param may itself be called
    # ``op`` (``derive``): ``DtkTransformer("derive", op="ratio")``.
    # ``op=None`` only makes the class default-constructible; ``fit`` refuses it.
    def __init__(self, op: str | None = None, /, *, strict: bool = False, **params):
        self.op = op
        self.strict = strict
        self._params = params

    # Kept out of the instance's public attributes: the op's params are
    # exposed through get_params / set_params, not as a constructor argument.
    @property
    def params(self) -> dict:
        return self._params

    @params.setter
    def params(self, value: dict) -> None:
        self._params = value

    # The op's params are the estimator's params, so grid search sees them.
    # ``op`` is the transform's identity; an op param also called ``op`` is
    # exposed as ``param_op`` (never a plain ``op``, which would shadow it).
    def __setstate__(self, state: dict) -> None:
        # Pickles written before ``params`` became a property store it as "params".
        if "params" in state and "_params" not in state:
            state = {**state, "_params": state.pop("params")}
        state.setdefault("strict", False)  # pickles written before ``strict``
        super().__setstate__(state)

    def get_params(self, deep: bool = True) -> dict:
        return {"op": self.op, "strict": self.strict, **{_ESC.get(k, k): v for k, v in self.params.items()}}

    def set_params(self, **params) -> DtkTransformer:
        # A new op keeps only the current params it declares (e.g. `columns`);
        # the old op's own params (e.g. `missing_ok`) are dropped.
        new_op = params.pop("op", self.op)
        if "strict" in params:
            self.strict = params.pop("strict")
        params = {_UNESC.get(k, k): v for k, v in params.items()}
        if not (isinstance(new_op, str) and new_op == self.op):
            try:
                fields = get_transform(new_op).params_model.model_fields
            except (UnknownTransformError, TypeError):
                fields = {}  # unknown / non-string op: fails at fit, carry nothing over
            self.op = new_op
            self.params = {k: v for k, v in self.params.items() if k in fields}
        self.params = {**self.params, **params}
        return self

    def __sklearn_clone__(self) -> DtkTransformer:
        # clone()'s default `type(self)(**get_params())` would hand `op` to the
        # positional-only argument as a keyword; rebuild it explicitly.
        return type(self)(self.op, strict=self.strict, **{k: clone(v, safe=False) for k, v in self.params.items()})

    def _transform_op(self):
        if self.op is None:
            raise ValueError("DtkTransformer needs an op name, e.g. DtkTransformer('scale', ...)")
        return get_transform(self.op)

    def _fit(self, X, y=None) -> pd.DataFrame:
        X = _as_frame(X)
        t = self._transform_op()
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
        t = self._transform_op()
        if not isinstance(X, pd.DataFrame):
            # Arrays carry no names: the width is the only thing to check.
            X = _as_frame(X)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but DtkTransformer is expecting "
                    f"{self.n_features_in_} features as input"
                )
        elif self.strict:
            _check_feature_names(self.feature_names_in_, X)
        return t.apply(X, t.parse(self.params), self.state_)

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
