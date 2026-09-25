"""align_to_train: realign a shifted numeric column onto train's distribution."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from dtk_engine.transform_registry import TransformParams, transform


class AlignToTrainParams(TransformParams):
    columns: list[str] = Field(min_length=1, description="Numeric columns to align")
    mode: Literal["shift_mean", "shift_median", "standardize", "robust", "quantile"] = (
        Field(
            default="shift_mean",
            description="shift_mean / shift_median: move the frame's mean / median "
            "onto train's; standardize: also rescale std; robust: median + IQR; "
            "quantile: map onto train's quantile function",
        )
    )
    group: str | None = Field(
        default=None,
        description="Column: align per group (statistics fitted on train); unseen "
        "or small groups fall back to the global statistics",
    )
    min_rows: int = Field(
        default=30,
        ge=1,
        description="Fewer non-null frame values than this leaves the column "
        "(or group) unchanged",
    )
    on_small: Literal["skip", "raise"] = Field(
        default="skip", description="A column with fewer than min_rows values"
    )
    n_quantiles: int = Field(
        default=101, ge=2, description="Train quantiles kept for mode 'quantile'"
    )

    @field_validator("mode", mode="before")
    @classmethod
    def _mode_alias(cls, value: Any) -> Any:
        return "standardize" if value == "standardize_to_train" else value

    @model_validator(mode="after")
    def _group_not_aligned(self) -> AlignToTrainParams:
        if self.group is not None and self.group in self.columns:
            raise ValueError("group column cannot be one of the aligned columns")
        return self


def _num(x: float) -> float | None:
    return None if pd.isna(x) else float(x)


def _iqr(s: pd.Series) -> float:
    q1, q3 = s.quantile([0.25, 0.75])
    return q3 - q1


def _stats(s: pd.Series, n_quantiles: int | None) -> dict:
    """Location / scale statistics of a numeric series (JSON-safe, NaN -> null)."""
    n = int(s.notna().sum())
    out = {
        "n": n,
        "mean": _num(s.mean()),
        "std": _num(s.std()),
        "median": _num(s.median()),
        "iqr": _num(_iqr(s)),
    }
    if n_quantiles is not None:
        qs = s.quantile(np.linspace(0, 1, n_quantiles)) if n else None
        out["quantiles"] = None if qs is None else [float(v) for v in qs]
    return out


def _fit_align(df: pd.DataFrame, params: AlignToTrainParams) -> dict:
    nq = params.n_quantiles if params.mode == "quantile" else None
    nums = {c: pd.to_numeric(df[c], errors="raise") for c in params.columns}
    state: dict = {"global": {c: _stats(s, nq) for c, s in nums.items()}, "groups": {}}
    if params.group is not None:
        gser = df[params.group]
        for c, s in nums.items():
            state["groups"][c] = {
                str(key): _stats(sub, nq) for key, sub in s.groupby(gser, dropna=True)
            }
    return state


def _positive(x: float | None) -> bool:
    return x is not None and not pd.isna(x) and x > 0


def _align_series(s: pd.Series, ref: dict, mode: str) -> pd.Series:
    """Align ``s`` onto the reference statistics ``ref`` (one column or group)."""
    if mode == "quantile":
        qs = ref.get("quantiles")
        if qs is None:
            return s
        n = int(s.notna().sum())
        rank = s.rank(method="average")
        q = (rank - 1) / (n - 1) if n > 1 else rank * 0 + 0.5
        vals = np.interp(q.to_numpy(dtype=float), np.linspace(0, 1, len(qs)), qs)
        return pd.Series(vals, index=s.index).where(s.notna())
    if mode in ("shift_median", "robust"):
        loc_f, scale_f = s.median(), _iqr(s)
        loc_t, scale_t = ref["median"], ref["iqr"]
    else:
        loc_f, scale_f = s.mean(), s.std()
        loc_t, scale_t = ref["mean"], ref["std"]
    if loc_t is None or pd.isna(loc_f):
        return s
    centered = s - loc_f
    if mode in ("standardize", "robust") and _positive(scale_f) and _positive(scale_t):
        centered = centered / scale_f * scale_t
    return centered + loc_t


@transform(
    "align_to_train",
    params_model=AlignToTrainParams,
    fit=_fit_align,
    title="Align to train statistics",
)
def align_to_train(
    df: pd.DataFrame, params: AlignToTrainParams, state: dict
) -> pd.DataFrame:
    """Realign a shifted / broken numeric column onto train's distribution.

    Modes: ``shift_mean`` (x - mean + train mean), ``shift_median``,
    ``standardize`` (also rescale std; ``standardize_to_train`` is an accepted
    alias), ``robust`` (median and IQR) and ``quantile`` (empirical quantile
    within the frame -> train's quantile function, ``n_quantiles`` train
    quantiles, average ranks for ties, NaN kept).

    State holds train's statistics (global and, with ``group``, per group);
    the frame's own statistics are measured at apply time, so applying to
    train itself is the identity (approximately so for ``quantile``). A zero
    scale (std / IQR) on either side means shift only. A column with fewer
    than ``min_rows`` non-null values is left unchanged (``on_small="raise"``
    raises). With ``group``, a group unseen in train or with fewer than
    ``min_rows`` rows on either side uses the global statistics.
    """
    missing = [c for c in [*params.columns, params.group] if c and c not in df]
    if missing:
        raise KeyError(f"columns not in frame: {missing}")
    out = df.copy()
    gvals = None if params.group is None else out[params.group].reset_index(drop=True)
    for col in params.columns:
        s = pd.to_numeric(out[col], errors="raise").astype(float)
        n = int(s.notna().sum())
        if n < params.min_rows:
            if params.on_small == "raise":
                raise ValueError(
                    f"column {col!r}: {n} non-null values < min_rows={params.min_rows}"
                )
            continue
        s = s.reset_index(drop=True)
        aligned = _align_series(s, state["global"][col], params.mode)
        if gvals is not None:
            groups = state["groups"].get(col, {})
            for key, sub in s.groupby(gvals, dropna=True):
                ref = groups.get(str(key))
                if (
                    ref is None
                    or ref["n"] < params.min_rows
                    or int(sub.notna().sum()) < params.min_rows
                ):
                    continue
                aligned.loc[sub.index] = _align_series(sub, ref, params.mode)
        out[col] = aligned.to_numpy()
    return out
