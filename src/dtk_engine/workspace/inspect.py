"""In-memory workspace inspection for the Studio grid (rows, profiles, diffs).

Workspaces are passed as unsaved dicts (validated like ``preview_workspace``);
nothing is written to the store. Row identity ``_rid`` is the row's position in
the raw frame of that role, carried on the DataFrame index so row-dropping
steps (``drop_duplicates``, ``filter_rows``, ``drop_missing_target``) keep it.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pydantic import ValidationError

from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.ops._util import json_scalar, py
from dtk_engine.ops.compare import schema_diff
from dtk_engine.ops.consistency import DATE_LIKE_MIN_SHARE, date_format
from dtk_engine.ops.consistency import normalize as _normalize_text
from dtk_engine.ops.join import label_columns
from dtk_engine.ops.missing import sentinel_counts
from dtk_engine.ops.outliers import univariate_outliers
from dtk_engine.ops.profile import (
    HIST_BINS,
    hashable,
    object_kind,
    pct_numeric_parsable,
    semantic_type,
)
from dtk_engine.sources import load
from dtk_engine.sources.dataset import labeled_frame
from dtk_engine.workspace.models import Step, Workspace
from dtk_engine.workspace.replay import (
    needs_train,
    replay,
    replay_fitted,
    validate_steps,
)

# Grid ``kind`` from ``ops.profile.semantic_type`` (plus dtype fallbacks for
# ``constant``). Fronts use these labels; do not invent others.
#
#   semantic_type   -> kind
#   numeric         -> number
#   binary          -> binary
#   boolean         -> bool
#   datetime        -> date
#   id_like         -> identifier
#   group_id        -> identifier
#   text            -> text
#   categorical     -> text
#   nested          -> text
#   constant        -> number | bool | date | binary | text (from dtype / cells)
KIND_NUMBER = "number"
KIND_BINARY = "binary"
KIND_TEXT = "text"
KIND_DATE = "date"
KIND_IDENTIFIER = "identifier"
KIND_BOOL = "bool"

_SEMANTIC_TO_KIND = {
    "numeric": KIND_NUMBER,
    "binary": KIND_BINARY,
    "boolean": KIND_BOOL,
    "datetime": KIND_DATE,
    "id_like": KIND_IDENTIFIER,
    "group_id": KIND_IDENTIFIER,
    "text": KIND_TEXT,
    "categorical": KIND_TEXT,
    "nested": KIND_TEXT,
}

_CHANGED_CAP = 2000
_TOP_VALUES = 20
_SAMPLE_N = 3
# Prototype skew heuristic: non-negative, median > 0, mean > 1.3 * median.
_SKEW_MEAN_OVER_MEDIAN = 1.3
_SKEW_MIN_N = 5


def column_kind(series: pd.Series) -> str:
    """Map a column to a Studio grid kind (see module docstring table)."""
    st = semantic_type(series)
    if st != "constant":
        return _SEMANTIC_TO_KIND[st]
    if pdt.is_bool_dtype(series):
        return KIND_BOOL
    if pdt.is_datetime64_any_dtype(series):
        return KIND_DATE
    if pdt.is_numeric_dtype(series):
        return KIND_NUMBER
    kind = object_kind(series.dropna())
    if kind == "binary":
        return KIND_BINARY
    return KIND_TEXT


def _check_role(role: str) -> None:
    if role not in ("train", "test"):
        raise KeyParamsError(f"role must be 'train' or 'test', got {role!r}")


def _resolve_version(n_steps: int, version: int | None) -> int:
    if version is None:
        return n_steps
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise KeyParamsError(f"version must be a non-negative int or None, got {version!r}")
    if version > n_steps:
        raise KeyParamsError(
            f"version {version} exceeds workspace step count ({n_steps})"
        )
    return version


def _with_rids(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` indexed by stable raw-row positions 0..n-1."""
    out = frame.copy()
    out.index = pd.RangeIndex(len(out), name="_rid")
    return out


def _raw_role(ws: Workspace, role: str, labeled: bool = True) -> pd.DataFrame:
    dataset = getattr(ws.datasets, role)
    if dataset is None:
        raise SourceError(f"workspace {ws.name!r} has no {role} dataset")
    return _with_rids(labeled_frame(dataset, ws.label, labeled))


def _replay_role(
    ws: Workspace, role: str, steps: list[Step], labeled: bool = True
) -> pd.DataFrame:
    """Replay ``steps`` on ``role``, preserving ``_rid`` on the index."""
    frame = _raw_role(ws, role, labeled)
    train = None
    if role == "test" and needs_train(steps):
        train = _raw_role(ws, "train", labeled)
    return replay(steps, role, frame, train)


def _frame_at(
    ws: Workspace, role: str, version: int | None, labeled: bool = True
) -> tuple[pd.DataFrame, int]:
    n = _resolve_version(len(ws.steps), version)
    return _replay_role(ws, role, ws.steps[:n], labeled), n


def _cell_json(value: object) -> Any:
    """JSON-safe cell: NaN -> null, datetimes -> ISO, numpy scalars unwrapped."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if pdt.is_scalar(value) and pd.isna(value):
        return None
    return json_scalar(py(value))


def _records(df: pd.DataFrame) -> list[dict]:
    """Row dicts with ``_rid``; NaN -> null, datetimes -> ISO strings."""
    if df.empty:
        return []
    # to_json handles NaN / timestamps; then stamp _rid from the preserved index.
    payload = json.loads(df.to_json(orient="records", date_format="iso"))
    for row, rid in zip(payload, df.index, strict=True):
        row["_rid"] = int(rid)
    return payload


def _column_meta(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "name": str(c),
            "dtype": str(df[c].dtype),
            "kind": column_kind(df[c]),
        }
        for c in df.columns
    ]


def workspace_rows(
    ws: Workspace,
    role: str,
    version: int | None = None,
    offset: int = 0,
    limit: int = 500,
) -> dict:
    """Paged rows for ``role`` at ``version`` (None = all steps replayed)."""
    _check_role(role)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise KeyParamsError(f"offset must be a non-negative int, got {offset!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise KeyParamsError(f"limit must be a positive int, got {limit!r}")
    df, n = _frame_at(ws, role, version)
    page = df.iloc[offset : offset + limit]
    return {
        "columns": _column_meta(df),
        "rows": _records(page),
        "total": len(df),
        "version": n,
    }


def _sentinel_candidates(series: pd.Series) -> list[dict]:
    table = sentinel_counts(series.to_frame("_"))
    if table.empty:
        return []
    return [
        {"value": json_scalar(py(r.sentinel)), "count": int(r.count)}
        for r in table.itertuples()
    ]


def _histogram(series: pd.Series) -> dict | None:
    values = series.dropna()
    if values.empty or not pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return None
    counts, edges = np.histogram(values.astype(float).to_numpy(), bins=HIST_BINS)
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def _top_values(series: pd.Series) -> list[dict]:
    counts = hashable(series).value_counts(dropna=False).head(_TOP_VALUES)
    out = []
    for value, count in counts.items():
        out.append({"value": _cell_json(value), "count": int(count)})
    return out


def _iqr_and_outliers(series: pd.Series) -> tuple[dict | None, int]:
    if not pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return None, 0
    if series.dropna().empty:
        return None, 0
    table = univariate_outliers(series.to_frame("_"), ["_"])
    row = table.iloc[0]
    lo, hi = row["lower_fence"], row["upper_fence"]
    if pd.isna(lo) or pd.isna(hi):
        return None, int(row["n_iqr"])
    return {"lo": float(lo), "hi": float(hi)}, int(row["n_iqr"])


def _variants(series: pd.Series) -> dict | None:
    """``{raw, normalized}`` distinct string counts after strip+lower, or null."""
    if not (pdt.is_object_dtype(series) or pdt.is_string_dtype(series)):
        return None
    strings = series.dropna()
    strings = strings[strings.map(lambda v: isinstance(v, str))]
    if strings.empty:
        return None
    raw = int(strings.nunique())
    normalized = int(strings.map(_normalize_text).nunique())
    if raw <= normalized:
        return None
    return {"raw": raw, "normalized": normalized}


def _looks_like_dates(series: pd.Series) -> bool:
    if pdt.is_datetime64_any_dtype(series):
        return False  # already a date column
    if not (pdt.is_object_dtype(series) or pdt.is_string_dtype(series)):
        return False
    strings = series.dropna()
    strings = strings[strings.map(lambda v: isinstance(v, str))]
    strings = strings[strings.str.strip() != ""]
    if strings.empty:
        return False
    dated = strings.map(date_format).dropna()
    return len(dated) >= DATE_LIKE_MIN_SHARE * len(strings)


def _numbers_as_text(series: pd.Series) -> bool:
    if pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return False
    if pdt.is_datetime64_any_dtype(series) or object_kind(series.dropna()):
        return False
    return bool(series.notna().any()) and pct_numeric_parsable(series) == 100.0


def _skewed(series: pd.Series) -> bool:
    if not pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return False
    values = series.dropna().astype(float)
    if len(values) < _SKEW_MIN_N:
        return False
    if float(values.min()) < 0:
        return False
    median = float(values.median())
    if median <= 0:
        return False
    return float(values.mean()) > _SKEW_MEAN_OVER_MEDIAN * median


def _profile_one(name: str, series: pd.Series) -> dict:
    kind = column_kind(series)
    n_missing = int(series.isna().sum())
    count = int(series.notna().sum())
    distinct = int(hashable(series).nunique(dropna=True))
    iqr_bounds, outliers = _iqr_and_outliers(series)
    hist = _histogram(series) if kind == KIND_NUMBER else None
    top = None if kind == KIND_NUMBER else _top_values(series)
    return {
        "name": name,
        "kind": kind,
        "count": count,
        "missing": n_missing,
        "sentinel_candidates": _sentinel_candidates(series),
        "distinct": distinct,
        "histogram": hist,
        "top_values": top,
        "iqr_bounds": iqr_bounds,
        "outliers": outliers,
        "variants": _variants(series),
        "looks_like_dates": _looks_like_dates(series),
        "numbers_as_text": _numbers_as_text(series),
        "skewed": _skewed(series),
    }


def column_profiles(
    ws: Workspace, role: str, version: int | None = None
) -> dict:
    """Per-column profile for ``role`` at ``version`` (None = all steps)."""
    _check_role(role)
    df, n = _frame_at(ws, role, version)
    return {
        "columns": [_profile_one(str(c), df[c]) for c in df.columns],
        "version": n,
    }


def _parse_step(step: dict) -> Step:
    try:
        return Step.model_validate(step)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc


def _diff_cells(
    before: pd.DataFrame, after: pd.DataFrame
) -> tuple[list[dict], int, list[int]]:
    """Changed cells (capped), total count, and removed ``_rid``s."""
    before_rids = set(map(int, before.index))
    after_rids = set(map(int, after.index))
    removed = sorted(before_rids - after_rids)
    common_rids = sorted(before_rids & after_rids)
    common_cols = [c for c in before.columns if c in after.columns]
    changed: list[dict] = []
    changed_total = 0
    if common_rids and common_cols:
        b = before.loc[common_rids, common_cols]
        a = after.loc[common_rids, common_cols]
        # Align dtypes for comparison via JSON round-trip equality.
        for rid in common_rids:
            for col in common_cols:
                bv, av = b.at[rid, col], a.at[rid, col]
                if _cell_json(bv) != _cell_json(av):
                    changed_total += 1
                    if len(changed) < _CHANGED_CAP:
                        changed.append(
                            {
                                "_rid": int(rid),
                                "column": str(col),
                                "before": _cell_json(bv),
                                "after": _cell_json(av),
                            }
                        )
    return changed, changed_total, removed


def preview_step(ws: Workspace, step: dict | Step, role: str) -> dict:
    """Append ``step``, replay in memory, return shape / column / cell diffs.

    Invalid step params / unknown op -> ``KeyParamsError`` /
    ``UnknownTransformError`` (same as ``save_workspace``). A step that fails
    on the data -> ``SourceError`` naming the step.
    """
    _check_role(role)
    parsed_step = step if isinstance(step, Step) else _parse_step(step)
    steps = [*ws.steps, parsed_step]
    validate_steps(steps)

    before = _replay_role(ws, role, ws.steps)
    train_raw = _raw_role(ws, "train")
    test_raw = None
    if ws.datasets.test is not None:
        test_raw = _raw_role(ws, "test")
    train_after, test_after, fitted = replay_fitted(steps, train_raw, test_raw)
    after = train_after if role == "train" else test_after
    if after is None:
        raise SourceError(f"workspace {ws.name!r} has no {role} dataset")

    last = fitted[-1]
    state = last["state"]
    # Ensure state is plain JSON (numpy scalars etc.).
    state_json = json.loads(json.dumps(state, default=str))

    before_cols = [str(c) for c in before.columns]
    after_cols = [str(c) for c in after.columns]
    added = [c for c in after_cols if c not in set(before_cols)]
    removed_cols = [c for c in before_cols if c not in set(after_cols)]
    changed, changed_total, removed_rids = _diff_cells(before, after)

    return {
        "shape": [len(after), after.shape[1]],
        "columns": after_cols,
        "added_columns": added,
        "removed_columns": removed_cols,
        "removed_rids": removed_rids,
        "changed": changed,
        "changed_total": changed_total,
        "state": state_json,
        "fitted_on": last["fitted_on"],
    }


def _label_name(ws: Workspace) -> str | None:
    train = ws.datasets.train
    if train.target_column is not None:
        return train.target_column
    if train.y is None:
        return None
    _, value = label_columns(load(train.y), ws.label.key)
    return value


def _samples(series: pd.Series) -> list[Any]:
    uniques = hashable(series.dropna()).unique()[:_SAMPLE_N]
    return [_cell_json(v) for v in uniques]


def _side_info(df: pd.DataFrame | None, name: str) -> dict | None:
    if df is None or name not in df.columns:
        return None
    col = df[name]
    return {
        "name": name,
        "kind": column_kind(col),
        "samples": _samples(col),
    }


def _mean(series: pd.Series) -> float | None:
    if not pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return None
    values = series.dropna()
    if values.empty:
        return None
    return round(float(values.mean()), 4)


def _similar_names(name: str, candidates: list[str]) -> list[str]:
    """Test-only column names ranked by ``difflib`` similarity to ``name``."""
    if not candidates:
        return []
    scored = sorted(
        candidates,
        key=lambda c: difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio(),
        reverse=True,
    )
    return scored


def _kinds_match(a: str, b: str) -> bool:
    if a == b:
        return True
    # Prototype: num and bin are compatible.
    numericish = {KIND_NUMBER, KIND_BINARY}
    return a in numericish and b in numericish


def align_report(ws: Workspace) -> dict:
    """Train / test column alignment after every workspace step.

    The front typically sends only alignment steps; we replay whatever is in
    ``ws.steps``. Per column: train/test side info, status, means, similar
    names for missing-in-test rows. No test dataset -> ``SourceError``.
    """
    if ws.datasets.test is None:
        raise SourceError(f"workspace {ws.name!r} has no test dataset")
    train = _replay_role(ws, "train", ws.steps)
    test = _replay_role(ws, "test", ws.steps)

    label = _label_name(ws)
    diff = schema_diff(train, test)
    test_only = list(diff["only_test"])
    rows: list[dict] = []

    def _row(
        name: str,
        *,
        status: str,
        similar: list[str] | None = None,
    ) -> dict:
        t_info = _side_info(train, name)
        e_info = _side_info(test, name)
        train_col = train[name] if name in train.columns else None
        test_col = test[name] if name in test.columns else None
        numbers_as_text = False
        if train_col is not None:
            numbers_as_text = numbers_as_text or _numbers_as_text(train_col)
        if test_col is not None:
            numbers_as_text = numbers_as_text or _numbers_as_text(test_col)
        return {
            "train": t_info,
            "test": e_info,
            "status": status,
            "numbers_as_text": numbers_as_text,
            "train_mean": _mean(train_col) if train_col is not None else None,
            "test_mean": _mean(test_col) if test_col is not None else None,
            "similar": similar if similar is not None else [],
        }

    for name in map(str, train.columns):
        if name in test.columns:
            tk, ek = column_kind(train[name]), column_kind(test[name])
            status = "match" if _kinds_match(tk, ek) else "type_mismatch"
            rows.append(_row(name, status=status))
        elif name == label:
            rows.append(_row(name, status="label"))
        else:
            rows.append(
                _row(
                    name,
                    status="missing_in_test",
                    similar=_similar_names(name, test_only),
                )
            )

    for name in test_only:
        rows.append(_row(name, status="extra_in_test"))

    return {"columns": rows}
