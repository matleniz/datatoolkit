"""In-memory workspace inspection for the Studio grid (rows, profiles, diffs).

Workspaces are passed as unsaved dicts (validated like ``preview_workspace``);
nothing is written to the store. Row identity ``_rid`` is the row's position in
the raw frame of that role, carried on the DataFrame index so row-dropping
steps (``drop_duplicates``, ``filter_rows``, ``drop_missing_target``) keep it.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from pydantic import ValidationError

from dtk_engine.cache import memoize
from dtk_engine.errors import KeyParamsError, SourceError, key_params_from_validation
from dtk_engine.ops._util import json_scalar, py
from dtk_engine.ops.compare import schema_diff
from dtk_engine.ops.compare.schema import (
    CATEGORY_TYPES,
    category_shift,
    value_mismatch_is_blocking,
)
from dtk_engine.ops.consistency import DATE_LIKE_MIN_SHARE, date_format
from dtk_engine.ops.consistency import normalize as _normalize_text
from dtk_engine.ops.join import label_columns
from dtk_engine.ops.missing import sentinel_counts
from dtk_engine.ops.outliers import univariate_outliers
from dtk_engine.ops.profile import (
    HIST_BINS,
    hashable,
    numeric_text_format,
    object_kind,
    semantic_type,
)
from dtk_engine.ops.suggested import suggested_params as _suggested_params
from dtk_engine.sources import load
from dtk_engine.workspace.dataset import cached_frame, raw_workspace_frame
from dtk_engine.workspace.models import Step, Workspace
from dtk_engine.workspace.replay import (
    needs_train,
    replay,
    replay_step,
    resolve_version,
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
# Name heuristic: text column named like an id with high distinctness -> identifier.
# semantic_type remains the primary signal (id_like / group_id already map above).
_ID_NAME_DISTINCT_RATIO = 0.9
# Numbers stored as text: optional sign, digits, optional decimal with '.' or ','
# (thousands-free; a single ',' is treated as the decimal separator).
_NUMBER_AS_TEXT_RE = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")


def _name_looks_like_id(name: str) -> bool:
    return name == "id" or name.endswith(("_id", "Id"))


def column_kind(series: pd.Series, name: str | None = None) -> str:
    """Map a column to a Studio grid kind (see module docstring table).

    ``name`` enables a secondary heuristic: a text column named ``id`` or ending
    in ``_id`` / ``Id`` with >= 90% distinct non-null values becomes
    ``identifier``. ``semantic_type`` remains the primary signal.
    """
    st = semantic_type(series)
    if st != "constant":
        kind = _SEMANTIC_TO_KIND[st]
    elif pdt.is_bool_dtype(series):
        kind = KIND_BOOL
    elif pdt.is_datetime64_any_dtype(series):
        kind = KIND_DATE
    elif pdt.is_numeric_dtype(series):
        kind = KIND_NUMBER
    else:
        obj = object_kind(series.dropna())
        kind = KIND_BINARY if obj == "binary" else KIND_TEXT
    if kind == KIND_TEXT and name is not None and _name_looks_like_id(name):
        values = series.dropna()
        if len(values) > 0:
            ratio = hashable(values).nunique() / len(values)
            if ratio >= _ID_NAME_DISTINCT_RATIO:
                return KIND_IDENTIFIER
    return kind


def _check_role(role: str) -> None:
    if role not in ("train", "test"):
        raise KeyParamsError(f"role must be 'train' or 'test', got {role!r}")


def _with_rids(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``frame`` indexed by stable raw-row positions 0..n-1."""
    out = frame.copy()
    out.index = pd.RangeIndex(len(out), name="_rid")
    return out


def _raw_role(ws: Workspace, role: str, labeled: bool = True) -> pd.DataFrame:
    return cached_frame(
        ws,
        "raw_rid",
        role,
        [],
        labeled,
        lambda: _with_rids(raw_workspace_frame(ws, role, labeled)),
    )


def _replay_role(
    ws: Workspace, role: str, steps: list[Step], labeled: bool = True
) -> pd.DataFrame:
    """Replay ``steps`` on ``role``, preserving ``_rid`` on the index."""

    def compute() -> pd.DataFrame:
        frame = _raw_role(ws, role, labeled)
        train = None
        if role == "test" and needs_train(steps):
            train = _raw_role(ws, "train", labeled)
        return replay(steps, role, frame, train)

    return cached_frame(ws, "replay_rid", role, steps, labeled, compute)


def _frame_at(
    ws: Workspace, role: str, version: int | None, labeled: bool = True
) -> tuple[pd.DataFrame, int]:
    n = resolve_version(len(ws.steps), version)
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
            "kind": column_kind(df[c], str(c)),
        }
        for c in df.columns
    ]


def _resolve_columns(
    df: pd.DataFrame, columns: list[str] | None, op: str
) -> list[str] | None:
    """Non-empty ``columns`` → names in that order; unknown → KeyParamsError.

    ``None`` / empty list → ``None`` (caller keeps all-columns behaviour).
    """
    if not columns:
        return None
    if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
        raise KeyParamsError(
            f"{op}: columns must be a list of strings, got {columns!r}"
        )
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyParamsError(f"{op}: columns not in the frame {absent}")
    return list(dict.fromkeys(columns))


def workspace_rows(
    ws: Workspace,
    role: str,
    version: int | None = None,
    offset: int = 0,
    limit: int = 500,
    columns: list[str] | None = None,
) -> dict:
    """Paged rows for ``role`` at ``version`` (None = all steps replayed).

    Optional ``columns`` (non-empty) restricts the page and column meta to those
    names in that order; unknown names raise ``KeyParamsError``.
    """
    _check_role(role)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise KeyParamsError(f"offset must be a non-negative int, got {offset!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise KeyParamsError(f"limit must be a positive int, got {limit!r}")
    df, n = _frame_at(ws, role, version)
    picked = _resolve_columns(df, columns, "workspace_rows")
    view = df[picked] if picked is not None else df
    page = view.iloc[offset : offset + limit]
    return {
        "columns": _column_meta(view),
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
    """True when every non-null cell is a string number ('.' or ',' decimal)."""
    if pdt.is_numeric_dtype(series) or pdt.is_bool_dtype(series):
        return False
    if pdt.is_datetime64_any_dtype(series) or object_kind(series.dropna()):
        return False
    values = series.dropna()
    if values.empty:
        return False
    if not values.map(lambda v: isinstance(v, str)).all():
        return False
    return bool(
        values.map(lambda v: bool(_NUMBER_AS_TEXT_RE.fullmatch(v.strip()))).all()
    )


def _currency_as_text(series: pd.Series) -> dict | None:
    """``{decimal, thousands, percent}`` when the column is mostly numbers-as-text
    with a currency / percent / thousands marker ('$2.39', '65.9567%',
    '1 200,50', '990,-'), else None (see ``ops.profile.numeric_text_format``).

    Field kept as ``currency_as_text`` (MAT-168 widened detection beyond
    currency; fronts still key off this name)."""
    if pdt.is_datetime64_any_dtype(series) or object_kind(series.dropna()):
        return None
    return numeric_text_format(series)


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


@memoize()
def _profile_one(name: str, series: pd.Series) -> dict:
    kind = column_kind(series, name)
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
        # Identifier values (Index 0..n, patient ids) often include sentinel-looking
        # numbers like 999 / -1; those are real ids, not missing-value markers.
        "sentinel_candidates": (
            [] if kind == KIND_IDENTIFIER else _sentinel_candidates(series)
        ),
        "distinct": distinct,
        "histogram": hist,
        "top_values": top,
        "iqr_bounds": iqr_bounds,
        "outliers": outliers,
        "variants": _variants(series),
        "looks_like_dates": _looks_like_dates(series),
        "numbers_as_text": _numbers_as_text(series),
        "currency_as_text": _currency_as_text(series),
        "skewed": _skewed(series),
        "suggested_params": _suggested_params(series, kind=kind),
    }


def column_profiles(
    ws: Workspace,
    role: str,
    version: int | None = None,
    columns: list[str] | None = None,
) -> dict:
    """Per-column profile for ``role`` at ``version`` (None = all steps).

    Optional ``columns`` (non-empty) profiles only those names in that order;
    unknown names raise ``KeyParamsError``. ``None`` / empty = every column.
    """
    _check_role(role)
    df, n = _frame_at(ws, role, version)
    picked = _resolve_columns(df, columns, "column_profiles")
    names = picked if picked is not None else [str(c) for c in df.columns]
    return {
        "columns": [_profile_one(name, df[name]) for name in names],
        "version": n,
    }


def _parse_step(step: dict) -> Step:
    try:
        return Step.model_validate(step)
    except ValidationError as exc:
        raise key_params_from_validation(exc) from exc


def _candidate_mask(bc: pd.Series, ac: pd.Series) -> np.ndarray:
    """Cells that *may* differ (no false negatives), vectorised.

    A cell is excluded only when both sides are null or compare equal, which
    implies equal ``_cell_json`` output. Callers confirm candidates exactly.
    """
    all_cells = np.ones(len(bc), dtype=bool)
    if bc.dtype != ac.dtype and not (
        pdt.is_numeric_dtype(bc.dtype) and pdt.is_numeric_dtype(ac.dtype)
    ):
        # e.g. tz-aware vs naive, object vs typed: equality may not match JSON.
        return all_cells
    try:
        differs = bc.ne(ac).fillna(True).to_numpy(dtype=bool)
    except (TypeError, ValueError):
        return all_cells
    return differs & ~(bc.isna() & ac.isna()).to_numpy(dtype=bool)


def _diff_cells(
    before: pd.DataFrame, after: pd.DataFrame
) -> tuple[list[dict], int, list[int]]:
    """Changed cells (capped), total count, and removed ``_rid``s.

    Equality is JSON round-trip equality (``_cell_json``), so int 5 and
    float 5.0 are the same cell. A vectorised pre-filter narrows the cells
    to those worth serialising; they are then confirmed with ``_cell_json``
    in row-major order (rid, then column).
    """
    common_cols = [c for c in before.columns if c in after.columns]
    if before.index.equals(after.index) and before.index.is_monotonic_increasing:
        # Common case (no row dropped): skip the set algebra and the reindex.
        removed: list[int] = []
        common_rids = before.index.to_numpy()
        b, a = before[common_cols], after[common_cols]
    else:
        before_rids = set(map(int, before.index))
        after_rids = set(map(int, after.index))
        removed = sorted(before_rids - after_rids)
        common_rids = sorted(before_rids & after_rids)
        b = before.loc[common_rids, common_cols]
        a = after.loc[common_rids, common_cols]
    # (row position, column position, before, after) of each confirmed cell.
    # Column-wise: one array per column, not a boxed ``.iat`` per cell (MAT-212).
    found: list[tuple[int, int, Any, Any]] = []
    if len(common_rids) and common_cols:
        for j in range(len(common_cols)):
            bc, ac = b.iloc[:, j], a.iloc[:, j]
            rows = np.flatnonzero(_candidate_mask(bc, ac))
            if not len(rows):
                continue
            bvals, avals = bc.array.take(rows), ac.array.take(rows)
            for i, bv, av in zip(rows.tolist(), bvals, avals, strict=True):
                bj, aj = _cell_json(bv), _cell_json(av)
                if bj != aj:
                    found.append((i, j, bj, aj))
    found.sort(key=lambda cell: (cell[0], cell[1]))  # row-major: rid, then column
    changed = [
        {
            "_rid": int(common_rids[i]),
            "column": str(common_cols[j]),
            "before": bj,
            "after": aj,
        }
        for i, j, bj, aj in found[:_CHANGED_CAP]
    ]
    return changed, len(found), removed


def preview_step(ws: Workspace, step: dict | Step, role: str) -> dict:
    """Append ``step``, replay in memory, return shape / column / cell diffs.

    Invalid step params / unknown op -> ``KeyParamsError`` /
    ``UnknownTransformError`` (same as ``save_workspace``). A step that fails
    on the data -> ``SourceError`` naming the step.
    """
    _check_role(role)
    parsed_step = step if isinstance(step, Step) else _parse_step(step)
    validate_steps([*ws.steps, parsed_step])

    # Only the pending step is fitted here: the frames after the saved steps come
    # from the replay-result cache, so a keystroke costs one step, not N (MAT-212).
    before = _replay_role(ws, role, ws.steps)
    # Shallow copy: copy-on-write keeps ``before`` intact whatever the op does.
    prefix = before.copy(deep=False)
    target = parsed_step.target
    train = None
    if target != "test" or role == "train":
        train = prefix if role == "train" else _replay_role(ws, "train", ws.steps)
    test = None
    if ws.datasets.test is not None and (target != "train" or role == "test"):
        test = prefix if role == "test" else _replay_role(ws, "test", ws.steps)
    train_after, test_after, last = replay_step(parsed_step, len(ws.steps), train, test)
    after = train_after if role == "train" else test_after
    if after is None:
        raise SourceError(f"workspace {ws.name!r} has no {role} dataset")

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
        "kind": column_kind(col, name),
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


def _value_set_fields(train_col: pd.Series, test_col: pd.Series) -> dict | None:
    """Flag when a categorical / label column has test-only values."""
    shift = category_shift(train_col, test_col)
    if shift["n_unseen_categories"] == 0:
        return None
    near = shift["_near_matches"]
    pct = shift["pct_test_rows_unseen"]
    return {
        "only_in_test": shift["_only_in_test"],
        "pct_test_rows_unseen": pct,
        "near_match_hint": shift["near_match_hint"],
        "near_matches": near,
        "blocking": value_mismatch_is_blocking(near, pct),
    }


def _align_row(
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    *,
    status: str,
    similar: list[str] | None = None,
    value_set: dict | None = None,
) -> dict:
    train_col = train[name] if name in train.columns else None
    test_col = test[name] if name in test.columns else None
    out = {
        "train": _side_info(train, name),
        "test": _side_info(test, name),
        "status": status,
        "numbers_as_text": any(
            col is not None and _numbers_as_text(col) for col in (train_col, test_col)
        ),
        "train_mean": _mean(train_col) if train_col is not None else None,
        "test_mean": _mean(test_col) if test_col is not None else None,
        "similar": similar if similar is not None else [],
        "only_in_test": None,
        "pct_test_rows_unseen": None,
        "near_match_hint": None,
        "near_matches": [],
        "blocking": False,
    }
    if value_set is not None:
        out.update(value_set)
    return out


def _align_column(
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    label: str | None,
    test_only: list[str],
) -> dict:
    """The alignment row of one train column: its status decides the fields."""
    if name not in test.columns:
        if name == label:
            return _align_row(train, test, name, status="label")
        return _align_row(
            train,
            test,
            name,
            status="missing_in_test",
            similar=_similar_names(name, test_only),
        )
    train_col, test_col = train[name], test[name]
    if not _kinds_match(column_kind(train_col, name), column_kind(test_col, name)):
        return _align_row(train, test, name, status="type_mismatch")
    check_values = name == label or {
        semantic_type(train_col),
        semantic_type(test_col),
    } & set(CATEGORY_TYPES)
    value_set = _value_set_fields(train_col, test_col) if check_values else None
    status = "match" if value_set is None else "value_mismatch"
    return _align_row(train, test, name, status=status, value_set=value_set)


def align_report(ws: Workspace) -> dict:
    """Train / test column alignment after every workspace step.

    The front typically sends only alignment steps; we replay whatever is in
    ``ws.steps``. Per column: train/test side info, status, means, similar
    names for missing-in-test rows. Categorical / label columns present on
    both sides with test-only values get status ``value_mismatch`` plus
    ``only_in_test`` (value+count), ``pct_test_rows_unseen``, optional
    ``near_match_hint`` / ``near_matches`` (strip / casefold / trailing
    punctuation), and ``blocking`` (Studio "to decide" when near_matches
    exist or pct unseen exceeds 50). No test dataset -> ``SourceError``.
    """
    if ws.datasets.test is None:
        raise SourceError(f"workspace {ws.name!r} has no test dataset")
    train = _replay_role(ws, "train", ws.steps)
    test = _replay_role(ws, "test", ws.steps)

    label = _label_name(ws)
    test_only = list(schema_diff(train, test)["only_test"])
    rows = [
        _align_column(train, test, name, label, test_only)
        for name in map(str, train.columns)
    ]
    rows += [
        _align_row(train, test, name, status="extra_in_test") for name in test_only
    ]
    return {"columns": rows}
