"""Label join: attach y (labels) to X, by row order or on a key column.

Both modes refuse to lose or duplicate a row: any mismatch raises a
LabelJoinError whose message lists the counts.
"""

from __future__ import annotations

import pandas as pd

from dtk_engine.errors import KeyParamsError

# Column names read as a row index (e.g. `Index` in `y_train = Index,target`).
INDEX_NAMES = {"index", "idx", "unnamed: 0", ""}


class LabelJoinError(ValueError):
    """X and y cannot be joined without losing, duplicating or misaligning rows."""


class MergeJoinError(KeyParamsError):
    """A merge table cannot be joined onto X."""


def is_index_like(s: pd.Series) -> bool:
    """A row-index column: named like one, or integers 0..n-1 (or 1..n) in order."""
    if str(s.name).strip().lower() in INDEX_NAMES:
        return True
    if not pd.api.types.is_integer_dtype(s) or s.empty:
        return False
    values = s.to_numpy()
    start = values[0]
    return start in (0, 1) and bool(
        (values == pd.RangeIndex(start, start + len(s)).to_numpy()).all()
    )


def label_columns(y: pd.DataFrame, key: str | None = None) -> tuple[str | None, str]:
    """(index column or None, value column) of a y table joined by order."""
    columns = list(y.columns)
    if len(columns) == 1:
        return None, columns[0]
    if len(columns) == 2:
        if key is not None and key in columns:
            index = [key]
        else:
            index = [c for c in columns if is_index_like(y[c])]
            named = [c for c in index if str(c).strip().lower() in INDEX_NAMES]
            index = named or index
        if len(index) == 1:
            value = next(c for c in columns if c != index[0])
            return index[0], value
    raise LabelJoinError(
        "label join by order: y must have exactly one value column (plus an optional "
        f"key / index column), got {len(columns)} columns {columns}"
    )


def join_labels(
    x: pd.DataFrame, y: pd.DataFrame, mode: str, key: str | None = None
) -> pd.DataFrame:
    """X with y's value column(s) appended; ``mode`` is "order" or "key"."""
    if mode == "order":
        return _join_by_order(x, y, key)
    if mode == "key":
        if key is None:
            raise LabelJoinError("label join by key: a key column is required")
        return _join_on_key(x, y, key)
    raise LabelJoinError(
        f"unknown label join mode {mode!r} (expected 'order' or 'key')"
    )


def _check_no_clash(x: pd.DataFrame, value_columns: list) -> None:
    clash = [c for c in value_columns if c in x.columns]
    if clash:
        raise LabelJoinError(f"label join: y column(s) {clash} already exist in X")


def _join_by_order(x: pd.DataFrame, y: pd.DataFrame, key: str | None) -> pd.DataFrame:
    index, value = label_columns(y, key)
    if len(x) != len(y):
        raise LabelJoinError(
            f"label join by order: X has {len(x)} rows, y has {len(y)} rows"
        )
    if index is not None and index in x.columns:
        differ = x[index].to_numpy() != y[index].to_numpy()
        if differ.any():
            first = int(differ.argmax())
            raise LabelJoinError(
                f"label join by order: column {index!r} differs between X and y on "
                f"{int(differ.sum())} rows (first at row {first}); rows are not aligned"
            )
    _check_no_clash(x, [value])
    out = x.copy()
    out[value] = y[value].to_numpy()
    return out


def _join_on_key(x: pd.DataFrame, y: pd.DataFrame, key: str) -> pd.DataFrame:
    missing = [side for side, df in (("X", x), ("y", y)) if key not in df.columns]
    if missing:
        raise LabelJoinError(
            f"label join by key: column {key!r} not in {' and '.join(missing)}"
        )
    values = [c for c in y.columns if c != key]
    if not values:
        raise LabelJoinError(f"label join by key: y has no column besides {key!r}")
    _check_no_clash(x, values)
    counts = {
        "x_rows": len(x),
        "y_rows": len(y),
        "x_duplicated_keys": int(x[key].duplicated().sum()),
        "y_duplicated_keys": int(y[key].duplicated().sum()),
        "x_missing_keys": int(x[key].isna().sum()),
        "y_missing_keys": int(y[key].isna().sum()),
        "x_unmatched": int((~x[key].isin(y[key])).sum()),
        "y_unmatched": int((~y[key].isin(x[key])).sum()),
    }
    if any(v for k, v in counts.items() if not k.endswith("_rows")):
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        raise LabelJoinError(
            f"label join by key {key!r} is not one-to-one without row loss: {detail}"
        )
    out = x.merge(y, on=key, how="left", validate="one_to_one")
    out.index = x.index
    return out


def merge_match_stats(
    x: pd.DataFrame, merge_df: pd.DataFrame, key: str
) -> dict[str, int]:
    """Report ``{'matched': int, 'total': int}`` for merging ``merge_df`` onto ``x`` on ``key``."""
    missing = [
        side
        for side, df in (("X", x), ("merge table", merge_df))
        if key not in df.columns
    ]
    if missing:
        raise MergeJoinError(
            f"merge on key: column {key!r} not in {' and '.join(missing)}"
        )
    matched = int(x[key].isin(merge_df[key]).sum())
    return {"matched": matched, "total": len(x)}


def merge_table(
    x: pd.DataFrame,
    merge_df: pd.DataFrame,
    key: str,
    columns: list[str] | None = None,
    *,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Left join ``merge_df`` onto ``x`` on ``key`` (many-to-one).

    ``columns``: non-key columns to pull from ``merge_df`` (None = all non-key columns).
    The merge source must have unique non-null keys (else raises MergeJoinError listing
    the duplicated keys). X rows are never lost or duplicated (assert row count), X row
    order and index kept. Unmatched rows get NaN. A merged column clashing with an
    existing X column raises.
    """
    missing = [
        side
        for side, df in (("X", x), ("merge table", merge_df))
        if key not in df.columns
    ]
    if missing:
        raise MergeJoinError(
            f"merge on key: column {key!r} not in {' and '.join(missing)}"
        )

    null_keys = int(merge_df[key].isna().sum())
    if null_keys:
        raise MergeJoinError(
            f"merge table key {key!r} contains {null_keys} null value(s)"
        )

    dup_mask = merge_df[key].duplicated(keep=False)
    if dup_mask.any():
        dup_keys = merge_df.loc[dup_mask, key].dropna().unique().tolist()
        raise MergeJoinError(
            f"merge table key {key!r} has {len(dup_keys)} duplicated keys: {dup_keys}"
        )

    if columns is not None:
        missing_cols = [c for c in columns if c not in merge_df.columns]
        if missing_cols:
            raise MergeJoinError(
                f"merge: column(s) {missing_cols} not in merge table"
            )
        merge_cols = [c for c in dict.fromkeys(columns) if c != key]
    else:
        merge_cols = [c for c in merge_df.columns if c != key]

    clash = [c for c in merge_cols if c in x.columns]
    if clash:
        raise MergeJoinError(f"merge: column(s) {clash} already exist in X")

    matched = int(x[key].isin(merge_df[key]).sum())
    stats = {"matched": matched, "total": len(x)}

    if not merge_cols:
        out = x.copy()
    else:
        sub_merge = merge_df[[key, *merge_cols]]
        out = x.merge(sub_merge, on=key, how="left", validate="many_to_one")
        out.index = x.index
        assert len(out) == len(x), (
            f"merge altered row count: was {len(x)}, now {len(out)}"
        )

    if return_stats:
        return out, stats
    return out

