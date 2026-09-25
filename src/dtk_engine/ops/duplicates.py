"""Duplicate rows: exact, partial (on a subset of identity columns), conflicts."""

from __future__ import annotations

import pandas as pd

from dtk_engine.ops.profile import columns_of_type

ID_TYPES = ("id_like", "group_id")

GROUP_COLUMN = "duplicate_group"
CONFLICT_COLUMNS = ["column", "n_groups", "pct_groups"]


def default_subset(df: pd.DataFrame) -> list[str]:
    """Detected identifier columns (id_like / group_id), the default identity."""
    return columns_of_type(df, *ID_TYPES)


def _hashable(df: pd.DataFrame) -> pd.DataFrame:
    """Copy where unhashable cells (lists, dicts) become strings."""
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, list | dict | set)).any():
            out[col] = out[col].astype(str)
    return out


def exact_duplicates(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """(n redundant rows, all rows involved in a duplicate set, keep=False)."""
    flat = _hashable(df)
    n_redundant = int(flat.duplicated().sum())
    involved = df[flat.duplicated(keep=False).to_numpy()]
    return n_redundant, involved


def duplicate_groups(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Rows sharing their ``subset`` values with another row (keep=False), plus a
    ``duplicate_group`` number, grouped together and sorted by group."""
    flat = _hashable(df)
    mask = flat.duplicated(subset=subset, keep=False).to_numpy()
    rows = df[mask].copy()
    if rows.empty:
        rows.insert(0, GROUP_COLUMN, pd.Series(dtype="int64"))
        return rows
    ids = flat[mask].groupby(subset, dropna=False, sort=False).ngroup() + 1
    rows.insert(0, GROUP_COLUMN, ids.to_numpy())
    return rows.sort_values(GROUP_COLUMN, kind="stable")


def conflicts(df: pd.DataFrame, subset: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same key, different values elsewhere.

    Returns (per-column table: in how many duplicate groups the column disagrees,
    the rows of groups with at least one disagreement).
    """
    groups = duplicate_groups(df, subset)
    empty = pd.DataFrame(columns=CONFLICT_COLUMNS)
    if groups.empty:
        return empty, groups
    n_groups = int(groups[GROUP_COLUMN].nunique())
    others = [c for c in df.columns if c not in subset]
    flat = _hashable(groups)
    disagree = pd.DataFrame(
        {c: flat.groupby(GROUP_COLUMN)[c].nunique(dropna=False) > 1 for c in others},
        index=pd.Index(sorted(flat[GROUP_COLUMN].unique()), name=GROUP_COLUMN),
    )
    if disagree.empty:
        return empty, groups.iloc[0:0]
    counts = disagree.sum().astype(int)
    table = pd.DataFrame(
        {
            "column": counts.index,
            "n_groups": counts.to_numpy(),
            "pct_groups": (100 * counts / n_groups).round(2).to_numpy(),
        }
    )
    table = table[table["n_groups"] > 0].sort_values("n_groups", ascending=False)
    bad = disagree.index[disagree.any(axis=1)]
    return table.reset_index(drop=True), groups[groups[GROUP_COLUMN].isin(bad)]
