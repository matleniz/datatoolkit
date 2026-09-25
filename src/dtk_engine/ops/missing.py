"""Missing-value analysis: rates, per-row counts, sentinels, co-occurrence, test spikes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops._util import pct as _pct
from dtk_engine.ops.profile import object_kind

# Missing rate (%) above which dropping the column is the default advice.
DROP_PCT = 60.0
REVIEW_PCT = 20.0

# A per-row missing-count k is a spike when its bin holds >= SPIKE_MIN_SHARE of the
# rows and >= SPIKE_FACTOR x the mean of its neighbours (k - 1, k + 1): a failed
# batch or an unjoined source misses the same fields on a block of rows.
SPIKE_MIN_SHARE = 0.02
SPIKE_FACTOR = 3.0

NUMERIC_SENTINELS = (-9999, -999, -99, -1, 999, 9999)
DATE_SENTINELS = ("1900-01-01", "1970-01-01", "0001-01-01")
STRING_SENTINELS = frozenset(
    {
        "",
        "unknown",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
        "nil",
        "-",
        "--",
        "?",
        "missing",
    }
)
# 0 is only suspicious in a continuous-looking column where it dwarfs every other value.
ZERO_MIN_UNIQUE = 20
ZERO_MIN_ROWS = 5
ZERO_DOMINANCE = 3.0

COOCCURRENCE_MIN_JACCARD = 0.5
COOCCURRENCE_MAX_COLUMNS = 30

# Test-imputation spike: a single value taking >= SPIKE_TEST_MIN_ROWS rows and
# >= SPIKE_TEST_MIN_SHARE of the test set while being >= SPIKE_TEST_RATIO x more
# common than in train (a column that is continuous in train: >= ZERO_MIN_UNIQUE values).
SPIKE_TEST_MIN_ROWS = 10
SPIKE_TEST_MIN_SHARE = 0.01
SPIKE_TEST_RATIO = 10.0

RATE_FIELDS = ["column", "n_missing", "pct_missing", "recommendation"]
SENTINEL_FIELDS = ["column", "sentinel", "count", "pct"]
SPIKE_FIELDS = ["column", "value", "n_test", "pct_test", "pct_train"]


def missing_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Missing rate per column, worst first, with a drop / review / keep advice."""
    n = len(df)
    counts = df.isna().sum()
    rows = []
    for col, count in counts.items():
        pct = _pct(count, n)
        if pct >= DROP_PCT:
            advice = (
                f"drop the column (>= {DROP_PCT:.0f}% missing) unless the "
                "missingness itself is predictive (keep an is-missing flag then)"
            )
        elif pct >= REVIEW_PCT:
            advice = "impute, and consider an is-missing indicator"
        elif count:
            advice = "impute"
        else:
            advice = "complete"
        rows.append((str(col), int(count), pct, advice))
    out = pd.DataFrame(rows, columns=RATE_FIELDS)
    return out.sort_values("pct_missing", ascending=False, kind="stable").reset_index(
        drop=True
    )


def missing_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Histogram of "number of missing fields in a row": n_missing, n_rows, pct, spike."""
    counts = df.isna().sum(axis=1).value_counts().sort_index()
    full = counts.reindex(range(df.shape[1] + 1), fill_value=0)
    n = len(df)
    values = full.to_numpy()
    spike = []
    for k, c in enumerate(values):
        neighbours = [values[j] for j in (k - 1, k + 1) if 0 <= j < len(values)]
        base = float(np.mean(neighbours)) if neighbours else 0.0
        spike.append(
            bool(
                k > 0
                and n
                and c / n >= SPIKE_MIN_SHARE
                and c >= SPIKE_FACTOR * max(base, 1.0)
            )
        )
    return pd.DataFrame(
        {
            "n_missing": full.index,
            "n_rows": values,
            "pct_rows": [_pct(c, n) for c in values],
            "spike": spike,
        }
    )


def _sentinel_hits(series: pd.Series) -> dict[str, int]:
    non_null = series.dropna()
    hits: dict[str, int] = {}
    if object_kind(non_null):
        # Lists / dicts / bytes (WKB, blobs) hold no textual sentinel.
        return hits
    if pdt.is_datetime64_any_dtype(series):
        stamps = non_null.dt.strftime("%Y-%m-%d")
        for s in DATE_SENTINELS:
            if c := int((stamps == s).sum()):
                hits[s] = c
        return hits
    if pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series):
        numbers = non_null.astype(float)
    else:
        text = non_null.astype(str).str.strip()
        lowered = text.str.lower()
        for s in sorted(STRING_SENTINELS):
            if c := int((lowered == s).sum()):
                hits[repr(s) if s == "" else s] = c
        for s in DATE_SENTINELS:
            if c := int(text.str.startswith(s).sum()):
                hits[s] = c
        numbers = pd.to_numeric(text, errors="coerce").dropna()
    for s in NUMERIC_SENTINELS:
        if c := int((numbers == s).sum()):
            hits[str(s)] = c
    zeros = int((numbers == 0).sum())
    if zeros >= ZERO_MIN_ROWS and numbers.nunique() >= ZERO_MIN_UNIQUE:
        other = numbers[numbers != 0].value_counts()
        if zeros >= ZERO_DOMINANCE * (int(other.iloc[0]) if len(other) else 0):
            hits["0"] = zeros
    return hits


def sentinel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Suspected disguised-missing values per column: -999/-99/-1, implausible 0,
    1900-01-01, "unknown", "N/A", "-", "" ... (counts over non-null cells)."""
    rows = []
    for col in df.columns:
        n = int(df[col].notna().sum())
        for sentinel, count in _sentinel_hits(df[col]).items():
            rows.append((str(col), sentinel, count, _pct(count, n)))
    out = pd.DataFrame(rows, columns=SENTINEL_FIELDS)
    return out.sort_values("count", ascending=False, kind="stable").reset_index(
        drop=True
    )


def cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Jaccard similarity of the missing masks of the columns that have gaps
    (square matrix, columns x columns; 1 = always missing together)."""
    gaps = df.isna()
    cols = [c for c in df.columns if gaps[c].any()]
    cols = sorted(cols, key=lambda c: -int(gaps[c].sum()))[:COOCCURRENCE_MAX_COLUMNS]
    mask = gaps[cols].to_numpy(dtype=float)
    both = mask.T @ mask
    counts = mask.sum(axis=0)
    union = counts[:, None] + counts[None, :] - both
    jac = np.divide(both, union, out=np.zeros_like(both), where=union > 0)
    names = [str(c) for c in cols]
    return pd.DataFrame(jac.round(3), index=names, columns=names)


def cooccurrence_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    """Column pairs whose missing masks overlap (Jaccard >= 0.5), strongest first."""
    rows = [
        (a, b, float(matrix.loc[a, b]))
        for i, a in enumerate(matrix.index)
        for b in matrix.columns[i + 1 :]
        if matrix.loc[a, b] >= COOCCURRENCE_MIN_JACCARD
    ]
    out = pd.DataFrame(rows, columns=["column_a", "column_b", "jaccard"])
    return out.sort_values("jaccard", ascending=False, kind="stable").reset_index(
        drop=True
    )


def value_spikes_vs_train(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Values over-represented in test vs train in continuous-looking columns
    (typical of a constant imputed on the test set only)."""
    rows = []
    for col in [c for c in train.columns if c in set(test.columns)]:
        tr, te = train[col].dropna(), test[col].dropna()
        if tr.empty or te.empty or tr.nunique() < ZERO_MIN_UNIQUE:
            continue
        te_counts = te.value_counts()
        tr_share = tr.value_counts(normalize=True)
        for value, count in te_counts.head(3).items():
            share = count / len(test)
            base = float(tr_share.get(value, 0.0))
            if (
                count >= SPIKE_TEST_MIN_ROWS
                and share >= SPIKE_TEST_MIN_SHARE
                and share >= SPIKE_TEST_RATIO * max(base, 1 / len(tr))
            ):
                rows.append(
                    (
                        str(col),
                        str(value),
                        int(count),
                        _pct(count, len(test)),
                        round(100 * base, 2),
                    )
                )
    out = pd.DataFrame(rows, columns=SPIKE_FIELDS)
    return out.sort_values("n_test", ascending=False, kind="stable").reset_index(
        drop=True
    )
