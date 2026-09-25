"""Column profiling: semantic types and a per-column summary table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from dtk_engine.ops._util import pct as _pct

SEMANTIC_TYPES = (
    "numeric",
    "categorical",
    "boolean",
    "datetime",
    "text",
    "id_like",
    "group_id",
    "constant",
    "nested",
    "binary",
)

# Distinct / non-null ratios: above ID_UNIQUE_RATIO an integer or spaceless string
# column is an identifier (tolerates a few duplicated rows); above TEXT_UNIQUE_RATIO
# a string column is free text rather than categories.
ID_UNIQUE_RATIO = 0.95
TEXT_UNIQUE_RATIO = 0.5
# Fewer non-null values than this: too little evidence to call a column id_like
# (a sparse column with a handful of distinct values is all-distinct by chance).
ID_MIN_NON_NULL = 20
# An integer column is identifier-shaped when its distinct values fill their
# [min, max] range densely (e.g. an `Index` 0..n-1); a sparse spread of distinct
# integers is a numeric feature.
ID_RANGE_DENSITY = 0.95
# group_id = entity key repeated over rows (e.g. `patient_id`): at least
# GROUP_MIN_UNIQUE distinct values with a distinct/non-null ratio in
# [GROUP_MIN_RATIO, ID_UNIQUE_RATIO). Strings must be whitespace-free; integers must
# fill at least GROUP_RANGE_DENSITY of their range (else they are a numeric feature).
GROUP_MIN_UNIQUE = 100
GROUP_MIN_RATIO = 0.01
GROUP_RANGE_DENSITY = 0.5
N_SAMPLES = 3
# Longer sample values (a WKB geometry, a big JSON list) are cut to this many chars.
SAMPLE_MAX_CHARS = 60

# Object cells that break hashing (nunique, duplicated, value_counts) or text
# conversion (bytes are not UTF-8): list / dict / ndarray from nested JSON or
# Parquet, bytes from GeoParquet WKB or blob columns.
NESTED_CELL_TYPES = (list, tuple, dict, set, frozenset, np.ndarray)
BINARY_CELL_TYPES = (bytes, bytearray, memoryview)


def object_kind(series: pd.Series) -> str | None:
    """The kind of an object column holding such cells ("nested" / "binary"), else None."""
    if series.dtype != object:
        return None
    values = series.dropna()
    if values.map(lambda v: isinstance(v, NESTED_CELL_TYPES)).any():
        return "nested"
    if values.map(lambda v: isinstance(v, BINARY_CELL_TYPES)).any():
        return "binary"
    return None


def _cell_key(value: object) -> object:
    if isinstance(value, np.ndarray):
        return repr(value.tolist())
    if isinstance(value, NESTED_CELL_TYPES):
        return repr(value)
    if isinstance(value, memoryview | bytearray):
        return repr(bytes(value))
    return value


def hashable(series: pd.Series) -> pd.Series:
    """`series` with nested cells replaced by their repr (hashable, comparable);
    bytes stay bytes. Unchanged (same object) when there is nothing to convert."""
    if object_kind(series) is None:
        return series
    return series.map(_cell_key)


def hashable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """`df` safe for duplicated / groupby / nunique (see `hashable`)."""
    if not any(object_kind(df[c]) for c in df.columns):
        return df
    out = df.copy()
    for i in range(out.shape[1]):
        out.isetitem(i, hashable(out.iloc[:, i]))
    return out


def as_text(series: pd.Series) -> pd.Series:
    """`series.astype(str)` that survives nested and non-UTF-8 bytes cells."""
    if object_kind(series) is None:
        return series.astype(str)
    return series.map(lambda v: str(_cell_key(v))).astype(str)


def _parses_as_datetime(strings: pd.Series) -> bool:
    # Pure digit strings ("2024", "7") are years / codes, not dates.
    if strings.str.fullmatch(r"\d+").all():
        return False
    parsed = pd.to_datetime(strings, errors="coerce", format="ISO8601")
    return bool(parsed.notna().all())


def _range_density(values: pd.Series, n_unique: int) -> float:
    span = int(values.max()) - int(values.min()) + 1
    return n_unique / span


def _group_band(n_unique: int, ratio: float) -> bool:
    return n_unique >= GROUP_MIN_UNIQUE and GROUP_MIN_RATIO <= ratio < ID_UNIQUE_RATIO


def semantic_type(series: pd.Series) -> str:
    """Classify a column into one of SEMANTIC_TYPES from its dtype and values."""
    values = series.dropna()
    kind = object_kind(values)
    n_unique = hashable(values).nunique()
    if n_unique <= 1:
        return "constant"
    if kind:
        return kind
    if pdt.is_bool_dtype(series):
        return "boolean"
    if pdt.is_datetime64_any_dtype(series):
        return "datetime"
    ratio = n_unique / len(values)
    enough = len(values) >= ID_MIN_NON_NULL
    if pdt.is_numeric_dtype(series):
        if n_unique == 2 and set(values.unique()) <= {0, 1}:
            return "boolean"
        if pdt.is_integer_dtype(series):
            density = _range_density(values, n_unique)
            if enough and ratio >= ID_UNIQUE_RATIO and density >= ID_RANGE_DENSITY:
                return "id_like"
            if _group_band(n_unique, ratio) and density >= GROUP_RANGE_DENSITY:
                return "group_id"
        return "numeric"
    strings = values.astype(str)
    if _parses_as_datetime(strings):
        return "datetime"
    spaceless = not strings.str.contains(r"\s").any()
    if enough and ratio >= ID_UNIQUE_RATIO and spaceless:
        return "id_like"
    if spaceless and _group_band(n_unique, ratio):
        return "group_id"
    if ratio > TEXT_UNIQUE_RATIO:
        return "text"
    return "categorical"


def pct_numeric_parsable(series: pd.Series) -> float:
    """% of non-null values that read as numbers (100 for a numeric dtype).

    A text column scoring high here is numbers polluted by a few tokens
    (e.g. "unknown" in an age column).
    """
    values = series.dropna()
    if values.empty or pdt.is_bool_dtype(series) or object_kind(values):
        return 0.0
    if pdt.is_numeric_dtype(series):
        return 100.0
    if pdt.is_datetime64_any_dtype(series):
        return 0.0
    parsed = pd.to_numeric(values.astype(str).str.strip(), errors="coerce")
    return round(100 * float(parsed.notna().mean()), 2)


def _short(value: object) -> str:
    text = str(_cell_key(value))
    if len(text) <= SAMPLE_MAX_CHARS:
        return text
    return text[: SAMPLE_MAX_CHARS - 3] + "..."


def _samples(series: pd.Series) -> str:
    uniques = hashable(series.dropna()).unique()[:N_SAMPLES]
    return ", ".join(_short(v) for v in uniques)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, semantic type, missing, uniques, sample values."""
    n_rows = len(df)
    rows = []
    for name in df.columns:
        col = df[name]
        n_missing = int(col.isna().sum())
        rows.append(
            {
                "column": str(name),
                "dtype": str(col.dtype),
                "semantic_type": semantic_type(col),
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n_rows, 2) if n_rows else 0.0,
                "n_unique": int(hashable(col).nunique()),
                "pct_numeric_parsable": pct_numeric_parsable(col),
                "sample_values": _samples(col),
            }
        )
    columns = [
        "column",
        "dtype",
        "semantic_type",
        "n_missing",
        "pct_missing",
        "n_unique",
        "pct_numeric_parsable",
        "sample_values",
    ]
    return pd.DataFrame(rows, columns=columns)


# --- Per-semantic-type statistics -------------------------------------------------

MISSING_LABEL = "(missing)"
# A categorical value below this share of the rows counts as "rare".
RARE_PCT = 1.0
OUTLIER_IQR_K = 1.5
OUTLIER_Z = 3.0
HIST_BINS = 30

_QUANTILES = {
    "p1": 0.01,
    "p5": 0.05,
    "q1": 0.25,
    "median": 0.5,
    "q3": 0.75,
    "p95": 0.95,
    "p99": 0.99,
}

NUMERIC_STATS_COLUMNS = [
    "column",
    "count",
    "mean",
    "std",
    "min",
    *_QUANTILES,
    "max",
    "skew",
    "kurtosis",
    "n_zeros",
    "n_negative",
    "n_outliers_iqr",
    "pct_outliers_iqr",
    "n_outliers_z",
    "pct_outliers_z",
]


def semantic_types(df: pd.DataFrame) -> dict[str, str]:
    """Semantic type of every column (compute once, reuse across the stats below)."""
    return {str(name): semantic_type(df[name]) for name in df.columns}


def columns_of_type(
    df: pd.DataFrame, *types: str, semantic: dict[str, str] | None = None
) -> list[str]:
    """Names of the columns whose semantic type is in `types` (df order).

    Pass `semantic` (from `semantic_types`) to avoid re-classifying the columns.
    """
    semantic = semantic if semantic is not None else semantic_types(df)
    return [str(c) for c in df.columns if semantic[str(c)] in types]


def _numeric_row(name: str, col: pd.Series) -> dict:
    values = col.dropna().astype(float)
    n = len(values)
    nan = float("nan")
    row: dict = {"column": name, "count": n}
    if n:
        q = dict(
            zip(_QUANTILES, values.quantile(list(_QUANTILES.values())), strict=True)
        )
    else:
        q = dict.fromkeys(_QUANTILES, nan)
    mean = values.mean() if n else nan
    std = values.std() if n > 1 else nan
    row.update(
        mean=mean,
        std=std,
        min=values.min() if n else nan,
        **q,
        max=values.max() if n else nan,
        skew=values.skew() if n > 2 else nan,
        kurtosis=values.kurt() if n > 3 else nan,
        n_zeros=int((values == 0).sum()),
        n_negative=int((values < 0).sum()),
    )
    n_iqr = n_z = 0
    if n:
        iqr = q["q3"] - q["q1"]
        lo, hi = q["q1"] - OUTLIER_IQR_K * iqr, q["q3"] + OUTLIER_IQR_K * iqr
        n_iqr = int(((values < lo) | (values > hi)).sum())
        if std and not np.isnan(std):
            n_z = int((((values - mean) / std).abs() > OUTLIER_Z).sum())
    row.update(
        n_outliers_iqr=n_iqr,
        pct_outliers_iqr=_pct(n_iqr, n),
        n_outliers_z=n_z,
        pct_outliers_z=_pct(n_z, n),
    )
    return row


def numeric_stats(
    df: pd.DataFrame, semantic: dict[str, str] | None = None
) -> pd.DataFrame:
    """One row per numeric column: moments, quantiles, shape, zeros, outliers.

    Outliers: outside [q1 - 1.5 IQR, q3 + 1.5 IQR] (`*_iqr`) or |z| > 3 (`*_z`);
    the pct is over the non-null count. Missing values are ignored.
    """
    names = columns_of_type(df, "numeric", semantic=semantic)
    rows = [_numeric_row(name, df[name]) for name in names]
    return pd.DataFrame(rows, columns=NUMERIC_STATS_COLUMNS)


def numeric_histograms(
    df: pd.DataFrame, bins: int = HIST_BINS, semantic: dict[str, str] | None = None
) -> pd.DataFrame:
    """Long-format histogram of the numeric columns: column, bin_left, bin_right, count."""
    frames = []
    for name in columns_of_type(df, "numeric", semantic=semantic):
        values = df[name].dropna().astype(float).to_numpy()
        if not len(values):
            continue
        counts, edges = np.histogram(values, bins=bins)
        frames.append(
            pd.DataFrame(
                {
                    "column": name,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "count": counts,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["column", "bin_left", "bin_right", "count"])
    return pd.concat(frames, ignore_index=True)


def category_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Long format `column, value, count, pct`: every value of each column.

    Missing values appear as their own value (`MISSING_LABEL`); pct is over all
    rows. Sorted by column name, then count descending (ties by value).
    """
    n_rows = len(df)
    frames = []
    for name in columns:
        counts = df[name].value_counts(dropna=False)
        frames.append(
            pd.DataFrame(
                {
                    "column": str(name),
                    "value": [
                        MISSING_LABEL if pd.isna(v) else str(v) for v in counts.index
                    ],
                    "count": counts.to_numpy(),
                }
            )
        )
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["column", "value", "count"])
    out["pct"] = [_pct(c, n_rows) for c in out["count"]]
    out = out.sort_values(
        ["column", "count", "value"], ascending=[True, False, True], kind="stable"
    )
    return out.reset_index(drop=True)


def category_summary(
    df: pd.DataFrame, semantic: dict[str, str] | None = None
) -> pd.DataFrame:
    """Per categorical / boolean column: cardinality, top value, rare-value mass.

    `top_pct` and `pct_rare` are over all rows; `pct_rare` = % of rows holding a
    value that itself covers < RARE_PCT % of the rows.
    """
    n_rows = len(df)
    rows = []
    for name in columns_of_type(df, "categorical", "boolean", semantic=semantic):
        counts = df[name].value_counts()
        rare = counts[counts * 100 < RARE_PCT * n_rows].sum()
        rows.append(
            {
                "column": name,
                "n_unique": len(counts),
                "top_value": str(counts.index[0]) if len(counts) else None,
                "top_pct": _pct(counts.iloc[0], n_rows) if len(counts) else 0.0,
                "pct_rare": _pct(rare, n_rows),
            }
        )
    return pd.DataFrame(
        rows, columns=["column", "n_unique", "top_value", "top_pct", "pct_rare"]
    )


def datetime_stats(
    df: pd.DataFrame, semantic: dict[str, str] | None = None
) -> pd.DataFrame:
    """Per datetime column: count, min, max (ISO strings) and span in days."""
    rows = []
    for name in columns_of_type(df, "datetime", semantic=semantic):
        col = df[name]
        if not pdt.is_datetime64_any_dtype(col):
            col = pd.to_datetime(col, errors="coerce", format="ISO8601")
        values = col.dropna()
        row = {"column": name, "count": len(values), "min": None, "max": None}
        row["span_days"] = None
        if len(values):
            lo, hi = values.min(), values.max()
            row.update(
                min=lo.isoformat(),
                max=hi.isoformat(),
                span_days=round((hi - lo).total_seconds() / 86400, 2),
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=["column", "count", "min", "max", "span_days"])


def text_stats(
    df: pd.DataFrame, semantic: dict[str, str] | None = None
) -> pd.DataFrame:
    """Per text column: count, distinct values, mean / min / max string length."""
    rows = []
    for name in columns_of_type(df, "text", semantic=semantic):
        values = df[name].dropna().astype(str)
        lengths = values.str.len()
        rows.append(
            {
                "column": name,
                "count": len(values),
                "n_unique": int(values.nunique()),
                "mean_len": round(float(lengths.mean()), 2) if len(values) else None,
                "min_len": int(lengths.min()) if len(values) else None,
                "max_len": int(lengths.max()) if len(values) else None,
            }
        )
    return pd.DataFrame(
        rows, columns=["column", "count", "n_unique", "mean_len", "min_len", "max_len"]
    )


def id_stats(df: pd.DataFrame, semantic: dict[str, str] | None = None) -> pd.DataFrame:
    """Per id_like / group_id column: distinct values and repeated identifiers.

    `n_duplicates` = non-null values minus distinct values (rows repeating an
    already-seen id): ~0 expected for id_like, large for a group_id.
    """
    semantic = semantic if semantic is not None else semantic_types(df)
    rows = []
    for name in columns_of_type(df, "id_like", "group_id", semantic=semantic):
        values = df[name].dropna()
        n_unique = int(values.nunique())
        n_dup = len(values) - n_unique
        rows.append(
            {
                "column": name,
                "semantic_type": semantic[name],
                "count": len(values),
                "n_unique": n_unique,
                "n_duplicates": n_dup,
                "pct_duplicates": _pct(n_dup, len(values)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "semantic_type",
            "count",
            "n_unique",
            "n_duplicates",
            "pct_duplicates",
        ],
    )
