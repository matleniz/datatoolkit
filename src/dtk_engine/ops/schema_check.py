"""Check a frame against an expected schema, collecting every violation.

Pure: a frame and a schema in, a list of findings out (the frame is never
mutated, the checks never stop at the first violation). A finding is an
``Issue`` of ``ops/compare/issues.py`` (``severity``, ``check``, ``column``,
``message``) plus ``count`` (offending rows or columns) and ``examples``: at
most ``max_examples`` ``{"row": position, "value": text}`` entries, the value
cut to ``VALUE_CHARS`` characters (a report never dumps data).

``schema`` is duck-typed (the ``expected_schema`` transform's params): it reads
``columns`` (rules with ``name``, ``required``, ``dtype``, ``nullable``,
``max_missing_pct``, ``min``, ``max``, ``allowed``, ``unique``),
``extra_columns``, ``min_rows``, ``max_rows`` and ``max_examples``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from dtk_engine.ops._util import pct, py
from dtk_engine.ops.compare.issues import ISSUE_FIELDS, SEVERITIES, Issue

CHECK_FIELDS = [*ISSUE_FIELDS, "count", "examples"]
VALUE_CHARS = 40
MAX_FINDINGS = 200  # findings kept in a report; the rest is summarised in one row

__all__ = [
    "CHECK_FIELDS",
    "ISSUE_FIELDS",
    "MAX_FINDINGS",
    "SEVERITIES",
    "VALUE_CHARS",
    "check_schema",
    "limit_findings",
    "summarize",
]


def _short(value: Any) -> str:
    text = str(py(value))
    return text if len(text) <= VALUE_CHARS else text[: VALUE_CHARS - 1] + "…"


def _finding(issue: Issue, count: int, examples: list[dict] | None = None) -> dict:
    out = dict(zip(ISSUE_FIELDS, issue, strict=True))
    return {**out, "count": int(count), "examples": examples or []}


def _examples(s: pd.Series, mask: pd.Series, limit: int) -> list[dict]:
    """First ``limit`` positions where ``mask`` holds, with the (cut) value."""
    positions = mask.to_numpy().nonzero()[0][:limit]
    return [{"row": int(p), "value": _short(s.iloc[p])} for p in positions]


# --- dtype ---------------------------------------------------------------------


def _is_string(s: pd.Series) -> bool:
    if pd.api.types.is_string_dtype(s.dtype) and s.dtype != object:
        return True
    return s.dtype == object and bool(s.dropna().map(lambda v: isinstance(v, str)).all())


def _matches(s: pd.Series, dtype: str) -> bool:
    if dtype == "numeric":
        return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if dtype == "bool":  # a CSV bool with blanks loads as object True / False / None
        return pd.api.types.is_bool_dtype(s) or (
            s.dtype == object and bool(s.dropna().map(lambda v: isinstance(v, bool)).all())
        )
    if dtype == "datetime":
        return pd.api.types.is_datetime64_any_dtype(s)
    if dtype == "category":  # CSV-loaded categories are strings: both count
        return isinstance(s.dtype, pd.CategoricalDtype) or _is_string(s)
    return _is_string(s)  # string


def effective_dtype(rule) -> str | None:
    """The declared dtype, else the one implied by a numeric / datetime bound."""
    if rule.dtype is not None:
        return rule.dtype
    bound = rule.min if rule.min is not None else rule.max
    if bound is None:
        return None
    return "numeric" if isinstance(bound, int | float) else "datetime"


# --- rules, one per constraint; each yields findings -----------------------------


def _bound(value: Any, dtype: str) -> Any:
    return pd.Timestamp(value) if dtype == "datetime" else value


def _dtype_findings(rule, s: pd.Series, dtype: str | None) -> Iterator[dict]:
    if dtype is not None and not _matches(s, dtype):
        msg = f"expected dtype {dtype}, found {s.dtype}"
        yield _finding(("error", "schema", rule.name, msg), 0)


def _missing_findings(rule, s: pd.Series, limit: int) -> Iterator[dict]:
    allowed_pct = 0.0 if rule.nullable is False else rule.max_missing_pct
    if allowed_pct is None:
        return
    missing = s.isna()
    n = int(missing.sum())
    share = pct(n, len(s))
    if n and share > allowed_pct:
        msg = f"{n} missing values ({share}%), at most {allowed_pct:g}% allowed"
        yield _finding(("error", "missing", rule.name, msg), n, _examples(s, missing, limit))


def _range_findings(rule, s: pd.Series, dtype: str | None, limit: int) -> Iterator[dict]:
    if rule.min is None and rule.max is None:
        return
    if dtype not in ("numeric", "datetime") or not _matches(s, dtype):
        return  # the dtype finding already says why the bounds cannot apply
    none = pd.Series(False, index=s.index)
    try:
        low = s < _bound(rule.min, dtype) if rule.min is not None else none
        high = s > _bound(rule.max, dtype) if rule.max is not None else none
    except TypeError:  # e.g. a timezone-aware column against a naive bound
        yield _finding(("error", "range", rule.name, "bounds not comparable with the column"), 0)
        return
    bad = (low | high).fillna(False)  # NaN / NaT are the missing rule's business
    n = int(bad.sum())
    if n:
        lo = "-inf" if rule.min is None else str(rule.min)
        hi = "inf" if rule.max is None else str(rule.max)
        msg = f"{n} values outside [{lo}, {hi}]"
        yield _finding(("error", "range", rule.name, msg), n, _examples(s, bad, limit))


def _allowed_findings(rule, s: pd.Series, limit: int) -> Iterator[dict]:
    if rule.allowed is None:
        return
    bad = s.notna() & ~s.isin(rule.allowed)
    n = int(bad.sum())
    if n:
        found = s[bad].drop_duplicates().head(4)  # before any str(): a column can be huge
        shown = ", ".join(repr(_short(v)) for v in found.head(3))
        more = " (+ more)" if len(found) > 3 else ""
        msg = f"{n} values not in the allowed set: {shown}{more}"
        yield _finding(("error", "allowed", rule.name, msg), n, _examples(s, bad, limit))


def _unique_findings(rule, s: pd.Series, limit: int) -> Iterator[dict]:
    if not rule.unique:
        return
    dup = s.notna() & s.duplicated(keep="first")
    n = int(dup.sum())
    if n:
        yield _finding(
            ("error", "duplicates", rule.name, f"{n} repeated values in a unique column"),
            n,
            _examples(s, dup, limit),
        )


def _column_findings(rule, s: pd.Series, limit: int) -> Iterator[dict]:
    dtype = effective_dtype(rule)
    yield from _dtype_findings(rule, s, dtype)
    yield from _missing_findings(rule, s, limit)
    yield from _range_findings(rule, s, dtype, limit)
    yield from _allowed_findings(rule, s, limit)
    yield from _unique_findings(rule, s, limit)


def _table_findings(df: pd.DataFrame, schema) -> Iterator[dict]:
    n = len(df)
    if schema.min_rows is not None and n < schema.min_rows:
        msg = f"{n} rows, at least {schema.min_rows} expected"
        yield _finding(("error", "schema", None, msg), 0)
    if schema.max_rows is not None and n > schema.max_rows:
        msg = f"{n} rows, at most {schema.max_rows} expected"
        yield _finding(("error", "schema", None, msg), 0)


def check_schema(df: pd.DataFrame, schema) -> list[dict]:
    """Every violation of ``schema`` in ``df``, errors first (rule order within).

    Columns are matched by their text, so integer column labels work.
    """
    by_name: dict[str, Any] = {}
    for col in df.columns:
        by_name.setdefault(str(col), col)
    limit = schema.max_examples
    found = list(_table_findings(df, schema))
    for rule in schema.columns:
        label = by_name.get(rule.name)
        if label is None:
            if rule.required:
                found.append(_finding(("error", "schema", rule.name, "required column missing"), 1))
            continue
        s = df[label]
        if isinstance(s, pd.DataFrame):  # duplicated column label: check the first
            s = s.iloc[:, 0]
        found.extend(_column_findings(rule, s, limit))
    if schema.extra_columns == "warn":
        declared = {r.name for r in schema.columns}
        extra = [str(c) for c in df.columns if str(c) not in declared]
        if extra:
            shown = ", ".join(repr(_short(c)) for c in extra[:5])
            more = f" (+{len(extra) - 5} more)" if len(extra) > 5 else ""
            msg = f"{len(extra)} columns not in the schema: {shown}{more}"
            found.append(_finding(("warning", "schema", None, msg), len(extra)))
    rank = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(found, key=lambda f: rank[f["severity"]])  # stable


def summarize(findings: list[dict]) -> dict[str, int]:
    """Findings per severity, e.g. ``{"error": 2, "warning": 1, "info": 0}``."""
    return {sev: sum(f["severity"] == sev for f in findings) for sev in SEVERITIES}


def limit_findings(found: list[dict], cap: int = MAX_FINDINGS) -> list[dict]:
    """The first ``cap`` findings (most severe first, as sorted), then one
    ``truncated`` row giving how many were left out and the totals per severity."""
    if len(found) <= cap:
        return found
    total = summarize(found)
    left = len(found) - cap
    msg = (
        f"{left} more violations not shown "
        f"(total: {total['error']} errors, {total['warning']} warnings, {total['info']} info)"
    )
    return [*found[:cap], _finding(("info", "truncated", None, msg), left)]
