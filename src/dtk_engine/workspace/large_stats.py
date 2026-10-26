"""Whole-table column statistics in DuckDB for a ``workspace.large`` plan (#210).

Every value is computed on every row of the plan (the role after its steps), never on a
sample. Exact and cheap aggregates stay exact (counts, min / max / mean / std, skewness,
histogram, the text rules below). Above ``APPROX_FROM_ROWS`` rows the costly ones use
DuckDB's approximate algorithms: ``approx_count_distinct`` (HyperLogLog) for distinct
counts, ``approx_quantile`` (t-digest) for quartiles, ``approx_top_k`` for the values of
a high-cardinality top list (their counts stay exact). The caller lists the fields that
are approximate (``approximate``), see ``docs/large-csv.md``.

The text rules of ``workspace.inspect`` / ``ops.profile`` / ``ops.missing`` (dates,
numbers or currency written as text, sentinels, variants, kind) are ported to SQL: same
thresholds, regexes rewritten for RE2 (a back-reference is spelled out per separator),
Python's ``str.strip`` / ``\\s`` as an explicit Unicode whitespace class, and numbers read
with ``TRY_CAST(... AS DOUBLE)`` (guarded to refuse what ``pd.to_numeric`` refuses:
``nan``, ``1_000``). ``kind`` also depends on the distinct count, so it is always listed
as approximate on this path.

Results are cached per (parquet, steps' WHERE, column): the parquet name is content
addressed (``sources.csv_duckdb``), so an entry can never be stale.
"""

from __future__ import annotations

import math
import os
from typing import Any

from dtk_engine.cache import LRU
from dtk_engine.ops.advisor.numeric import SKEW_THRESHOLD
from dtk_engine.ops.consistency import DATE_LIKE_MIN_SHARE
from dtk_engine.ops.missing import (
    CATEGORICAL_FILL,
    DATE_SENTINELS,
    NUMERIC_SENTINELS,
    NUMERIC_SPIKE_MIN_ROWS,
    NUMERIC_SPIKE_NEIGHBOURS,
    SPIKE_FACTOR,
    STRING_SENTINELS,
    ZERO_DOMINANCE,
    ZERO_MIN_ROWS,
    ZERO_MIN_UNIQUE,
)
from dtk_engine.ops.profile import (
    GROUP_MIN_RATIO,
    GROUP_MIN_UNIQUE,
    GROUP_RANGE_DENSITY,
    HIST_BINS,
    ID_MIN_NON_NULL,
    ID_RANGE_DENSITY,
    ID_UNIQUE_RATIO,
    NUMERIC_AS_TEXT_RATIO,
    NUMERIC_TEXT_RATIO,
    TEXT_UNIQUE_RATIO,
)
from dtk_engine.ops.suggested import (
    BINS_MAX,
    BINS_MIN,
    DISCRETE_INT_MAX_UNIQUE,
    LOG_SKEW_MIN_N,
    TOP_K_DEFAULT,
    TOP_K_SHOW_ALL_MAX,
)
from dtk_engine.sources.csv_duckdb import _ident, _lit
from dtk_engine.workspace.large import (
    _INTEGERS,
    _NUMBERS,
    ROW_NUMBER,
    TOP_VALUES,
    Plan,
    _connect,
    numbered,
    row_count,
)

# Above this many rows (after the steps) distinct counts, quartiles and big top lists
# are approximate; at or below it they are exact (cheap enough).
APPROX_FROM_ROWS = 1_000_000
# Estimated distinct values up to which a column is counted exactly (and a top list
# grouped exactly) in the approximate mode.
DISTINCT_EXACT_MAX = 100_000
# Approximate mode: variants are reported only when the normalized distinct estimate is
# below the raw one by more than this share (two HyperLogLog estimates differ by ~2 %).
VARIANTS_HLL_MARGIN = 0.05

_FACTS = LRU(max_entries=4096)

# Python's str.isspace() characters (what str.strip() and the \s of a str regex use).
_WS = r"[\s\x{0b}\x{1c}-\x{1f}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]"
_TIME = r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?" + _WS + r"*(?:[ap]m|Z|[+-]\d{2}:?\d{2})?)?"
_MONTH = r"(?i:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[A-Za-z]{0,6}"
# ops.consistency.date_format, one alternative per separator (RE2 has no \2).
_ISO = "|".join(rf"\d{{4}}{s}\d{{1,2}}{s}\d{{1,2}}" for s in ("-", "/", r"\."))
_NUMERIC_DATE = "|".join(rf"\d{{1,2}}{s}\d{{1,2}}{s}(?:\d{{4}}|\d{{2}})" for s in ("-", "/", r"\."))
_DAY_MON = rf"\d{{1,2}}[ -]{_MONTH}\.?,?[ -]\d{{2,4}}"
_MON_DAY = rf"{_MONTH}\.? \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}"
# workspace.inspect._NUMBER_AS_TEXT_RE, ops.profile._NUMERIC_TEXT_RE / _CURRENCY_TOKEN_RE.
# What pd.to_datetime(format="ISO8601") reads and a TIMESTAMP cast does not ("2020-01").
_YEAR_MONTH = r"\d{4}(?:-\d{1,2})?"
_NUMBER_AS_TEXT = r"[+-]?\d+(?:[.,]\d+)?"
_THOUSANDS = "[   ]"
_NUMERIC_TEXT = r"[+-]?\d{1,3}(?:[,. \x{a0}\x{202f}]\d{3})*(?:[.,]\d+)?%?"
_CURRENCY = r"(?i)[$€£]|USD|EUR|GBP"


def _strip(x: str) -> str:
    return f"regexp_replace({x}, {_lit(f'^{_WS}+|{_WS}+$')}, '', 'g')"


def _number(x: str) -> str:
    """``x`` (stripped text) as a DOUBLE where ``pd.to_numeric`` reads a number, else
    NULL. Only text starting like a number is cast (cheap), never ``nan``, ``1_000``."""
    starts = f"{x} <> '' AND contains('0123456789+-.iI', left({x}, 1)) AND NOT contains({x}, '_')"
    return f"CASE WHEN {starts} THEN nullif(TRY_CAST({x} AS DOUBLE), 'NaN'::DOUBLE) END"


def _full(x: str, pattern: str) -> str:
    return f"regexp_full_match({x}, {_lit(pattern)})"


def _is_date_text(st: str) -> str:
    """SQL of ``date_format(value) is not None`` on the stripped text ``st``."""
    a = f"TRY_CAST(regexp_extract({st}, '^(\\d{{1,2}})', 1) AS INTEGER)"
    b = f"TRY_CAST(regexp_extract({st}, '^\\d{{1,2}}[-/.](\\d{{1,2}})', 1) AS INTEGER)"
    numeric_ok = f"NOT ({a} > 31 OR {b} > 31 OR ({a} > 12 AND {b} > 12))"
    worded = _full(st, f"(?:{_ISO}|{_DAY_MON}|{_MON_DAY}){_TIME}")
    numeric = _full(st, f"(?:{_NUMERIC_DATE}){_TIME}")
    return f"(CASE WHEN {worded} THEN true WHEN {numeric} THEN {numeric_ok} ELSE false END)"


def _normalized(v: str) -> str:
    """SQL of ``ops.consistency.normalize``: runs of separators / whitespace become one
    space (what its sub-then-split-then-join does), stripped, lower-cased."""
    runs = _lit(f"(?:[-_.]|{_WS})+")
    return f"lower(trim(regexp_replace({v}, {runs}, ' ', 'g')))"


def _distinct(x: str, approx: bool) -> str:
    return f"approx_count_distinct({x})" if approx else f"count(DISTINCT {x})"


def _common(c: str, approx: bool) -> list[tuple[str, str]]:
    return [("rows", "count(*)"), ("count", f"count({c})"), ("distinct", _distinct(c, approx))]


def _numeric_parts(c: str, approx: bool) -> list[tuple[str, str]]:
    x = f"CAST({c} AS DOUBLE)"
    quart = "approx_quantile" if approx else "quantile_cont"
    parts = [
        ("min", f"min({x})"), ("max", f"max({x})"), ("mean", f"avg({x})"),
        ("std", f"stddev_samp({x})"), ("skew", f"skewness({x})"),
        ("quartiles", f"{quart}({x}, [0.25, 0.5, 0.75])"),
        ("all01", f"bool_and({x} IN (0, 1))"), ("integral", f"bool_and({x} = round({x}))"),
        ("zeros", f"count(*) FILTER (WHERE {x} = 0)"),
    ]  # fmt: skip
    return parts + [(f"code{s}", f"count(*) FILTER (WHERE {x} = {s})") for s in NUMERIC_SENTINELS]


def _rows_where(cond: str) -> str:
    """Rows (weights ``k``) where ``cond`` holds (a CASE: cheaper than 25 FILTERs)."""
    return f"coalesce(sum(CASE WHEN {cond} THEN k ELSE 0 END), 0)"


def _text_parts(normalized: str) -> list[tuple[str, str]]:
    """Aggregates over ``_text_rows`` (one row per value ``v`` of weight ``k``: its rows)."""
    nonempty = "st <> ''"
    parts = [
        ("all_digits", f"bool_and({_full('v', '[0-9]+')})"),
        ("all_dates", f"bool_and(coalesce(isfinite(TRY_CAST(v AS TIMESTAMP)), false) OR {_full('v', _YEAR_MONTH)})"),
        ("numeric_share", f"{_rows_where('num IS NOT NULL')} / sum(k)"),
        ("any_space", f"bool_or(regexp_matches(v, {_lit(_WS)}))"),
        ("nonempty", _rows_where(nonempty)),
        ("dated", _rows_where(f"{nonempty} AND {_is_date_text('st')}")),
        ("all_number_text", f"bool_and({_full('st', _NUMBER_AS_TEXT)})"),
        ("matched_share", f"{_rows_where('matched')} / sum(k)"),
        ("any_symbol", "bool_or(symbol)"), ("any_percent", "bool_or(percent)"),
        ("any_whole", "bool_or(whole)"),
        ("has_comma", "bool_or(matched AND contains(s4, ','))"),
        ("has_dot", "bool_or(matched AND contains(s4, '.'))"),
        ("has_space", f"bool_or(matched AND regexp_matches(s4, {_lit(_THOUSANDS)}))"),
        ("normalized", normalized),
        ("zeros", _rows_where("num = 0")),
    ]  # fmt: skip
    parts += [
        (f"str{s}", _rows_where(f"st <> {_lit(CATEGORICAL_FILL)} AND lower(st) = {_lit(s)}"))
        for s in sorted(STRING_SENTINELS)
    ]
    parts += [(f"date{s}", _rows_where(f"starts_with(st, {_lit(s)})")) for s in DATE_SENTINELS]
    return parts + [(f"code{s}", _rows_where(f"num = {s}")) for s in NUMERIC_SENTINELS]


def _text_rows(plan: Plan, name: str, grouped: bool) -> str:
    """The column's non-null values ``v`` with their weight ``k`` (one row per cell, or
    one per distinct value with its count when ``grouped``), the stripped text ``st``
    and the steps of ``ops.profile.numeric_text_format``."""
    c = _ident(name)
    cells = f"SELECT {c} AS v FROM ({plan.scan([name])}) WHERE {c} IS NOT NULL"
    values = f"SELECT v, count(*) AS k FROM ({cells}) GROUP BY v" if grouped else f"SELECT v, 1 AS k FROM ({cells})"
    sym = _lit(_CURRENCY)
    unsymboled = f"regexp_replace(st, {sym}, '', 'g')"
    s2 = f"CASE WHEN symbol THEN {_strip(unsymboled)} ELSE st END"
    unpercented = "replace(s2, '%', '')"
    s3 = f"CASE WHEN percent THEN {_strip(unpercented)} ELSE s2 END"
    steps = [  # each SELECT adds one column to the one below it
        f"SELECT v, k, {_strip('v')} AS st",
        f"SELECT v, k, st, {_number('st')} AS num",
        f"SELECT *, regexp_matches(st, {sym}) AS symbol",
        f"SELECT *, {s2} AS s2",
        "SELECT *, contains(s2, '%') AS percent",
        f"SELECT *, {s3} AS s3",
        "SELECT *, suffix(s3, ',-') OR suffix(s3, '.-') AS whole",
        "SELECT *, CASE WHEN whole THEN left(s3, length(s3) - 2) ELSE s3 END AS s4",
        f"SELECT *, regexp_full_match(s4, {_lit(_NUMERIC_TEXT)}) AS matched",
    ]
    query = values
    for step in steps:
        query = f"{step} FROM ({query})"
    return query


def _head(plan: Plan, name: str, approx: bool) -> dict[str, Any]:
    """Counts (and the numeric aggregates) of the column in one scan."""
    c = _ident(name)
    parts = _common(c, approx) + (_numeric_parts(c, approx) if plan.types[name] in _NUMBERS else [])
    with _connect() as con:
        row = con.execute(f"SELECT {', '.join(e for _, e in parts)} FROM ({plan.scan([name])})", plan.params).fetchone()
    out = dict(zip([k for k, _ in parts], row, strict=True))
    if "quartiles" in out:
        out["q1"], out["median"], out["q3"] = out.pop("quartiles") or (None, None, None)
    return out


def _exact_distinct(plan: Plan, name: str) -> int:
    c = _ident(name)
    with _connect() as con:
        return int(con.execute(f"SELECT count(DISTINCT {c}) FROM ({plan.scan([name])})", plan.params).fetchone()[0])


def _text(plan: Plan, name: str, grouped: bool, approx: bool) -> dict[str, Any]:
    """The text rules' aggregates; ``grouped`` also counts the distinct values exactly."""
    streaming = approx and not grouped  # no hash table: every core, no spill to thrash
    normalized = _distinct(_normalized("v"), streaming)
    parts = _text_parts(normalized) + ([("groups", "count(*)")] if grouped else [])
    with _connect(threads=os.cpu_count() if streaming else None) as con:
        row = con.execute(
            f"SELECT {', '.join(e for _, e in parts)} FROM ({_text_rows(plan, name, grouped)})", plan.params
        ).fetchone()
    return dict(zip([k for k, _ in parts], row, strict=True))


def facts(plan: Plan, name: str) -> dict[str, Any]:
    """The aggregates of column ``name`` on the whole table (cached), plus ``type``,
    ``nullable`` (missing values in the raw file: pandas reads such an int column as
    float64, a bool one as object), ``approx`` (quartiles and big top lists are
    approximate) and ``approx_distinct`` (the distinct count is).

    In the approximate mode a column whose estimated distinct count is at most
    ``DISTINCT_EXACT_MAX`` is counted exactly (cheap: few groups); a text column then
    evaluates its rules once per distinct value, weighted by its rows."""

    def compute() -> dict[str, Any]:
        approx = row_count(plan) > APPROX_FROM_ROWS
        out = _head(plan, name, approx)
        few = out["distinct"] <= DISTINCT_EXACT_MAX
        approx_distinct = approx and not few
        if plan.types[name] == "VARCHAR":
            out.update(_text(plan, name, grouped=few, approx=approx))
            if few:
                out["distinct"] = out.pop("groups")
        elif approx and few:
            out["distinct"] = _exact_distinct(plan, name)
        out["distinct"] = min(int(out["distinct"]), int(out["count"]))
        out.update(type=plan.types[name], nullable=name in plan.nullable, approx=approx, approx_distinct=approx_distinct)
        return out

    return _FACTS.memo(repr(("facts", plan.key(), name)), compute)


def is_number(f: dict) -> bool:
    """A numeric, non-bool pandas dtype (``ops.columns.is_numeric``)."""
    return f["type"] in _NUMBERS


def is_bool(f: dict) -> bool:
    """A pandas bool dtype: BOOLEAN without missing values (else object)."""
    return f["type"] == "BOOLEAN" and not f["nullable"]


def _group_band(n_unique: int, ratio: float) -> bool:
    return n_unique >= GROUP_MIN_UNIQUE and GROUP_MIN_RATIO <= ratio < ID_UNIQUE_RATIO


def _numeric_semantic(f: dict, n_unique: int, ratio: float) -> str:
    if n_unique == 2 and f["all01"]:
        return "boolean"
    if f["type"] in _INTEGERS and not f["nullable"]:
        density = n_unique / (int(f["max"]) - int(f["min"]) + 1)
        if f["count"] >= ID_MIN_NON_NULL and ratio >= ID_UNIQUE_RATIO and density >= ID_RANGE_DENSITY:
            return "id_like"
        if _group_band(n_unique, ratio) and density >= GROUP_RANGE_DENSITY:
            return "group_id"
    return "numeric"


def _text_semantic(f: dict, n_unique: int, ratio: float) -> str:
    if f["type"] == "VARCHAR":
        if not f["all_digits"] and f["all_dates"]:
            return "datetime"
        mostly = f["numeric_share"] >= NUMERIC_AS_TEXT_RATIO or currency_as_text(f) is not None
        if not mostly and not f["any_space"]:
            if f["count"] >= ID_MIN_NON_NULL and ratio >= ID_UNIQUE_RATIO:
                return "id_like"
            if _group_band(n_unique, ratio):
                return "group_id"
    return "text" if ratio > TEXT_UNIQUE_RATIO else "categorical"


def semantic(f: dict) -> str:
    """``ops.profile.semantic_type`` from the facts."""
    n_unique = f["distinct"]
    if n_unique <= 1:
        return "constant"
    if is_bool(f):
        return "boolean"
    ratio = n_unique / f["count"]
    if is_number(f):
        return _numeric_semantic(f, n_unique, ratio)
    return _text_semantic(f, n_unique, ratio)  # VARCHAR, or BOOLEAN with NULLs (object)


def currency_as_text(f: dict) -> dict | None:
    """``ops.profile.numeric_text_format`` from the facts."""
    if f["type"] != "VARCHAR" or not f["count"] or f["matched_share"] < NUMERIC_TEXT_RATIO:
        return None
    if not (f["any_symbol"] or f["any_percent"] or f["any_whole"] or f["has_comma"] or f["has_space"]):
        return None
    if f["has_comma"] and f["has_dot"]:
        decimal, thousands = ".", ","
    else:
        decimal = "," if f["has_comma"] else "."
        thousands = " " if f["has_space"] else None
    return {"decimal": decimal, "thousands": thousands, "percent": bool(f["any_percent"])}


def text_fields(f: dict) -> dict:
    """``variants``, ``looks_like_dates``, ``numbers_as_text``, ``currency_as_text``."""
    if f["type"] != "VARCHAR" or not f["count"]:
        return {"variants": None, "looks_like_dates": False, "numbers_as_text": False, "currency_as_text": None}
    raw, normalized = f["distinct"], min(int(f["normalized"]), f["distinct"])
    floor = raw * (1 - VARIANTS_HLL_MARGIN) if f["approx_distinct"] else raw
    return {
        "variants": {"raw": raw, "normalized": normalized} if normalized < floor else None,
        "looks_like_dates": bool(f["nonempty"]) and f["dated"] >= DATE_LIKE_MIN_SHARE * f["nonempty"],
        "numbers_as_text": bool(f["all_number_text"]),
        "currency_as_text": currency_as_text(f),
    }


def suggested_params(f: dict, kind: str) -> dict:
    """``ops.suggested.suggested_params(series, kind=kind)`` from the facts."""
    n_unique = f["distinct"]
    out: dict = {"top_k": max(1, n_unique) if 0 < n_unique <= TOP_K_SHOW_ALL_MAX else TOP_K_DEFAULT}
    if kind != "number":
        return out
    if not is_number(f):
        return {**out, "bins": HIST_BINS, "log_scale": False}
    n = f["count"]
    return {**out, "bins": _bins(f, n, n_unique), "log_scale": _log_scale(f, n)}


def _clip(n: int) -> int:
    return int(min(max(n, BINS_MIN), BINS_MAX))


def _bins(f: dict, n: int, n_unique: int) -> int:
    if n < 2:
        return BINS_MIN
    if f["integral"] and n_unique <= DISCRETE_INT_MAX_UNIQUE:
        return _clip(n_unique)
    iqr, span = f["q3"] - f["q1"], f["max"] - f["min"]
    if iqr > 0 and span > 0:
        width = 2.0 * iqr * (n ** (-1.0 / 3.0))
        if width > 0:
            return _clip(math.ceil(span / width))
    return _clip(math.ceil(math.log2(n) + 1))


def _log_scale(f: dict, n: int) -> bool:
    if n < LOG_SKEW_MIN_N or f["min"] < 0:
        return False
    skew = f["skew"]
    return skew is not None and not math.isnan(skew) and skew > SKEW_THRESHOLD


def stats(f: dict) -> dict | None:
    if not is_number(f):
        return None
    return {k: (None if f[k] is None else float(f[k])) for k in ("min", "max", "mean", "std", "q1", "median", "q3")}


# -- sentinels ---------------------------------------------------------------------


def _numbers(plan: Plan, name: str, f: dict) -> str:
    """Subquery of the column's numbers (``pd.to_numeric(...).dropna()`` for text)."""
    c = _ident(name)
    x = _number(_strip(c)) if f["type"] == "VARCHAR" else f"CAST({c} AS DOUBLE)"
    return f"SELECT {x} AS n FROM ({plan.scan([name])}) WHERE {x} IS NOT NULL"


def _is_code(con: Any, nums: str, params: list, code: float, count: int, codes: list[float]) -> bool:
    """``ops.missing._is_numeric_code``: does ``code`` stand out from the other values?"""
    others = f"SELECT n FROM ({nums}) WHERE n NOT IN ({', '.join(map(repr, codes))})"
    lo, hi = con.execute(f"SELECT min(n), max(n) FROM ({others})", params).fetchone()
    if lo is None or code < lo or code > hi:
        return True
    if count < NUMERIC_SPIKE_MIN_ROWS:
        return False
    nearest = con.execute(
        f"SELECT max(k) FROM (SELECT k FROM (SELECT n, count(*) AS k FROM ({others}) GROUP BY n) "
        f"ORDER BY abs(n - ?), n LIMIT {NUMERIC_SPIKE_NEIGHBOURS})",
        [*params, code],
    ).fetchone()[0]
    return count >= SPIKE_FACTOR * max(int(nearest or 0), 1)


def _numeric_hits(plan: Plan, name: str, f: dict) -> dict[str, int]:
    """``ops.missing._numeric_sentinel_hits`` in SQL."""
    codes = {str(s): int(f[f"code{s}"]) for s in NUMERIC_SENTINELS if f[f"code{s}"]}
    zeros = int(f["zeros"])
    if not codes and zeros < ZERO_MIN_ROWS:
        return {}
    nums, values = _numbers(plan, name, f), [float(c) for c in codes]
    with _connect() as con:
        hits = {
            label: count
            for label, count in codes.items()
            if _is_code(con, nums, plan.params, float(label), count, values)
        }
        if zeros >= ZERO_MIN_ROWS:
            unique, top = con.execute(
                f"SELECT {_distinct('n', f['approx_distinct'])}, (SELECT max(k) FROM (SELECT count(*) AS k FROM ({nums}) "
                f"WHERE n <> 0 GROUP BY n)) FROM ({nums})",
                plan.params * 2,
            ).fetchone()
            if unique >= ZERO_MIN_UNIQUE and zeros >= ZERO_DOMINANCE * int(top or 0):
                hits["0"] = zeros
    return hits


def sentinels(plan: Plan, name: str, f: dict) -> list[dict]:
    """``workspace.inspect._sentinel_candidates`` (count order, stable)."""
    if is_bool(f) or f["type"] not in (*_NUMBERS, "VARCHAR"):
        return []
    hits: dict[str, int] = {}
    if f["type"] == "VARCHAR":
        hits = {(repr(s) if s == "" else s): int(f[f"str{s}"]) for s in sorted(STRING_SENTINELS)}
        hits |= {s: int(f[f"date{s}"]) for s in DATE_SENTINELS}
        hits = {k: v for k, v in hits.items() if v}
    hits |= _numeric_hits(plan, name, f)
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    return [{"value": label, "count": count} for label, count in ranked]


# -- top values ----------------------------------------------------------------------


def top_values(plan: Plan, name: str, f: dict) -> tuple[list[tuple[Any, int]], bool]:
    """The ``TOP_VALUES`` most frequent values, NULL counted (as
    ``value_counts(dropna=False)``), and whether the list is approximate."""
    c, scan = _ident(name), plan.scan([name])
    missing = f["rows"] - f["count"]
    with _connect() as con:
        if f["distinct"] <= DISTINCT_EXACT_MAX or not f["approx"]:
            query = (  # ties in order of first appearance, as value_counts
                f"SELECT {c}, count(*) AS n FROM ({numbered(plan, [name])}) GROUP BY {c} "
                f"ORDER BY n DESC, min({ROW_NUMBER}) LIMIT {TOP_VALUES}"
            )
            return [(v, int(n)) for v, n in con.execute(query, plan.params).fetchall()], False
        values = con.execute(f"SELECT approx_top_k({c}, {TOP_VALUES}) FROM ({scan})", plan.params).fetchone()[0]
        counted = con.execute(
            f"SELECT {c}, count(*) AS n FROM ({scan}) WHERE {c} IN (SELECT unnest(?)) GROUP BY {c}",
            [*plan.params, list(values or [])],
        ).fetchall()
    found = [(v, int(n)) for v, n in counted] + ([(None, missing)] if missing else [])
    found.sort(key=lambda vn: -vn[1])
    return found[:TOP_VALUES], True

