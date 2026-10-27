"""DuckDB pushdown for very large CSVs (optional extra ``large``).

A role whose ``x`` is ``CsvSource(engine="duckdb")`` is read from the parquet DuckDB
cached for it (``sources.csv_duckdb``) and, for the calls below, never built as a
pandas frame: windowed rows, column profiles, the preview's head, source columns, row /
column counts and the filtered export are computed by DuckDB, with ``memory_limit`` and a
spill directory. Contract: ``docs/large-csv.md``.

``plan(ws, role, steps)`` decides, ``None`` meaning "use pandas" (the caller then runs
the unchanged pandas path). Pushed down only when the result is provably the one pandas
gives: the role has a plain csv ``x`` (no ``y`` file, merge or ``label_last``) and every
step is ``filter_rows`` or ``drop_columns`` (stateless, order preserving). A filter whose
value type does not match its column, an unknown column, an empty ``isin``... all return
``None``: pandas then keeps its own semantics and errors. Nothing here imports duckdb
unless a plan was made (the engine "duckdb" was asked for).

Safety: column names come only from the parquet's own ``DESCRIBE`` (a name a step uses
must be one of them) and are quoted; filter values are bound parameters, never text in
the SQL; the path is the engine's own cache file.

Windowed rows (``rows``): rows ``[offset, offset + limit)`` of the plan (plus a view-only
filter / sort) read by DuckDB with ``LIMIT / OFFSET``, ``_rid`` = the parquet's
``file_row_number`` (the row's position in the CSV); time does not depend on the offset.
Profiles: whole-table statistics, never a sample (``workspace.large_stats``).
Queries run under ``QUERY_MEMORY_LIMIT`` (spilling beyond it): what they hold is a page of
rows or one column's aggregates, so the process peak stays far below the file size.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dtk_engine.cache import LRU
from dtk_engine.errors import SourceError
from dtk_engine.ops.outliers import IQR_K
from dtk_engine.ops.profile import HIST_BINS
from dtk_engine.ops.transforms.cleaning import (
    Condition,
    DropColumnsParams,
    FilterRowsParams,
)
from dtk_engine.sources.csv_duckdb import (
    THREADS,
    _duckdb,
    _ident,
    _lit,
    csv_to_parquet,
)
from dtk_engine.sources.spec import CsvSource
from dtk_engine.workspace.models import Step, Workspace

# DuckDB's memory_limit for the queries below (it spills to a temp directory beyond it).
QUERY_MEMORY_LIMIT = "128MB"
# DuckDB's row position of a parquet row (``read_parquet(..., file_row_number=true)``).
ROW_NUMBER = "file_row_number"
TOP_VALUES = 20  # mirrors workspace.inspect._TOP_VALUES
PUSHABLE_OPS = ("filter_rows", "drop_columns")
EXPORT_FORMATS = ("parquet", "ipynb", "py")  # csv needs the frame

_INTEGERS = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
    "UINTEGER", "UBIGINT", "UHUGEINT",
}  # fmt: skip
_NUMBERS = _INTEGERS | {"FLOAT", "DOUBLE"}
_SHAPES = LRU(max_entries=256)
_COMPARES = {"eq": "=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}


@dataclass
class Plan:
    """A role as DuckDB sees it: a parquet file, the columns left and the WHERE."""

    parquet: Path
    types: dict[str, str]  # DuckDB type of every column in the parquet
    columns: list[str]  # columns after the steps, in order
    where: list[str] = field(default_factory=list)  # SQL predicates, combined by AND
    params: list[Any] = field(default_factory=list)  # their bound values, in order
    nullable: frozenset[str] = frozenset()  # columns with a missing value in the raw file

    def scan(self, columns: list[str] | None = None) -> str:
        cols = ", ".join(map(_ident, columns or self.columns))
        where = f" WHERE {' AND '.join(self.where)}" if self.where else ""
        return f"SELECT {cols} FROM read_parquet({_lit(str(self.parquet))}){where}"

    def key(self) -> str:
        """Cache key of the rows (not the columns): the parquet name is content addressed."""
        return repr((str(self.parquet), self.where, [(type(v).__name__, v) for v in self.params]))


def _eligible(ws: Workspace, role: str) -> CsvSource | None:
    dataset = getattr(ws.datasets, role)
    if dataset is None or ws.label_last or dataset.y is not None:
        return None
    if any(m.apply_to in ("both", role) for m in ws.merges):
        return None
    x = dataset.x
    if isinstance(x, CsvSource) and x.engine == "duckdb":
        return x
    return None


def wants(ws: Workspace, role: str) -> bool:
    """True when ``role`` reads a csv through the duckdb engine."""
    return _eligible(ws, role) is not None


def _step_applies(step: Step, role: str) -> bool:
    return step.target in (role, "both")


def _kind(duck_type: str) -> str:
    if duck_type in _NUMBERS:
        return "number"
    return "bool" if duck_type == "BOOLEAN" else "text"


def _value_fits(kind: str, value: Any) -> bool:
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if kind == "bool":
        return isinstance(value, bool)
    return isinstance(value, str)


def _membership(col: str, kind: str, c: Condition) -> tuple[str, list[Any]] | None:
    values = c.value
    if not values or not all(_value_fits(kind, v) for v in values):
        return None
    marks = ", ".join("?" * len(values))
    if c.op == "isin":
        return f"COALESCE({col} IN ({marks}), false)", list(values)
    return f"({col} IS NULL OR {col} NOT IN ({marks}))", list(values)


def _comparison(col: str, kind: str, c: Condition) -> tuple[str, list[Any]] | None:
    if not _value_fits(kind, c.value) or (kind == "bool" and c.op not in ("eq", "ne")):
        return None
    if c.op == "ne":  # pandas: NaN != v is True
        return f"{col} IS DISTINCT FROM ?", [c.value]
    return f"COALESCE({col} {_COMPARES[c.op]} ?, false)", [c.value]


def _condition(plan: Plan, c: Condition) -> tuple[str, list[Any]] | None:
    """SQL predicate (never NULL) and bound values for ``c``, or None when pandas must
    decide (see the module docstring)."""
    duck = plan.types.get(c.column)
    if c.column not in plan.columns or duck is None:
        return None
    col, kind = _ident(c.column), _kind(duck)
    if c.op in ("isna", "notna"):
        return f"{col} IS {'NOT ' if c.op == 'notna' else ''}NULL", []
    if c.op in ("isin", "notin"):
        return _membership(col, kind, c)
    return _comparison(col, kind, c)


def _filter(plan: Plan, params: FilterRowsParams) -> bool:
    parts = [_condition(plan, c) for c in params.conditions]
    if any(p is None for p in parts):
        return False
    glue = " AND " if params.combine == "and" else " OR "
    plan.where.append("(" + glue.join(sql for sql, _ in parts) + ")")  # type: ignore[misc]
    for _, values in parts:  # type: ignore[misc]
        plan.params.extend(values)
    return True


def _drop(plan: Plan, params: DropColumnsParams) -> bool:
    absent = [c for c in params.columns if c not in plan.columns]
    if absent and not params.missing_ok:
        return False
    plan.columns = [c for c in plan.columns if c not in params.columns]
    return bool(plan.columns)


class OutOfMemory(SourceError):
    """DuckDB ran out of ``QUERY_MEMORY_LIMIT`` (even spilling)."""


@contextmanager
def _connect(threads: int | None = None) -> Iterator[Any]:
    """A DuckDB connection under ``QUERY_MEMORY_LIMIT``, ``threads`` (default ``THREADS``)."""
    import tempfile

    duckdb = _duckdb()
    with tempfile.TemporaryDirectory(prefix="dtk-duckdb-") as spill:
        con = duckdb.connect(config={"memory_limit": QUERY_MEMORY_LIMIT, "threads": threads or THREADS, "temp_directory": spill})
        try:
            yield con
        except duckdb.OutOfMemoryException as exc:
            raise OutOfMemory(
                f"DuckDB ran out of its {QUERY_MEMORY_LIMIT} memory limit ({str(exc).splitlines()[0]})"
            ) from exc
        except duckdb.Error as exc:
            raise SourceError(f"DuckDB failed: {' '.join(str(exc).split())[:300]}") from exc
        finally:
            con.close()


def plan(ws: Workspace, role: str, steps: list[Step]) -> Plan | None:
    """The pushdown plan of ``role`` after ``steps``, or None to use pandas."""
    spec = _eligible(ws, role)
    if spec is None:
        return None
    mine = [s for s in steps if _step_applies(s, role)]
    if any(s.op not in PUSHABLE_OPS for s in steps):
        return None
    parquet = csv_to_parquet(spec)
    with _connect() as con:
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_lit(str(parquet))})").fetchall()
    types = {name: kind for name, kind, *_ in described}
    if ROW_NUMBER in types:  # would shadow DuckDB's row number
        return None
    out = Plan(parquet, types, list(types), nullable=_nullable(str(parquet)))
    target = getattr(ws.datasets, role).target_column
    if target is not None and target not in types:
        return None
    for step in mine:
        if step.op == "filter_rows":
            ok = _filter(out, FilterRowsParams.model_validate(step.params))
        else:
            ok = _drop(out, DropColumnsParams.model_validate(step.params))
        if not ok:
            return None
    return out


@functools.lru_cache(maxsize=64)
def _nullable(parquet: str) -> frozenset[str]:
    """Columns with a missing value, from the parquet's own statistics (no scan)."""
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(parquet).metadata
    names = meta.schema.to_arrow_schema().names
    out = set()
    for g in range(meta.num_row_groups):
        group = meta.row_group(g)
        for i, name in enumerate(names):
            stats = group.column(i).statistics
            if stats is None or not stats.has_null_count or stats.null_count:
                out.add(name)
    return frozenset(out)


def row_count(p: Plan) -> int:
    """Rows after the plan's steps (cached; parquet metadata when nothing filters)."""

    def compute() -> int:
        with _connect() as con:
            if not p.where:
                sql = f"SELECT sum(num_rows) FROM parquet_file_metadata({_lit(str(p.parquet))})"
                return int(con.execute(sql).fetchone()[0] or 0)
            return int(con.execute(f"SELECT count(*) FROM ({p.scan()})", p.params).fetchone()[0])

    return _SHAPES.memo(repr(("rows", p.key())), compute)


def shape(p: Plan) -> list[int]:
    """``[rows, columns]`` after the plan's steps."""
    return [row_count(p), len(p.columns)]


def dtypes(p: Plan, columns: list[str] | None = None) -> dict[str, str]:
    """The pandas dtype of each column, as the whole frame would have it: an int column
    with a missing value is float64, a bool one object (``read_parquet`` of the file)."""
    import pyarrow.parquet as pq

    empty = pq.ParquetFile(p.parquet).schema_arrow.empty_table().to_pandas()
    out = {}
    for name in columns or p.columns:
        dtype = str(empty[name].dtype)
        if name in p.nullable and p.types[name] in _INTEGERS:
            dtype = "float64"
        elif name in p.nullable and p.types[name] == "BOOLEAN":
            dtype = "object"
        out[name] = dtype
    return out


def _frame(p: Plan, con: Any, query: str, params: list[Any], columns: list[str]) -> pd.DataFrame:
    """The query's rows as pandas reads the parquet, with the whole-frame dtypes."""
    df = con.sql(query, params=params).to_arrow_table().to_pandas()
    for name, dtype in dtypes(p, columns).items():
        if dtype in ("float64", "object") and str(df[name].dtype) != dtype:
            df[name] = df[name].astype(dtype)
    return df


def _order(p: Plan, sort: list[tuple[str, bool]]) -> str:
    """ORDER BY of a stable multi-key sort, missing (NULL, NaN) last in both directions."""
    keys = []
    for name, desc in sort:
        c = _ident(name)
        if p.types[name] in ("FLOAT", "DOUBLE"):
            c = f"CASE WHEN isnan({c}) THEN NULL ELSE {c} END"
        keys.append(f"{c} {'DESC' if desc else 'ASC'} NULLS LAST")
    return " ORDER BY " + ", ".join([*keys, ROW_NUMBER])


def numbered(p: Plan, columns: list[str]) -> str:
    """``p.scan(columns)`` plus DuckDB's row number (the row's position in the CSV)."""
    cols = ", ".join([*map(_ident, columns), ROW_NUMBER])
    where = f" WHERE {' AND '.join(p.where)}" if p.where else ""
    return f"SELECT {cols} FROM read_parquet({_lit(str(p.parquet))}, file_row_number=true){where}"


def rows(
    p: Plan,
    offset: int,
    limit: int,
    columns: list[str],
    view: Plan | None = None,
    sort: list[tuple[str, bool]] | None = None,
) -> tuple[pd.DataFrame, int]:
    """``(page, total)``: rows ``[offset, offset + limit)`` of ``columns``, indexed by
    ``_rid``, and the row count after the steps and the view filter ``view`` (``p`` with
    the filter's predicates appended).

    A sorted page is read in two queries: the row numbers of the page (an external sort
    of the sort keys and row numbers only, which spills; DuckDB's top-N would hold
    ``offset + limit`` whole rows), then those rows, read on one thread: the row numbers
    are scattered over every row group, and each thread decodes whole row groups of all
    the columns (4 threads run out of ``QUERY_MEMORY_LIMIT`` on a wide file)."""
    q = view or p
    page_of = f" LIMIT {int(limit)} OFFSET {int(offset)}"
    # text keys sort in memory-hungry blocks per thread: one thread keeps them in the limit
    texty = any(p.types[name] not in _NUMBERS for name, _ in sort or [])
    with _connect(1 if texty else None) as con:
        if not sort:
            page = _frame(p, con, numbered(q, columns) + page_of, q.params, columns).set_index(ROW_NUMBER)
        else:
            con.execute("SET disabled_optimizers = 'top_n'")
            ids = f"SELECT {ROW_NUMBER} FROM ({numbered(q, [s for s, _ in sort])}){_order(p, sort)}{page_of}"
            rids = [r for (r,) in con.execute(ids, q.params).fetchall()]
            con.execute("SET threads = 1")  # the page's rows are scattered over every row group
            picked = f"{numbered(p, columns)} {'AND' if p.where else 'WHERE'} {ROW_NUMBER} IN (SELECT unnest(?))"
            page = _frame(p, con, picked, [*p.params, rids], columns).set_index(ROW_NUMBER).loc[rids]
    page.index.name = "_rid"
    return page, row_count(q)


def head(p: Plan, n: int) -> pd.DataFrame:
    """The first ``n`` rows of the plan, with the whole-frame dtypes."""
    with _connect() as con:
        return _frame(p, con, f"{p.scan()} LIMIT {max(int(n), 0)}", p.params, p.columns)


def source_columns(spec: CsvSource) -> list[dict]:
    """``contract.source_columns`` of a duckdb csv: names, dtypes, numeric (no scan)."""
    parquet = csv_to_parquet(spec)
    with _connect() as con:
        described = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_lit(str(parquet))})").fetchall()
    types = {name: kind for name, kind, *_ in described}
    p = Plan(parquet, types, list(types), nullable=_nullable(str(parquet)))
    return [
        {"name": name, "dtype": dtype, "numeric": types[name] in _NUMBERS}
        for name, dtype in dtypes(p).items()
    ]


def with_filter(p: Plan, params: FilterRowsParams) -> Plan | None:
    """A copy of ``p`` with ``params``'s predicates appended (None: pandas must decide)."""
    out = Plan(p.parquet, p.types, list(p.columns), list(p.where), list(p.params), p.nullable)
    return out if _filter(out, params) else None


def export_parquet(p: Plan, path: Path) -> dict:
    """Write the plan's rows to ``path`` (streamed by DuckDB); the manifest entry."""
    with _connect() as con:
        con.execute(f"COPY ({p.scan()}) TO {_lit(str(path))} (FORMAT parquet)", p.params)
        rows = con.execute(f"SELECT count(*) FROM read_parquet({_lit(str(path))})").fetchone()[0]
    return {"rows": int(rows), "columns": list(p.columns)}


# -- profile -----------------------------------------------------------------------


def histogram(p: Plan, name: str, lo: float, hi: float) -> dict | None:
    """``{edges, counts}`` as ``np.histogram(values, bins=HIST_BINS)`` (None: no range)."""
    if lo is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    first, last = (lo - 0.5, hi + 0.5) if lo == hi else (float(lo), float(hi))
    step = (last - first) / HIST_BINS
    edges = [first + i * step for i in range(HIST_BINS)] + [last]
    c = _ident(name)
    query = (
        f"SELECT least(floor((CAST({c} AS DOUBLE) - ?) / ?), {HIST_BINS - 1})::INTEGER AS b, count(*) "
        f"FROM ({p.scan([name])}) WHERE {c} IS NOT NULL GROUP BY b"
    )
    counts = [0] * HIST_BINS
    with _connect() as con:
        for b, n in con.execute(query, [first, step, *p.params]).fetchall():
            counts[max(0, min(HIST_BINS - 1, int(b)))] += int(n)
    return {"edges": edges, "counts": counts}


def iqr(p: Plan, name: str, q1: float | None, q3: float | None) -> tuple[dict | None, int]:
    """``({lo, hi}, n_outside)`` for the 1.5 * IQR fences, as ``ops.outliers``."""
    if q1 is None or q3 is None:
        return None, 0
    lo, hi = q1 - IQR_K * (q3 - q1), q3 + IQR_K * (q3 - q1)
    c = _ident(name)
    query = f"SELECT count(*) FROM ({p.scan([name])}) WHERE CAST({c} AS DOUBLE) < ? OR CAST({c} AS DOUBLE) > ?"
    with _connect() as con:
        n = con.execute(query, [*p.params, lo, hi]).fetchone()[0]
    return {"lo": float(lo), "hi": float(hi)}, int(n)
