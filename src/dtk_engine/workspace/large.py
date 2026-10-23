"""DuckDB pushdown for very large CSVs (optional extra ``large``).

A role whose ``x`` is ``CsvSource(engine="duckdb")`` is read from the parquet DuckDB
cached for it (``sources.csv_duckdb``) and, for the calls below, never built as a
pandas frame: column profiles, row / column counts and the filtered export are
computed by DuckDB, with ``memory_limit`` and a spill directory.

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

Profiles: counts, distinct, histogram, IQR fences / outliers, top values and skew are exact
(SQL). The heuristics that read cells in Python (kind, sentinels, variants, date / number
text detection, suggested params) run on a seeded sample of ``DEFAULT_SAMPLE`` rows (or the
caller's ``sample_rows``), so they are estimates, as in the ``sample_rows`` mode of
``column_profiles``. SUMMARIZE is not used for the payload: its null percentage is rounded
and its quartiles / distinct counts approximate.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.ops.outliers import IQR_K
from dtk_engine.ops.profile import HIST_BINS
from dtk_engine.ops.transforms.cleaning import (
    Condition,
    DropColumnsParams,
    FilterRowsParams,
)
from dtk_engine.sources.csv_duckdb import (
    MEMORY_LIMIT,
    _duckdb,
    _ident,
    _lit,
    csv_to_parquet,
)
from dtk_engine.sources.spec import CsvSource
from dtk_engine.workspace.models import Step, Workspace

# Rows the Python heuristics of a profile read (seeded reservoir sample).
DEFAULT_SAMPLE = 100_000
SAMPLE_SEED = 0
TOP_VALUES = 20  # mirrors workspace.inspect._TOP_VALUES
PUSHABLE_OPS = ("filter_rows", "drop_columns")
EXPORT_FORMATS = ("parquet", "ipynb", "py")  # csv needs the frame

_NUMBERS = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT",
    "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE",
}  # fmt: skip
_COMPARES = {"eq": "=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}


@dataclass
class Plan:
    """A role as DuckDB sees it: a parquet file, the columns left and the WHERE."""

    parquet: Path
    types: dict[str, str]  # DuckDB type of every column in the parquet
    columns: list[str]  # columns after the steps, in order
    where: list[str] = field(default_factory=list)  # SQL predicates, combined by AND
    params: list[Any] = field(default_factory=list)  # their bound values, in order

    def scan(self, columns: list[str] | None = None) -> str:
        cols = ", ".join(map(_ident, columns or self.columns))
        where = f" WHERE {' AND '.join(self.where)}" if self.where else ""
        return f"SELECT {cols} FROM read_parquet({_lit(str(self.parquet))}){where}"


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


@contextmanager
def _connect() -> Iterator[Any]:
    import tempfile

    duckdb = _duckdb()
    with tempfile.TemporaryDirectory(prefix="dtk-duckdb-") as spill:
        con = duckdb.connect(config={"memory_limit": MEMORY_LIMIT, "temp_directory": spill})
        try:
            yield con
        except duckdb.OutOfMemoryException as exc:
            raise SourceError(
                f"DuckDB ran out of its {MEMORY_LIMIT} memory limit ({str(exc).splitlines()[0]})"
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
    out = Plan(parquet, types, list(types))
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


def shape(p: Plan) -> list[int]:
    """``[rows, columns]`` after the plan's steps."""
    with _connect() as con:
        n = con.execute(f"SELECT count(*) FROM ({p.scan()})", p.params).fetchone()[0]
    return [int(n), len(p.columns)]


def export_parquet(p: Plan, path: Path) -> dict:
    """Write the plan's rows to ``path`` (streamed by DuckDB); the manifest entry."""
    with _connect() as con:
        con.execute(f"COPY ({p.scan()}) TO {_lit(str(path))} (FORMAT parquet)", p.params)
        rows = con.execute(f"SELECT count(*) FROM read_parquet({_lit(str(path))})").fetchone()[0]
    return {"rows": int(rows), "columns": list(p.columns)}


# -- profile -----------------------------------------------------------------------


def _numeric_aggregates(names: list[str], numeric: list[str]) -> str:
    parts = ["count(*)"]
    for name in names:
        c = _ident(name)
        parts += [f"count({c})", f"count(DISTINCT {c})"]
        if name in numeric:
            parts += [
                f"min({c})", f"max({c})", f"avg({c})",
                f"quantile_cont({c}, 0.25)", f"quantile_cont({c}, 0.5)", f"quantile_cont({c}, 0.75)",
            ]  # fmt: skip
    return ", ".join(parts)


def column_stats(p: Plan, names: list[str]) -> tuple[int, dict[str, dict]]:
    """``(rows, {name: exact aggregates})`` for ``names`` in one pass over the file."""
    numeric = [n for n in names if p.types[n] in _NUMBERS]
    with _connect() as con:
        row = con.execute(f"SELECT {_numeric_aggregates(names, numeric)} FROM ({p.scan(names)})", p.params).fetchone()
    stats: dict[str, dict] = {}
    i = 1
    for name in names:
        info = {"count": int(row[i]), "distinct": int(row[i + 1])}
        i += 2
        if name in numeric:
            keys = ("min", "max", "mean", "q1", "median", "q3")
            info.update(zip(keys, row[i : i + 6], strict=True))
            i += 6
        stats[name] = info
    return int(row[0]), stats


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


def top_values(p: Plan, name: str, info: dict, rows: int) -> list[tuple[Any, int]]:
    """The ``TOP_VALUES`` most frequent values (NULL counted, as ``value_counts(dropna=False)``)."""
    c = _ident(name)
    if info["count"] == rows and info["distinct"] == rows:  # all unique: any 20 values tie
        query = f"SELECT {c}, 1 FROM ({p.scan([name])}) LIMIT {TOP_VALUES}"
    else:
        query = (
            f"SELECT {c}, count(*) AS n FROM ({p.scan([name])}) GROUP BY {c} "
            f"ORDER BY n DESC, {c} LIMIT {TOP_VALUES}"
        )
    with _connect() as con:
        return [(v, int(n)) for v, n in con.execute(query, p.params).fetchall()]


def sample_frame(p: Plan, names: list[str], rows: int, cap: int) -> pd.DataFrame:
    """The columns ``names`` of the plan's rows: all of them when they fit in ``cap``,
    else a seeded reservoir sample of ``cap`` rows."""
    query = p.scan(names)
    if rows > cap:
        query = f"SELECT * FROM ({query}) USING SAMPLE reservoir({cap} ROWS) REPEATABLE ({SAMPLE_SEED})"
    with _connect() as con:
        return con.execute(query, p.params).df()
