"""CSV reader backed by DuckDB (optional extra ``large``): ``CsvSource(engine="duckdb")``.

The file is converted ONCE to a parquet file by DuckDB, streaming and memory-capped
(it spills to a temp directory), then read with the parquet reader: the rest of the
engine only ever sees a pandas frame. The parquet is cached under
``$DTK_HOME/cache/csv_duckdb`` (default ``~/.datatoolkit``), keyed by the spec and
the file's (path, mtime, size); one entry per source path, an older one is removed.
Disk use: a parquet is about half the size of its CSV. Entries of files that were
deleted, moved or uploaded once are never purged: clear ``cache_dir()`` by hand.
It is a plain parquet file: ``ParquetSource(path=...)`` can read it later.

What it buys: time (1M x 20 CSV, cold: 2.7 s against 8.0 s with pandas; 0.9 s once
cached) and a parquet reusable across sessions. It does NOT lower the peak memory of
loading a frame: the result is still a whole pandas frame (peak RSS ~800 MB against
~690 MB). The memory cap applies to DuckDB's conversion only.

Known differences with the pandas reader, none of them silent (each is listed here,
and options that cannot be honoured raise ``SourceError`` instead of being ignored):

- Types: ints (int64), floats, booleans, else strings. Dates and timestamps stay
  strings, as in pandas without ``parse_dates`` (which is refused here).
- A column is typed from a sample, then checked on the whole file: a value that
  breaks the type triggers a re-read typing the column from every row (so a column
  with one stray ``abc`` becomes strings, as pandas does).
- Missing values: the pandas defaults (``""``, ``NA``, ``N/A``, ``NaN``, ``null``,
  ``None``, ...) plus ``na_values``; a quoted ``""`` is missing too, as in pandas.
- Leading zeros: with ``keep_leading_zeros`` (default) a digits-only column with a
  value like ``007`` stays strings, as in the pandas reader.
- Encoding: only utf-8 (a BOM is dropped), utf-16 (with a BOM) and latin-1, DuckDB's
  own. ``encoding="auto"`` is BOM or utf-8, never a guess: a cp1252 / latin-1 file
  fails with a message asking for ``encoding="latin-1"``. Anything else is refused.
- ``sep="auto"`` uses the pandas reader's sniffer on the head of the file, not
  DuckDB's (which reads a ragged file as one column). There is no ``skiprows="auto"``,
  no ``mixed_sep`` handling and no ``on_bad_lines``: those options are refused. A ragged row (more or fewer fields) is an error.
- Column names: duplicates become ``a``, ``a_1``; with ``header=None`` the columns
  are ``column0``, ``column1``, ... (pandas: ``0``, ``1``, ...). Rows keep file order
  (``preserve_insertion_order`` stays on: replay, head and tail depend on it).
- ``dtype`` accepts ``str``, ``int64`` (or ``int``), ``float64`` (or ``float``) and
  ``bool``; other dtypes are refused.
"""

from __future__ import annotations

import codecs
import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from dtk_engine.cache import file_stamp
from dtk_engine.errors import SourceError
from dtk_engine.sources.csv_pandas import SNIFF_CHARS, resolve_path, sniff_sep
from dtk_engine.sources.spec import CsvSource

# DuckDB's ``memory_limit`` for the conversion (not configurable): beyond it DuckDB
# spills to a temp directory, and fails with a SourceError when even that is not
# enough (256MB converts the 1M x 20 benchmark file, 128MB does not).
MEMORY_LIMIT = "1GB"
# Bumped when the conversion changes, so parquet files cached by an older one are not reused.
CONVERSION_VERSION = 1
# Rows DuckDB types a column from before the whole-file re-read (its default).
SAMPLE_ROWS = 20480
# pandas' default missing-value strings (pandas._libs.parsers.STR_NA_VALUES).
PANDAS_NA = (
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND",
    "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
)  # fmt: skip
# Column types DuckDB may infer: no DATE / TIMESTAMP (pandas keeps them as str).
TYPE_CANDIDATES = ("BIGINT", "DOUBLE", "BOOLEAN", "VARCHAR")
_INTEGER_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}  # fmt: skip
_ENCODINGS = {"utf-8": "utf-8", "utf-8-sig": "utf-8", "utf-16": "utf-16", "iso8859-1": "latin-1"}
_DTYPES = {
    "str": "VARCHAR", "string": "VARCHAR", "object": "VARCHAR",
    "int": "BIGINT", "int64": "BIGINT",
    "float": "DOUBLE", "float64": "DOUBLE",
    "bool": "BOOLEAN",
}  # fmt: skip
_MISSING_EXTRA = (
    "csv engine 'duckdb' needs the \"large\" extra: "
    'uv pip install "dtk-engine[large] @ git+https://github.com/matleniz/datatoolkit"'
)
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


def _duckdb() -> Any:
    """The duckdb module, imported on first use (the extra is optional)."""
    try:
        import duckdb
    except ImportError:
        raise SourceError(_MISSING_EXTRA) from None
    return duckdb


def cache_dir() -> Path:
    """``$DTK_HOME/cache/csv_duckdb``, ``~/.datatoolkit/cache/csv_duckdb`` when unset."""
    home = os.environ.get("DTK_HOME")
    return (Path(home) if home else Path.home() / ".datatoolkit") / "cache" / "csv_duckdb"


def read_csv_duckdb(spec: CsvSource) -> pd.DataFrame:
    """The csv as a pandas frame, via a cached parquet written by DuckDB."""
    from dtk_engine.sources.parquet import read_parquet
    from dtk_engine.sources.spec import ParquetSource

    return read_parquet(ParquetSource(path=str(csv_to_parquet(spec))))


def csv_to_parquet(spec: CsvSource) -> Path:
    """The parquet file for ``spec`` (converted now unless already cached)."""
    _refuse_unsupported(spec)
    path = resolve_path(spec.path)
    stamp = file_stamp(path)
    if stamp is None:
        missing = not os.path.exists(path)
        raise SourceError(
            f"csv source not found: {spec.path}"
            if missing
            else f"cannot read csv {spec.path}: not a regular file"
        )
    duckdb = _duckdb()
    root = cache_dir()
    prefix = hashlib.sha256(stamp[0].encode()).hexdigest()[:16]
    key = hashlib.sha256(repr((spec.model_dump_json(), stamp, duckdb.__version__, CONVERSION_VERSION)).encode())
    out = root / f"{prefix}-{key.hexdigest()[:24]}.parquet"
    if out.exists():
        return out
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".partial-", suffix=".parquet")
    os.close(fd)
    try:
        _convert(duckdb, spec, path, tmp)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    for partial in root.glob(".partial-*"):  # left by a killed conversion
        if time.time() - partial.stat().st_mtime > 3600:
            partial.unlink(missing_ok=True)
    for old in root.glob(f"{prefix}-*.parquet"):
        if old != out:
            old.unlink(missing_ok=True)
    return out


def _refuse_unsupported(spec: CsvSource) -> None:
    refused = [
        name
        for name, bad in (
            ("parse_dates", spec.parse_dates is not None),
            ("mixed_sep", spec.mixed_sep != "error"),
            ("on_bad_lines", spec.on_bad_lines != "error"),
            ("skiprows='auto'", spec.skiprows == "auto"),
            ("header (only 0 or null)", spec.header not in (0, None)),
        )
        if bad
    ]
    if refused:
        raise SourceError(
            f"csv engine 'duckdb' does not support {', '.join(refused)}: use "
            "engine='pandas', or drop the option"
        )


def _encoding(spec: CsvSource, path: str) -> str:
    if spec.encoding == "auto":
        with open(path, "rb") as fh:
            return "utf-16" if fh.read(2) in _UTF16_BOMS else "utf-8"
    try:
        name = codecs.lookup(spec.encoding).name
    except LookupError:
        raise SourceError(f"cannot read csv {spec.path}: unknown encoding {spec.encoding!r}") from None
    if name not in _ENCODINGS:
        raise SourceError(
            f"csv engine 'duckdb' reads utf-8, utf-16 (with a BOM) and latin-1, not "
            f"{spec.encoding!r}: re-encode the file, or use engine='pandas'"
        )
    return _ENCODINGS[name]


def _sep(spec: CsvSource, path: str) -> str | None:
    """``spec.sep``, or the delimiter sniffed from the head of the file (None: let
    DuckDB sniff)."""
    if spec.sep != "auto":
        return spec.sep
    with open(path, "rb") as fh:
        head = fh.read(4 * SNIFF_CHARS)
    text = head.decode(_encoding(spec, path), errors="replace")
    return sniff_sep(text[:SNIFF_CHARS])


def _lit(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_options(spec: CsvSource, path: str, types: dict[str, str], *, sample: int) -> str:
    """``read_csv(...)`` call for ``spec`` (every value escaped; settings from
    ``duckdb_functions()``: sep, header, skip, encoding, nullstr, decimal_separator,
    types, auto_type_candidates, sample_size)."""
    na = [*PANDAS_NA, *(spec.na_values or [])]
    args = [
        _lit(path),
        f"header={'true' if spec.header == 0 else 'false'}",
        f"skip={spec.skiprows}",
        f"encoding={_lit(_encoding(spec, path))}",
        f"nullstr=[{', '.join(map(_lit, dict.fromkeys(na)))}]",
        f"auto_type_candidates=[{', '.join(map(_lit, TYPE_CANDIDATES))}]",
        f"sample_size={sample}",
    ]
    sep = _sep(spec, path)
    if sep:
        args.append(f"sep={_lit(sep)}")
    if spec.decimal not in ("auto", "."):
        args.append(f"decimal_separator={_lit(spec.decimal)}")
    if types:
        pairs = ", ".join(f"{_lit(k)}: {_lit(v)}" for k, v in types.items())
        args.append(f"types={{{pairs}}}")
    return f"read_csv({', '.join(args)})"


def _pinned_types(spec: CsvSource) -> dict[str, str]:
    out = {}
    for column, dtype in (spec.dtype or {}).items():
        if dtype not in _DTYPES:
            raise SourceError(
                f"csv engine 'duckdb': dtype {dtype!r} for {column!r} is not supported "
                f"(known: {sorted(_DTYPES)}); use engine='pandas'"
            )
        out[column] = _DTYPES[dtype]
    return out


def _leading_zero_columns(con: Any, spec: CsvSource, path: str, types: dict[str, str]) -> list[str]:
    """Integer-typed columns holding a digits-only value with leading zeros (``007``)."""
    typed = con.execute(f"DESCRIBE SELECT * FROM {_read_options(spec, path, types, sample=SAMPLE_ROWS)}")
    ints = [c for c, t, *_ in typed.fetchall() if t in _INTEGER_TYPES and c not in types]
    if not ints:
        return []
    raw_types = {c: "VARCHAR" for c in ints}
    flags = ", ".join(f"bool_or(regexp_matches({_ident(c)}, '^0[0-9]+$'))" for c in ints)
    row = con.execute(
        f"SELECT {flags} FROM {_read_options(spec, path, types | raw_types, sample=SAMPLE_ROWS)}"
    ).fetchone()
    return [c for c, hit in zip(ints, row, strict=True) if hit]


def _convert(duckdb: Any, spec: CsvSource, path: str, out: str) -> None:
    types = _pinned_types(spec)
    with tempfile.TemporaryDirectory(prefix="dtk-duckdb-") as spill:
        con = duckdb.connect(
            config={
                "memory_limit": MEMORY_LIMIT,
                "temp_directory": spill,
            }
        )
        try:
            _copy(con, duckdb, spec, path, out, types)
        except duckdb.OutOfMemoryException as exc:
            raise SourceError(
                f"cannot read csv {spec.path}: DuckDB ran out of its {MEMORY_LIMIT} memory "
                f"limit ({str(exc).splitlines()[0]})"
            ) from exc
        except duckdb.Error as exc:
            raise SourceError(f"cannot read csv {spec.path}: {_reason(exc)}") from exc
        finally:
            con.close()


def _copy(con: Any, duckdb: Any, spec: CsvSource, path: str, out: str, types: dict[str, str]) -> None:
    if spec.keep_leading_zeros:
        types = types | dict.fromkeys(_leading_zero_columns(con, spec, path, types), "VARCHAR")
    cols = "*" if not spec.usecols else ", ".join(map(_ident, spec.usecols))
    for sample in (SAMPLE_ROWS, -1):  # -1: type every column from every row
        query = f"SELECT {cols} FROM {_read_options(spec, path, types, sample=sample)}"
        try:
            con.execute(f"COPY ({query}) TO {_lit(out)} (FORMAT parquet)")
            return
        except duckdb.ConversionException:
            if sample == -1:
                raise


def _reason(exc: Exception) -> str:
    """First lines of a DuckDB error, with the encoding hint when it is that."""
    text = str(exc)
    if "not utf-8 encoded" in text:
        return (
            "the file is not valid utf-8; pass encoding='latin-1' (or 'utf-16') if it "
            "is, or re-encode it (the duckdb engine never guesses an encoding)"
        )
    if "does not exist" in text and "Column" in text:
        return text.splitlines()[0] + " (usecols / dtype name a column the file lacks)"
    return " ".join(line.strip() for line in text.splitlines()[:3] if line.strip())

