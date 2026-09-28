"""Raw look at a data file before parsing: bytes, BOM, encoding, delimiter, header."""

import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import Field

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import SourceError
from dtk_engine.params import KeyParams
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources.csv_pandas import (
    BOMS,
    SNIFF_CHARS,
    find_bad_record,
    guess_decimal,
    guess_encoding,
    is_leading_zero_column,
    resolve_path,
    sniff_sep,
)

PREVIEW_BYTES = 120
# Records read to guess the header / leading zeros.
HEADER_SCAN = 200
NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
PARQUET_SUFFIXES = {".parquet"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
JSON_LINES_SUFFIXES = {".jsonl", ".ndjson"}
JSON_MAX_BYTES = 50_000_000  # bigger files are not parsed just to suggest a path
PARQUET_MAGIC = b"PAR1"
ZIP_MAGIC = b"PK\x03\x04"


class Params(KeyParams):
    path: str = Field(default=TRAIN_CSV, description="Local path to the file")


@key(
    id="file_inspect",
    title="File inspect",
    category="analysis",
    description=(
        "Raw file facts: first bytes, BOM, encoding / decimal guesses, line endings, "
        "delimiter, header line (or none), empty `Unnamed:` columns, leading-zero "
        "columns, first malformed line, size and mtime, and a `load_spec` applying "
        "the guesses (csv / parquet / excel / json); sheets for Excel; record paths "
        "for enveloped JSON."
    ),
)
def run(params: Params) -> Result:
    path = resolve_path(params.path)
    try:
        stat = os.stat(path)
        with open(path, "rb") as fh:
            raw = fh.read(SNIFF_CHARS)
    except FileNotFoundError:
        raise SourceError(f"file not found: {params.path}") from None
    except OSError as exc:
        raise SourceError(f"cannot read {params.path}: {exc}") from exc

    result = Result(
        metrics={
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(
                timespec="seconds"
            ),
        }
    )
    suffix = Path(path).suffix.lower()
    if suffix in PARQUET_SUFFIXES or raw[:4] == PARQUET_MAGIC:
        result.metrics["load_spec"] = json.dumps(
            {"kind": "parquet", "path": params.path}
        )
        return result
    if suffix in EXCEL_SUFFIXES or raw[:4] == ZIP_MAGIC:
        sheets = _excel_sheets(path, params.path)
        result.add_table("sheets", sheets)
        result.metrics["load_spec"] = json.dumps(_excel_load_spec(params.path, sheets))
        return result
    if suffix in JSON_SUFFIXES:
        result.metrics.update(
            _json_facts(path, params.path, suffix, stat.st_size, result)
        )
        return result
    result.metrics["first_bytes"] = repr(raw[:PREVIEW_BYTES])
    result.metrics.update(_text_facts(raw, params.path))
    return result


def _excel_load_spec(shown: str, sheets: pd.DataFrame) -> dict:
    """Default excel load_spec: first sheet + its suggested header row."""
    if sheets.empty:
        return {"kind": "excel", "path": shown, "sheet": 0, "header": 0}
    first = sheets.iloc[0]
    return {
        "kind": "excel",
        "path": shown,
        "sheet": first["sheet"],
        "header": int(first["suggested_header"]),
    }


def _json_facts(
    path: str, shown: str, suffix: str, size: int, result: Result
) -> dict:
    """load_spec for JSON / JSONL; optional record_paths table for envelopes."""
    lines = suffix in JSON_LINES_SUFFIXES
    spec: dict = {"kind": "json", "path": shown, "lines": lines}
    facts: dict = {}
    if not lines and size <= JSON_MAX_BYTES:
        _add_record_paths(result, path)
        suggested = result.metrics.pop("suggested_record_path", None)
        if suggested is not None:
            facts["suggested_record_path"] = suggested
            spec["record_path"] = suggested
    facts["load_spec"] = json.dumps(spec)
    return facts


def _add_record_paths(result: Result, path: str) -> None:
    """Suggest ``record_path`` when the records sit under an API-style envelope."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    found = sorted(_record_lists(data, ""), key=lambda item: -item[1])
    if found:
        result.metrics["suggested_record_path"] = found[0][0]
        result.add_table(
            "record_paths", pd.DataFrame(found, columns=["record_path", "records"])
        )


def _record_lists(node: dict, prefix: str) -> list[tuple[str, int]]:
    """(dotted path, length) of every non-empty list of objects under dict keys."""
    out = []
    for name, value in node.items():
        here = f"{prefix}{name}"
        if (
            isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        ):
            out.append((here, len(value)))
        elif isinstance(value, dict):
            out.extend(_record_lists(value, f"{here}."))
    return out


def _excel_sheets(path: str, shown: str) -> pd.DataFrame:
    """One row per sheet; ``suggested_header`` is a 0-based value for ``header``."""
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    except Exception as exc:
        raise SourceError(f"cannot read excel {shown}: {exc}") from exc
    return pd.DataFrame(
        [
            {
                "sheet": n,
                "rows": d.shape[0],
                "cols": d.shape[1],
                "suggested_header": _excel_header_row(d),
            }
            for n, d in sheets.items()
        ]
    )


def _excel_header_row(raw: pd.DataFrame) -> int:
    """First row at least as full as the modal row, preferring mostly-text rows.

    Title / banner rows are sparser than the modal count; data rows with missing
    cells can be sparser than the header, so the header only needs to reach it.
    """
    head = raw.head(50)
    counts = head.notna().sum(axis=1).tolist()
    modal = max(set(counts) - {0}, key=counts.count, default=None)
    if modal is None:
        return 0
    full = [i for i, c in enumerate(counts) if c >= modal]
    for i in full:
        cells = head.iloc[i].dropna()
        if sum(isinstance(v, str) for v in cells) * 2 > len(cells):
            return i
    return full[0]


def _text_facts(raw: bytes, shown: str) -> dict:
    bom = next((name for mark, name in BOMS if raw.startswith(mark)), None)
    encoding = guess_encoding(raw, complete=len(raw) < SNIFF_CHARS)
    text = raw.decode(encoding, errors="replace").removeprefix("\ufeff")
    crlf, lf = text.count("\r\n"), text.count("\n") - text.count("\r\n")
    cr = text.count("\r") - crlf
    endings = [n for n, c in (("CRLF", crlf), ("LF", lf), ("CR", cr)) if c]
    lines = text.splitlines()
    if len(raw) >= SNIFF_CHARS and lines:
        lines = lines[:-1]  # last line is cut by the sample limit
    sample = "\n".join(lines) + "\n" if lines else ""
    sep = sniff_sep(sample) if lines else None
    decimal = guess_decimal(sample, sep)
    facts = {
        "bom": bom or "none",
        "encoding_guess": encoding,
        "line_endings": "+".join(endings) or "none",
        "delimiter": repr(sep) if sep else "unknown",
        "decimal_guess": decimal,
    }
    spec: dict = {"kind": "csv", "path": shown, "sep": sep or "auto"}
    spec |= {"encoding": encoding, "decimal": decimal}
    records = _records(lines, sep)
    header = _header_record(records)
    if header is None:
        facts["header_guess"] = "none"
        spec["header"] = None
    elif not _has_header(records, header, decimal):
        facts["header_guess"] = "none"
        spec["header"] = None
        # header=None reads title lines as data: report them so they can be dropped.
        if header:
            facts["title_lines_above_data"] = records[header][0]
    else:
        line, fields = records[header]
        facts["header_guess"] = "present"
        facts["header_line"] = line + 1
        facts["title_lines_above_header"] = line
        facts["unnamed_columns"] = ", ".join(
            f"Unnamed: {i}" for i, name in enumerate(fields) if not name.strip()
        )
        spec["header"] = sum(1 for _, f in records[:header] if f)
        zeros = _leading_zero_columns(records[header:], fields)
        facts["leading_zero_columns"] = ", ".join(zeros)
        if zeros:
            spec["dtype"] = dict.fromkeys(zeros, "str")
    if sep:
        bad = find_bad_record(sample, sep, spec["header"])
        if bad and bad[1] == "unclosed quote" and len(raw) >= SNIFF_CHARS:
            bad = None  # a quoted field cut by the sample limit
        facts["bad_line"] = f"line {bad[0]}: {bad[1]}" if bad else "none"
    facts["load_spec"] = json.dumps(spec)
    return facts


def _records(lines: list[str], sep: str | None) -> list[tuple[int, list[str]]]:
    """(0-based start line, fields) of the first records, quotes honoured like the
    csv reader: a quoted ``,`` does not split, a quoted newline spans lines."""
    if not sep:
        return [(i, [line]) for i, line in enumerate(lines[:HEADER_SCAN])]
    reader = csv.reader(lines, delimiter=sep)
    records, start = [], 0
    try:
        for fields in reader:
            records.append((start, fields))
            start = reader.line_num
            if len(records) >= HEADER_SCAN:
                break
    except csv.Error:
        pass  # malformed tail: judge on the records read so far
    return records


def _header_record(records: list[tuple[int, list[str]]]) -> int | None:
    """Index of the first record with the modal field count (title lines have fewer)."""
    counts = [len(fields) for _, fields in records if fields]
    if not counts:
        return None
    # Ties (tiny files): the count seen first.
    modal = max(counts, key=lambda c: (counts.count(c), -counts.index(c)))
    return next(i for i, (_, fields) in enumerate(records) if len(fields) == modal)


def _has_header(
    records: list[tuple[int, list[str]]], header: int, decimal: str
) -> bool:
    """False when the would-be header looks like data: every column numeric in the
    rows below is numeric on that line too (UCI dumps: ``39,State-gov,77516``)."""
    first = records[header][1]
    data = [fields for _, fields in records[header + 1 :] if fields]
    numeric = [
        i
        for i in range(len(first))
        if _mostly_numeric([row[i] for row in data if i < len(row)], decimal)
    ]
    if not numeric:
        return True  # all-text columns: cannot tell, assume a header
    return not all(_is_number(first[i], decimal) for i in numeric)


def _mostly_numeric(values: list[str], decimal: str) -> bool:
    values = [v for v in values if v.strip()]
    return bool(values) and sum(_is_number(v, decimal) for v in values) >= 0.8 * len(
        values
    )


def _is_number(value: str, decimal: str) -> bool:
    value = value.strip()
    if decimal == ",":
        value = value.replace(",", ".")
    return bool(NUMBER.fullmatch(value))


def _leading_zero_columns(
    records: list[tuple[int, list[str]]], header: list[str]
) -> list[str]:
    """Header names of the columns whose sample values would lose leading zeros."""
    rows = [fields for _, fields in records[1:] if fields]
    return [
        name
        for i, name in enumerate(header)
        if is_leading_zero_column(
            pd.Series(
                [row[i].strip() for row in rows if i < len(row) and row[i].strip()],
                dtype=str,
            )
        )
    ]
