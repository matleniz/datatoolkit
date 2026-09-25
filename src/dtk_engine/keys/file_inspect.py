"""Raw look at a data file before parsing: bytes, BOM, encoding, delimiter, header."""

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
from dtk_engine.sources.csv_pandas import SNIFF_CHARS, resolve_path, sniff_sep

PREVIEW_BYTES = 120
BOMS = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
JSON_MAX_BYTES = 50_000_000  # bigger files are not parsed just to suggest a path


class Params(KeyParams):
    path: str = Field(default=TRAIN_CSV, description="Local path to the file")


@key(
    id="file_inspect",
    title="File inspect",
    category="analysis",
    description=(
        "Raw file facts: first bytes, BOM, encoding guess, line endings, delimiter, "
        "likely header line, empty `Unnamed:` columns, size and mtime; sheets for Excel."
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
    if Path(path).suffix.lower() in EXCEL_SUFFIXES or raw[:4] == b"PK\x03\x04":
        result.add_table("sheets", _excel_sheets(path, params.path))
        return result
    result.metrics["first_bytes"] = repr(raw[:PREVIEW_BYTES])
    result.metrics.update(_text_facts(raw))
    if Path(path).suffix.lower() == ".json" and stat.st_size <= JSON_MAX_BYTES:
        _add_record_paths(result, path)
    return result


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
    """First row with the modal non-empty cell count (title / banner rows have fewer)."""
    counts = raw.head(50).notna().sum(axis=1).tolist()
    modal = max(set(counts) - {0}, key=counts.count, default=None)
    return counts.index(modal) if modal is not None else 0


def _text_facts(raw: bytes) -> dict:
    bom = next((name for mark, name in BOMS if raw.startswith(mark)), None)
    if bom in ("utf-16-le", "utf-16-be"):
        encoding = bom
    else:
        try:
            # A sample cut mid-character must not read as invalid utf-8.
            raw.decode("utf-8", errors="strict" if len(raw) < SNIFF_CHARS else "ignore")
            encoding = "utf-8-sig" if bom else "utf-8"
        except UnicodeDecodeError:
            encoding = "cp1252"
    text = raw.decode(encoding if bom != "utf-8-sig" else "utf-8-sig", errors="replace")
    crlf, lf = text.count("\r\n"), text.count("\n") - text.count("\r\n")
    cr = text.count("\r") - crlf
    endings = [n for n, c in (("CRLF", crlf), ("LF", lf), ("CR", cr)) if c]
    lines = text.splitlines()
    if len(raw) >= SNIFF_CHARS and lines:
        lines = lines[:-1]  # last line is cut by the sample limit
    sep = sniff_sep("\n".join(lines) + "\n") if lines else None
    facts = {
        "bom": bom or "none",
        "encoding_guess": encoding,
        "line_endings": "+".join(endings) or "none",
        "delimiter": repr(sep) if sep else "unknown",
    }
    header = _header_line(lines, sep)
    if header is not None:
        facts["header_line"] = header + 1
        facts["title_lines_above_header"] = header
        if sep:
            facts["unnamed_columns"] = ", ".join(_unnamed_columns(lines[header], sep))
    return facts


def _header_line(lines: list[str], sep: str | None) -> int | None:
    """Index of the first line with the modal field count (title lines have fewer)."""
    if not lines:
        return None
    if not sep:
        return 0
    counts = [len(line.split(sep)) for line in lines[:50]]
    modal = max(set(counts), key=counts.count)
    return counts.index(modal)


def _unnamed_columns(header: str, sep: str) -> list[str]:
    """Pandas' names for empty header cells (``Unnamed: 3``)."""
    return [
        f"Unnamed: {i}"
        for i, name in enumerate(header.split(sep))
        if not re.sub(r'[\s"]', "", name)
    ]
