"""CSV reader backed by pandas.read_csv."""

from __future__ import annotations

import csv
import io
import itertools
import os
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import CsvSource

# sep="auto": characters read to sniff the delimiter, and the candidates tried.
SNIFF_CHARS = 64 * 1024
SNIFF_DELIMITERS = ",;\t|"
BOMS = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
]
_COMMA_NUMBER = re.compile(r"[-+]?\d+,\d+")
_DOT_NUMBER = re.compile(r"[-+]?\d*\.\d+")
# csv (strict) error -> reason shown to the user.
_CSV_ERRORS = [
    (
        "expected after",
        "text after a closing quote (unescaped quote in a quoted field?)",
    ),
    ("unexpected end of data", "unclosed quote"),
]

# Lines read to find the junk lines above the header, and the longest one allowed.
JUNK_SCAN = 200
MAX_LISTED_LINES = 10

_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def resolve_path(
    path: str,
    *,
    platform: str = sys.platform,
    exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """Map a Windows path (`C:\\x\\f.csv`, `C:/x/f.csv`) to its WSL mount.

    Only on Linux and only when the path as given does not exist; anything else
    is returned unchanged.
    """
    match = _WINDOWS_DRIVE.match(path)
    if not match or not platform.startswith("linux") or exists(path):
        return path
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def sniff_sep(sample: str) -> str | None:
    """Delimiter guessed from a text sample, None if it cannot tell.

    Scores each candidate by how many non-empty rows share the modal field count
    (must be ≥ 2). A leading title line — even one that contains a comma — must
    not outvote a consistent multi-field body (course ``store_b.csv``: title,
    ``sep=';'``, decimal commas in prices).
    """
    sample = _drop_cut_line(sample)
    if not sample.strip():
        return None
    scored = _score_sep(sample)
    if scored is not None:
        return scored
    # No candidate yields ≥ 2 fields: a single-column file. Say so explicitly,
    # else pandas' python-engine sniffing may pick a letter as the separator.
    header = sample.split("\n", 1)[0]
    present = [d for d in SNIFF_DELIMITERS if d in header]
    if not present:
        return ","
    return present[0] if len(present) == 1 else None


def _score_sep(sample: str) -> str | None:
    """Best delimiter: most rows sharing a modal field count of at least 2."""
    lines = [ln for ln in sample.splitlines() if ln.strip()]
    if not lines:
        return None
    best: str | None = None
    best_score = (-1, -1)  # (consistent rows, modal fields)
    for sep in SNIFF_DELIMITERS:
        try:
            counts = [len(row) for row in csv.reader(lines, delimiter=sep) if row]
        except csv.Error:
            continue
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        if modal < 2:
            continue
        consistent = sum(1 for c in counts if c == modal)
        score = (consistent, modal)
        if score > best_score:
            best_score, best = score, sep
    return best


def guess_encoding(raw: bytes, *, complete: bool = True) -> str:
    """BOM, else utf-8 if the bytes decode, else cp1252 (latin-1 if even that fails).

    ``complete=False``: ``raw`` is a sample that may end mid-character.
    """
    bom = next((name for mark, name in BOMS if raw.startswith(mark)), None)
    if bom:
        return bom
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError as exc:
        # A sample only tolerates a multi-byte character cut at its very end.
        if (
            not complete
            and exc.reason == "unexpected end of data"
            and exc.start >= len(raw) - 3
        ):
            return "utf-8"
    try:
        raw.decode("cp1252")
        return "cp1252"
    except UnicodeDecodeError:
        return "latin-1"


def guess_decimal(sample: str, sep: str | None) -> str:
    """``,`` when a non-comma-separated sample holds more ``1,5`` than ``1.5`` values."""
    if sep in (None, ","):
        return "."
    comma = dot = 0
    try:
        for row in itertools.islice(
            csv.reader(io.StringIO(sample), delimiter=sep), 1000
        ):
            for field in row:
                field = field.strip()
                comma += bool(_COMMA_NUMBER.fullmatch(field))
                dot += bool(_DOT_NUMBER.fullmatch(field))
    except csv.Error:
        pass
    return "," if comma > dot else "."


def find_bad_record(
    text: str, sep: str, header: int | None, *, locate: bool = False
) -> tuple[int, str] | None:
    """First malformed record after the header, as (1-based line, reason).

    Malformed = more fields than the header row, an unclosed quote, or text after a
    closing quote: the cases pandas either rejects or silently mangles (a stray
    quote swallowing the next fields, an extra field turned into the index). Fewer
    fields are fine (padded with missing values). Blank lines are skipped and do
    not count towards ``header``, as in pandas.
    """
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=sep, strict=True)
    rows = filter(None, reader)  # drop blank lines
    try:
        for _ in range(header or 0):
            next(rows, None)
        first = next(rows, None)
        if first is not None:
            expected = len(first)
            if locate:
                return _first_wide_record(reader, rows, expected)
            # Fast path in C: the widest record; walk again only to locate it.
            if max(map(len, rows), default=0) > expected:
                return find_bad_record(text, sep, header, locate=True)
    except csv.Error as exc:
        return _csv_error_record(reader, exc)
    return None


def _first_wide_record(reader, rows, expected: int) -> tuple[int, str] | None:
    for row in rows:
        if len(row) > expected:
            return reader.line_num, f"{len(row)} fields, expected {expected}"
    return None


def _csv_error_record(reader, exc: csv.Error) -> tuple[int, str] | None:
    message = str(exc)
    if "field larger than field limit" in message:
        return None  # a huge field is not malformed; let pandas read it
    for pattern, reason in _CSV_ERRORS:
        if pattern in message:
            message = reason
    return reader.line_num, message


class MixedSeparatorError(SourceError):
    """Some lines use another delimiter than the file's; ``lines`` are 1-based."""

    def __init__(self, message: str, lines: list[int]) -> None:
        super().__init__(message)
        self.lines = lines


@dataclass
class MixedReport:
    """Lines of a csv body that fit another delimiter than ``sep``.

    ``counts``: delimiter -> lines using it (``sep``: lines with the modal field
    count). ``lines``: alt delimiter -> 1-based offending lines. An offending
    line holds no ``sep`` and no quote: a line with ``sep`` is an ordinary (ragged)
    row, whatever else it contains.
    """

    sep: str
    fields: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    lines: dict[str, list[int]] = field(default_factory=dict)

    @property
    def offending(self) -> list[int]:
        return sorted(n for nums in self.lines.values() for n in nums)


def drop_lines(text: str, count: int) -> str:
    """``text`` without its first ``count`` physical lines."""
    pos = 0
    for _ in range(count):
        nxt = text.find("\n", pos)
        if nxt < 0:
            return ""
        pos = nxt + 1
    return text[pos:]


def detect_junk(lines: list[str], seps: str) -> tuple[int, str | None]:
    """(junk line count, delimiter) of the best reading of a sample's lines.

    The body is the tail of the sample: per candidate delimiter, its width is the
    modal field count (≥ 2) of the second half of the non-blank records, and the
    header is the first record of that width; every line above it is junk. The
    candidate with the most records of body width wins (ties: ``seps`` order).
    """
    best: tuple[int, int, int, str] | None = None
    for sep in seps:
        found = _body_start(lines, sep)
        if found is None:
            continue
        start, width, rows = found
        if best is None or (rows, width) > best[:2]:
            best = (rows, width, start, sep)
    return (best[2], best[3]) if best else (0, None)


def _body_start(lines: list[str], sep: str) -> tuple[int, int, int] | None:
    """(line index of the header, body width, records of that width) for ``sep``."""
    records = [
        (i, ln.count(sep) + 1) for i, ln in enumerate(lines) if ln.strip() and '"' not in ln
    ]
    tail = [n for _, n in records[len(records) // 2 :]]
    if not tail:
        return None
    width = max(set(tail), key=tail.count)
    if width < 2:
        return None
    rows = [i for i, n in records if n == width]
    return rows[0], width, len(rows)


def _trim_and_sniff(text: str, spec: CsvSource) -> tuple[str, str | None, int]:
    """(text without the junk top lines, delimiter, lines dropped)."""
    sep: str | None = spec.sep
    skipped = spec.skiprows if isinstance(spec.skiprows, int) else 0
    if spec.skiprows == "auto":
        sample = _drop_cut_line(text[:SNIFF_CHARS]).splitlines()[:JUNK_SCAN]
        skipped, found = detect_junk(sample, SNIFF_DELIMITERS if sep == "auto" else sep)
        if sep == "auto" and found:
            sep = found
    text = drop_lines(text, skipped)
    if sep == "auto":
        sep = sniff_sep(text[:SNIFF_CHARS])
    return text, sep, skipped


def mixed_separator_report(
    lines: list[str], sep: str, header: int | None, offset: int = 0
) -> MixedReport:
    """Which body lines use another delimiter than ``sep``.

    ``lines`` start at the file's line ``offset + 1``. Only lines without quotes
    count (a quoted field can legitimately hold any delimiter); a line is
    offending when it holds no ``sep`` at all and splits into the modal field count
    under another delimiter.
    """
    start = _header_index(lines, header)
    records = _plain_records(lines, sep, start)
    report = MixedReport(sep)
    counts = [n for _, n in records]
    if not counts:
        return report
    width = max(set(counts), key=counts.count)
    report.fields = width
    report.counts[sep] = counts.count(width)
    if width < 2:
        return report
    for i, n in records:
        if n == 1:  # no ``sep`` in the line
            _classify(report, lines[i], i + 1 + offset, sep, width)
    return report


def _header_index(lines: list[str], header: int | None) -> int:
    """Index of the header line (pandas counts non-blank lines); 0 without one."""
    seen = 0
    for i, line in enumerate(lines):
        if line.strip():
            if seen == (header or 0):
                return i
            seen += 1
    return len(lines)


def _plain_records(lines: list[str], sep: str, start: int) -> list[tuple[int, int]]:
    """(line index, field count) of the non-blank lines clear of quotes."""
    out, in_quote = [], False
    for i in range(start, len(lines)):
        line = lines[i]
        quotes = line.count('"')
        if in_quote or quotes:
            in_quote ^= bool(quotes % 2)
        elif line.strip():
            out.append((i, line.count(sep) + 1))
    return out


def _classify(report: MixedReport, line: str, number: int, sep: str, width: int):
    for alt in SNIFF_DELIMITERS:
        if alt != sep and line.count(alt) + 1 == width:
            report.lines.setdefault(alt, []).append(number)
            report.counts[alt] = report.counts.get(alt, 0) + 1
            return


def _has_sepless_line(lines: list[str], sep: str) -> bool:
    """Whether a non-blank line holds no ``sep`` (the only possible offenders).

    A cheap membership pass, so the usual clean file skips the quote-aware scan.
    """
    return any(sep not in line and line.strip() for line in lines)


def _check_mixed_sep(text: str, sep: str | None, spec: CsvSource, skipped: int) -> str:
    """``text`` with mixed-delimiter lines normalized, or the typed error."""
    if not sep or spec.mixed_sep == "ignore":
        return text
    lines = text.split("\n")
    if not _has_sepless_line(lines, sep):
        return text
    report = mixed_separator_report(lines, sep, spec.header, skipped)
    if not report.lines:
        return text
    if spec.mixed_sep == "error":
        raise _mixed_error(spec.path, report)
    for alt, numbers in report.lines.items():
        for number in numbers:
            i = number - 1 - skipped
            lines[i] = lines[i].replace(alt, sep)
    return "\n".join(lines)


def _listed(numbers: list[int]) -> str:
    shown = ", ".join(map(str, numbers[:MAX_LISTED_LINES]))
    more = len(numbers) - MAX_LISTED_LINES
    return shown + (f" (and {more} more)" if more > 0 else "")


def _mixed_error(path: str, report: MixedReport) -> MixedSeparatorError:
    sep = report.sep
    parts = [
        f"{alt!r} on line(s) {_listed(nums)}" for alt, nums in report.lines.items()
    ]
    fix = "pass mixed_sep='normalize' to rewrite them, or mixed_sep='ignore' to load as is"
    return MixedSeparatorError(
        f"cannot read csv {path}: mixed separators: the file uses {sep!r} "
        f"({report.fields} fields) but {'; '.join(parts)} use another delimiter; {fix}",
        report.offending,
    )


@reader("csv")
def read_csv(spec: CsvSource) -> pd.DataFrame:
    path = resolve_path(spec.path)
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        encoding = guess_encoding(raw) if spec.encoding == "auto" else spec.encoding
        # A BOM is never content (utf-8 / utf-16 codecs without -sig keep it).
        text = raw.decode(encoding).removeprefix("\ufeff")
        del raw
        text, sep, skipped = _trim_and_sniff(text, spec)
        text = _check_mixed_sep(text, sep, spec, skipped)
        decimal = spec.decimal
        if decimal == "auto":
            decimal = guess_decimal(_drop_cut_line(text[:SNIFF_CHARS]), sep)
        if spec.on_bad_lines == "error" and sep:
            bad = find_bad_record(text, sep, spec.header)
            if bad:
                line, reason = bad
                raise SourceError(
                    f"cannot read csv {spec.path}: line {line + skipped}: {reason}. Fix the "
                    "quoting, or pass on_bad_lines='warn' / 'skip' to load anyway "
                    "(rows with too many fields are dropped)"
                )
        options = {
            # Sniffer failed: let pandas' python engine sniff the whole file.
            "sep": sep,
            "engine": "c" if sep else "python",
            "decimal": decimal,
            "header": spec.header,
            "na_values": spec.na_values,
            "usecols": spec.usecols,
            "on_bad_lines": spec.on_bad_lines,
        }
        df = pd.read_csv(
            io.StringIO(text), dtype=spec.dtype, parse_dates=spec.parse_dates, **options
        )
        if spec.keep_leading_zeros:
            df = _keep_leading_zeros(df, text, sep, spec, options)
        return df
    except MixedSeparatorError:
        raise
    except FileNotFoundError:
        raise SourceError(f"csv source not found: {spec.path}") from None
    # ParserError, EmptyDataError and UnicodeDecodeError are ValueErrors;
    # LookupError = unknown encoding; csv.Error = sniffing failed (sep="auto").
    except (OSError, ValueError, LookupError, csv.Error) as exc:
        raise SourceError(f"cannot read csv {spec.path}: {exc}") from exc


def is_leading_zero_column(values: pd.Series) -> bool:
    """Digits-only strings, one at least with a leading zero (ZIP, id): numeric
    parsing would drop the zeros."""
    return bool(
        len(values)
        and values.str.fullmatch(r"\d+").all()
        and values.str.fullmatch(r"0\d+").any()
    )


def _keep_leading_zeros(
    df: pd.DataFrame, text: str, sep: str | None, spec: CsvSource, options: dict
) -> pd.DataFrame:
    """Numeric columns whose raw values carry leading zeros go back to strings."""
    seps = re.escape(sep) if sep else re.escape(SNIFF_DELIMITERS)
    if not re.search(rf'(?:^|[{seps}])"?0\d', text, re.MULTILINE):
        return df  # cheap exit: no value starts with 0 then a digit
    pinned = set(spec.dtype or {})
    numeric = [
        i
        for i, (name, dtype) in enumerate(df.dtypes.items())
        if name not in pinned
        and pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
    ]
    if not numeric:
        return df
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # on_bad_lines="warn" already warned once
        # Only the numeric columns, by name if the spec already selects columns.
        usecols = [df.columns[i] for i in numeric] if spec.usecols else numeric
        raw = pd.read_csv(
            io.StringIO(text), dtype=str, **(options | {"usecols": usecols})
        )
    if raw.shape != (len(df), len(numeric)):
        return df
    for j, i in enumerate(numeric):
        if is_leading_zero_column(raw.iloc[:, j].dropna()):
            df.isetitem(i, raw.iloc[:, j])
    return df


def _drop_cut_line(sample: str) -> str:
    """A sample cut by the size limit without its last (partial) line."""
    if len(sample) >= SNIFF_CHARS and "\n" in sample:
        return sample[: sample.rindex("\n")]
    return sample
