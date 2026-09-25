"""CSV reader backed by pandas.read_csv."""

from __future__ import annotations

import csv
import os
import re
import sys
from collections.abc import Callable

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import CsvSource

# sep="auto": characters read to sniff the delimiter, and the candidates tried.
SNIFF_CHARS = 64 * 1024
SNIFF_DELIMITERS = ",;\t|"

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
    """Delimiter detected by csv.Sniffer on a text sample, None if it cannot tell."""
    # Drop a line cut in the middle by the sample limit.
    if len(sample) >= SNIFF_CHARS and "\n" in sample:
        sample = sample[: sample.rindex("\n")]
    try:
        return csv.Sniffer().sniff(sample, delimiters=SNIFF_DELIMITERS).delimiter
    except csv.Error:
        pass
    # No candidate on the header line: a single-column file. Say so explicitly,
    # else pandas' python-engine sniffing may pick a letter as the separator.
    header = sample.split("\n", 1)[0]
    if sample and not any(d in header for d in SNIFF_DELIMITERS):
        return ","
    return None


@reader("csv")
def read_csv(spec: CsvSource) -> pd.DataFrame:
    path = resolve_path(spec.path)
    try:
        sep: str | None = spec.sep
        if sep == "auto":
            with open(path, encoding=spec.encoding, newline="") as fh:
                sep = sniff_sep(fh.read(SNIFF_CHARS))
        return pd.read_csv(
            path,
            # Sniffer failed: let pandas' python engine sniff the whole file.
            sep=sep,
            engine="c" if sep else "python",
            encoding=spec.encoding,
            decimal=spec.decimal,
            header=spec.header,
            na_values=spec.na_values,
            dtype=spec.dtype,
            parse_dates=spec.parse_dates,
            usecols=spec.usecols,
        )
    except FileNotFoundError:
        raise SourceError(f"csv source not found: {spec.path}") from None
    # ParserError, EmptyDataError and UnicodeDecodeError are ValueErrors;
    # LookupError = unknown encoding; csv.Error = sniffing failed (sep="auto").
    except (OSError, ValueError, LookupError, csv.Error) as exc:
        raise SourceError(f"cannot read csv {spec.path}: {exc}") from exc
