"""JSON / JSON lines reader. Nested objects are flattened with ``json_normalize``
(``a_b``); list-valued fields are left as-is."""

from __future__ import annotations

import json

import pandas as pd

from dtk_engine.errors import SourceError
from dtk_engine.sources.csv_pandas import resolve_path
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import JsonSource


def _records(spec: JsonSource, path: str) -> list | dict:
    with open(path, encoding=spec.encoding) as fh:
        if spec.lines:
            return [json.loads(line) for line in fh if line.strip()]
        return json.load(fh)


@reader("json")
def read_json(spec: JsonSource) -> pd.DataFrame:
    path = resolve_path(spec.path)
    try:
        data = _records(spec, path)
        for key in spec.record_path.split(".") if spec.record_path else []:
            data = data[key]
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise TypeError("expected a list of records")
        return pd.json_normalize(data, sep="_")
    except FileNotFoundError:
        raise SourceError(f"json source not found: {spec.path}") from None
    except LookupError:
        raise SourceError(f"unknown encoding {spec.encoding!r}") from None
    # JSONDecodeError / UnicodeDecodeError are ValueErrors.
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SourceError(f"cannot read json {spec.path}: {exc}") from exc
