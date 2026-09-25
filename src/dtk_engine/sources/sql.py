"""SQL reader: the query runs on the server through SQLAlchemy.

The connection URL is never part of the spec (it would end up in stored
workspaces and logs): the spec names an environment variable holding it.
"""

from __future__ import annotations

import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from dtk_engine.errors import SourceError
from dtk_engine.sources.registry import reader
from dtk_engine.sources.spec import SqlSource


@reader("sql")
def read_sql(spec: SqlSource) -> pd.DataFrame:
    url = os.environ.get(spec.url_env)
    if not url:
        raise SourceError(f"environment variable {spec.url_env!r} is not set")
    engine = None
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            # Driver-level: a ':' in the query is not a bind parameter.
            result = conn.exec_driver_sql(spec.query)
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    except (SQLAlchemyError, ValueError) as exc:
        # Report the driver error, not the URL (may hold credentials).
        raise SourceError(
            f"cannot run sql query from ${spec.url_env}: {type(exc).__name__}: "
            f"{getattr(exc, 'orig', None) or exc.args[:1]}"
        ) from None
    finally:
        if engine is not None:
            engine.dispose()
