"""Row-level recommendations: rows without a target, exact duplicates."""

from __future__ import annotations

import pandas as pd

from dtk_engine.ops.advisor.common import ROWS, Rec
from dtk_engine.ops.duplicates import exact_duplicates


def row_recs(train: pd.DataFrame, target: str | None) -> list[Rec]:
    recs = []
    if target is not None and train[target].isna().any():
        n = int(train[target].isna().sum())
        recs.append(
            Rec(
                ROWS,
                "rows",
                "warning",
                f"{n} train rows have no target {target!r}: drop them first",
                "drop_missing_target",
                "train",
                {"target": target},
            )
        )
    n_dup, _ = exact_duplicates(train)
    if n_dup:
        recs.append(
            Rec(
                ROWS,
                "rows",
                "warning",
                f"{n_dup} exact duplicate train rows: keep one copy each "
                "(identical rows, so which one is irrelevant)",
                "drop_duplicates",
                "train",
                {"keep": "first", "sort_by": [str(train.columns[0])]},
            )
        )
    return recs
