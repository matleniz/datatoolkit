"""Label join preview: X and y side by side, join candidates ranked before joining."""

import pandas as pd

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.ops.join import key_candidates, key_diagnostics, order_diagnostics
from dtk_engine.params import KeyParams, columns_field
from dtk_engine.registry import key
from dtk_engine.result import Result
from dtk_engine.sources import CsvSource, SourceSpec, load

HEAD_ROWS = 5
SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


class Params(KeyParams):
    # Demo: the train table split into features (X) and labels (y, id + Survived).
    x: SourceSpec = CsvSource(
        path=TRAIN_CSV,
        usecols=[
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ],
    )
    y: SourceSpec = CsvSource(path=TRAIN_CSV, usecols=["PassengerId", "Survived"])
    key_columns: list[str] | None = columns_field(
        "Candidate key columns tried for a join by key; "
        "null = auto (columns shared by X and y that are (nearly) unique or index-like)",
        source="x",
        nullable=True,
    )


@key(
    id="label_join_preview",
    title="Label join preview",
    category="analysis",
    description="X and y side by side before the label join: shape, columns, head "
    "rows, then every join candidate (by order, and by each shared key column) "
    "with uniqueness, match rates, resulting row count and row-multiplying "
    "duplicates, a recommended mode + key and severity-ranked issues.",
)
def run(params: Params) -> Result:
    return preview_result(load(params.x), load(params.y), params.key_columns)


def preview_result(
    x: pd.DataFrame, y: pd.DataFrame, key_columns: list[str] | None = None
) -> Result:
    """The key's Result on two DataFrames (shared with ``dtk_engine.api.label_join_preview``)."""
    keys = key_candidates(x, y, key_columns)
    usable = [k for k in keys if k in x.columns and k in y.columns]
    key_diags = [key_diagnostics(x, y, k) for k in usable]
    two_cols = len(y.columns) == 2
    order = order_diagnostics(x, y, usable[0] if usable and two_cols else None)
    mode, rec_key = _recommend(order, key_diags)
    issues = _issues(keys, usable, order, key_diags, mode)

    result = Result(
        headline=_headline(mode, rec_key, x, y, order, key_diags),
        metrics={
            "x_rows": len(x),
            "x_columns": x.shape[1],
            "y_rows": len(y),
            "y_columns": y.shape[1],
            "n_key_candidates": len(usable),
            "recommended_mode": mode,
            "recommended_key": rec_key or "",
            "n_issues": len(issues),
            "n_errors": sum(i["severity"] == "error" for i in issues),
            "n_warnings": sum(i["severity"] == "warning" for i in issues),
        },
    )
    result.add_table("candidates", _candidates_table(order, key_diags, mode, rec_key))
    result.add_table("issues", pd.DataFrame(issues, columns=ISSUE_COLUMNS))
    result.add_table("shapes", _shapes_table(x, y))
    result.add_table("columns", _columns_table(x, y))
    result.add_table("head", _side_by_side(x, y))
    return result


ISSUE_COLUMNS = ["severity", "check", "candidate", "message"]


def _recommend(order: dict, key_diags: list[dict]) -> tuple[str, str | None]:
    """(mode, key) of the safest join that ``join_labels`` accepts; ("none", best
    partial key or None) when no candidate joins cleanly."""
    clean = [d for d in key_diags if d["would_join"]]
    if clean:
        return "key", clean[0]["key"]
    if order["would_join"]:
        return "order", order["index_column"]
    if key_diags:
        best = max(key_diags, key=lambda d: min(d["match_x_to_y"], d["match_y_to_x"]))
        return "none", best["key"]
    return "none", None


def _issues(
    keys: list[str],
    usable: list[str],
    order: dict,
    key_diags: list[dict],
    mode: str,
) -> list[dict]:
    # A failing candidate is only an error when nothing else joins cleanly.
    level = "error" if mode == "none" else "warning"
    out: list[dict] = []

    def add(severity: str, check: str, candidate: str, message: str) -> None:
        out.append(
            {
                "severity": severity,
                "check": check,
                "candidate": candidate,
                "message": message,
            }
        )

    for k in keys:
        if k not in usable:
            add("error", "key_missing", k, f"column {k!r} is not in both X and y")
    for d in key_diags:
        _key_issues(d, level, add)
    _order_issues(order, level, add, has_keys=bool(usable))
    if not keys:
        add("info", "no_key", "key", "X and y share no plausible key column")
    out.sort(key=lambda i: SEVERITY_RANK[i["severity"]])
    return out


def _key_issues(d: dict, level: str, add) -> None:
    cand = f"key:{d['key']}"
    if d["y_duplicated"]:
        add(
            level,
            "duplicates",
            cand,
            f"{d['y_duplicated']} duplicated key(s) in y would multiply rows "
            f"(left join gives {d['result_rows']} rows from {d['x_rows']})",
        )
    if d["x_duplicated"]:
        add(level, "duplicates", cand, f"{d['x_duplicated']} duplicated key(s) in X")
    if d["x_unmatched"]:
        add(
            level,
            "unmatched",
            cand,
            f"{d['x_unmatched']} X row(s) have no match in y "
            f"(match rate X->y {d['match_x_to_y']:.1%})",
        )
    if d["y_unmatched"]:
        add(
            level,
            "unmatched",
            cand,
            f"{d['y_unmatched']} y row(s) have no match in X "
            f"(match rate y->X {d['match_y_to_x']:.1%})",
        )
    for side in ("x", "y"):
        if d[f"{side}_missing"]:
            add(
                level,
                "missing_key",
                cand,
                f"{d[f'{side}_missing']} missing key value(s) in {side.upper() if side == 'x' else side}",
            )
    if d["clashing_columns"]:
        add(
            level,
            "column_clash",
            cand,
            f"y column(s) {d['clashing_columns']} already exist in X",
        )
    if not d["value_columns"]:
        add(level, "no_value", cand, f"y has no column besides {d['key']!r}")


def _order_issues(order: dict, level: str, add, *, has_keys: bool) -> None:
    if order["value_error"]:
        add(level, "order", "order", order["value_error"])
        return
    if not order["same_rows"]:
        add(
            level,
            "row_count",
            "order",
            f"X has {order['x_rows']} rows, y has {order['y_rows']}",
        )
    if order["aligned"] is False:
        add(
            level,
            "misaligned",
            "order",
            f"index column {order['index_column']!r} differs on {order['n_misaligned']} "
            f"rows (first at row {order['first_misaligned']}); rows are not aligned",
        )
    elif order["same_rows"] and order["aligned"] is None:
        add(
            "info",
            "unverified_order",
            "order",
            "same row count, but no shared id column to check the order against"
            + ("; prefer a join by key" if has_keys else ""),
        )
    if order["clashing_columns"]:
        add(
            level,
            "column_clash",
            "order",
            f"y column(s) {order['clashing_columns']} already exist in X",
        )


def _headline(mode, rec_key, x, y, order, key_diags) -> str:
    shape = f"X {x.shape[0]}x{x.shape[1]}, y {y.shape[0]}x{y.shape[1]}"
    if mode == "key":
        d = next(d for d in key_diags if d["key"] == rec_key)
        return f"{shape}: join by key {rec_key!r} is clean ({d['result_rows']} rows)"
    if mode == "order":
        how = (
            f"aligned on {order['index_column']!r}"
            if order["aligned"]
            else "same row count"
        )
        return f"{shape}: join by order is possible ({how})"
    return f"{shape}: no join candidate works without losing or multiplying rows"


def _candidates_table(order, key_diags, mode, rec_key) -> pd.DataFrame:
    rows = [
        {
            "mode": "order",
            "key": order["index_column"],
            "would_join": order["would_join"],
            "recommended": mode == "order",
            "result_rows": order["x_rows"] if order["would_join"] else None,
            "same_rows": order["same_rows"],
            "aligned": order["aligned"],
            "x_unique": None,
            "y_unique": None,
            "x_duplicated": None,
            "y_duplicated": None,
            "match_x_to_y": None,
            "match_y_to_x": None,
            "extra_rows": None,
        }
    ]
    rows.extend(
        {
            "mode": "key",
            "key": d["key"],
            "would_join": d["would_join"],
            "recommended": mode == "key" and d["key"] == rec_key,
            "result_rows": d["result_rows"],
            "same_rows": d["x_rows"] == d["y_rows"],
            "aligned": None,
            **{
                k: d[k]
                for k in (
                    "x_unique",
                    "y_unique",
                    "x_duplicated",
                    "y_duplicated",
                    "match_x_to_y",
                    "match_y_to_x",
                    "extra_rows",
                )
            },
        }
        for d in key_diags
    )
    return pd.DataFrame(rows)


def _shapes_table(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"side": "X", "rows": len(x), "columns": x.shape[1]},
            {"side": "y", "rows": len(y), "columns": y.shape[1]},
        ]
    )


def _columns_table(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "side": side,
            "column": str(c),
            "dtype": str(df[c].dtype),
            "n_unique": int(df[c].nunique()),
            "pct_missing": float(df[c].isna().mean() * 100) if len(df) else 0.0,
            "shared": c in other.columns,
        }
        for side, df, other in (("X", x, y), ("y", y, x))
        for c in df.columns
    ]
    return pd.DataFrame(rows)


def _side_by_side(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    hx = x.head(HEAD_ROWS).add_prefix("X: ").reset_index(drop=True)
    hy = y.head(HEAD_ROWS).add_prefix("y: ").reset_index(drop=True)
    return pd.concat([hx, hy], axis=1)
