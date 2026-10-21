"""Small helpers shared by the analysis and transform ops."""

import numpy as np
import pandas as pd


def pct(count: float, total: float) -> float:
    """`count` as a percentage of `total`, rounded to 2 decimals (0.0 if empty)."""
    return round(100 * count / total, 2) if total else 0.0


def py(value):
    """numpy scalar -> plain Python value (JSON-safe)."""
    return value.item() if isinstance(value, np.generic) else value


def json_scalar(value):
    """Like `py`, and NaN -> None."""
    value = py(value)
    return None if isinstance(value, float) and np.isnan(value) else value


def require_numeric(df: pd.DataFrame, columns: list[str], op: str) -> None:
    bad = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"{op} needs numeric columns, got non-numeric {bad}")


def sampling_metrics(n_source: int, n_sampled_out: int) -> dict[str, int]:
    """Row-level display-sampling metrics, same names as the ``chart`` key."""
    return {"n_rows_source": int(n_source), "n_sampled_out": int(n_sampled_out)}


def sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """``df`` itself when ``max_rows`` is None or not exceeded, else a seeded
    sample of ``max_rows`` rows in original order (same input, same sample)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_index(kind="stable")


def sampling_note(
    n_shown: int, n_sampled_out: int, what: str = "the figure", end: str = "."
) -> str:
    """One-line note when a row sample was really taken, else ``""``."""
    if not n_sampled_out:
        return ""
    return f"Sampled {n_shown} rows for {what} ({n_sampled_out} rows left out){end}"


def points_disclosure(
    entries: list[tuple[str, int, int]], unit: str
) -> tuple[dict[str, int], str]:
    """``(metrics, note)`` for point-level sampling; ``entries`` = ``(label, n_total, n_shown)``.

    Metrics describe the most truncated entry; the note names it and counts the
    sampled ones. Nothing (``({}, "")``) when there is no entry.
    """
    if not entries:
        return {}, ""
    label, total, shown = max(entries, key=lambda e: (e[1] - e[2], e[1]))
    metrics = {"n_points_shown": int(shown), "n_points_total": int(total)}
    n_sampled = sum(t > s for _, t, s in entries)
    if not n_sampled:
        return metrics, ""
    note = f"Sampled {shown} of {total} {unit} for {label} ({total - shown} left out)"
    if n_sampled > 1:
        note += f"; {n_sampled} of {len(entries)} plots are sampled, this is the most truncated"
    return metrics, note + "."
