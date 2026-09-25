"""Per-column distributions split into groups (train / test side, label class).

Numeric columns: histogram on bins shared by every group (shares comparable
across groups) + a summary per group. Categorical columns: value counts per
group on the top-k values of all groups together, the rest as ``(other)``.

Pure pandas / numpy, no ``Result``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from dtk_engine.ops.profile import HIST_BINS, MISSING_LABEL
from dtk_engine.ops.selection import infer_task

OTHER_LABEL = "(other)"
TOP_K = 10
# A numeric target split by label: this many quantile bins.
TARGET_BINS = 4
GROUP = "group"

HISTOGRAM_FIELDS = ["column", GROUP, "bin_left", "bin_right", "count", "share"]
SUMMARY_FIELDS = [
    "column",
    GROUP,
    "count",
    "n_missing",
    "pct_missing",
    "mean",
    "std",
    "min",
    "q1",
    "median",
    "q3",
    "max",
]
COUNT_FIELDS = ["column", GROUP, "value", "count", "pct"]


def value_label(value) -> str:
    """Display label of one value: missing -> ``(missing)``, 1.0 -> ``1``."""
    if pd.isna(value):
        return MISSING_LABEL
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def label_binner(
    y: pd.Series, bins: int = TARGET_BINS
) -> Callable[[pd.Series], pd.Series]:
    """Map a target to group labels: its classes, or (regression target) quantile
    bins of ``y`` whose outer edges are open (values of another frame always fit)."""
    if infer_task(y) == "classification":
        return lambda s: s.map(value_label)
    values = y.dropna().astype(float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])

    def binned(s: pd.Series) -> pd.Series:
        cut = pd.cut(s.astype(float), edges)
        return cut.astype(str).where(cut.notna(), MISSING_LABEL)

    return binned


def grouped_frame(
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    target: str | None = None,
    binner: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame:
    """Rows of every frame stacked with a ``group`` column: the frame name, the
    label class (``binner`` of ``target``), or ``"<frame> / <class>"`` with
    several frames. A frame without ``target`` keeps its name as group; columns
    absent from a frame are missing there.

    ``group`` is an ordered categorical: frames in order, classes sorted within.
    """
    parts, order = [], []
    for side, df in frames.items():
        part = df.reindex(columns=columns).reset_index(drop=True)
        if binner is not None and target in df.columns:
            classes = binner(df[target]).reset_index(drop=True)
            prefix = "" if len(frames) == 1 else side + " / "
            part[GROUP] = prefix + classes
            order += [prefix + c for c in sorted(classes.unique(), key=natural_key)]
        else:
            part[GROUP] = side
            order.append(side)
        parts.append(part)
    data = pd.concat(parts, ignore_index=True)
    data[GROUP] = pd.Categorical(data[GROUP], categories=order, ordered=True)
    return data


def group_order(groups: pd.Series) -> list[str]:
    """The groups of a ``grouped_frame``, in order."""
    return [str(g) for g in groups.cat.categories]


def natural_key(label: str):
    """Sort key of a class label: "2" as a number, a bin "(22.0, 35.5]" by its
    left edge, anything else as text after the numbers."""
    number = label[1:].split(",")[0] if label[:1] in "([" and "," in label else label
    try:
        return (0, float(number), "")
    except ValueError:
        return (1, 0.0, label)


def numeric_distribution(
    data: pd.DataFrame, column: str, bins: int = HIST_BINS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(histogram, summary)`` of ``column`` per group, on shared bins.

    ``share`` = count / non-missing rows of the group (each group sums to 1).
    """
    values = pd.to_numeric(data[column], errors="coerce")
    groups = group_order(data[GROUP])
    all_values = values.dropna().to_numpy(dtype=float)
    edges = np.histogram_bin_edges(all_values, bins=bins) if len(all_values) else None
    hist, summary = [], []
    for group in groups:
        v = values[data[GROUP] == group]
        present = v.dropna().astype(float)
        n = len(present)
        q = present.quantile([0.25, 0.5, 0.75]).to_numpy() if n else [np.nan] * 3
        summary.append(
            {
                "column": column,
                GROUP: group,
                "count": n,
                "n_missing": int(v.isna().sum()),
                "pct_missing": round(100 * v.isna().mean(), 2) if len(v) else 0.0,
                "mean": present.mean() if n else np.nan,
                "std": present.std() if n > 1 else np.nan,
                "min": present.min() if n else np.nan,
                "q1": q[0],
                "median": q[1],
                "q3": q[2],
                "max": present.max() if n else np.nan,
            }
        )
        if edges is None:
            continue
        counts = np.histogram(present.to_numpy(), bins=edges)[0]
        hist.append(
            pd.DataFrame(
                {
                    "column": column,
                    GROUP: group,
                    "bin_left": edges[:-1],
                    "bin_right": edges[1:],
                    "count": counts,
                    "share": counts / n if n else 0.0,
                }
            )
        )
    histogram = (
        pd.concat(hist, ignore_index=True)
        if hist
        else pd.DataFrame(columns=HISTOGRAM_FIELDS)
    )
    return histogram, pd.DataFrame(summary, columns=SUMMARY_FIELDS)


def top_labels(series: pd.Series, top_k: int = TOP_K) -> pd.Series:
    """Values as labels; beyond the ``top_k`` most frequent -> ``(other)``
    (missing values keep their own label)."""
    labels = series.map(value_label)
    counts = labels[labels != MISSING_LABEL].value_counts()
    keep = set(counts.index[:top_k]) | {MISSING_LABEL}
    return labels.where(labels.isin(keep), OTHER_LABEL)


def categorical_distribution(
    data: pd.DataFrame, column: str, top_k: int = TOP_K
) -> pd.DataFrame:
    """Value counts of ``column`` per group; ``pct`` over the group's rows.

    Values = the ``top_k`` most frequent over all groups (same for every group),
    most frequent first, then ``(other)`` and ``(missing)``.
    """
    labels = top_labels(data[column], top_k)
    total = labels.value_counts()
    tail = [v for v in (OTHER_LABEL, MISSING_LABEL) if v in total.index]
    order = [v for v in total.index if v not in tail] + tail
    rows = []
    for group in group_order(data[GROUP]):
        in_group = labels[data[GROUP] == group]
        counts = in_group.value_counts()
        for value in order:
            count = int(counts.get(value, 0))
            rows.append(
                {
                    "column": column,
                    GROUP: group,
                    "value": value,
                    "count": count,
                    "pct": round(100 * count / len(in_group), 2)
                    if len(in_group)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=COUNT_FIELDS)
