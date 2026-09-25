"""Workspace demo on the real (local, not shipped) Parkinson CSVs; skipped if absent."""

import time
from pathlib import Path

import pytest

from dtk_engine import get_workspace, run_key, save_workspace

DATA = Path("/mnt/c/Users/mat24/Downloads")
X_TRAIN, X_TEST = DATA / "X_train_6ZIKlTY.csv", DATA / "X_test_oiZ2ukx.csv"
Y_TRAIN = DATA / "y_train_lXj6X5y.csv"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (X_TRAIN, X_TEST, Y_TRAIN)),
    reason="real data not available",
)


def build_parkinson_workspace(name: str = "parkinson") -> dict:
    """Workspace over the real CSVs: y_train joined to X_train by row order."""
    return save_workspace(
        {
            "name": name,
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": str(X_TRAIN)},
                    "y": {"kind": "csv", "path": str(Y_TRAIN)},
                },
                "test": {"x": {"kind": "csv", "path": str(X_TEST)}},
            },
            "label": {"mode": "order"},
        }
    )


def test_parkinson_workspace(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    build_parkinson_workspace()
    assert get_workspace("parkinson")["steps"] == []
    train = {"kind": "dataset", "workspace": "parkinson", "role": "train"}
    test = {"kind": "dataset", "workspace": "parkinson", "role": "test"}

    t0 = time.perf_counter()
    overview = run_key("dataset_overview", {"source": train})
    t1 = time.perf_counter()
    check = run_key("train_test_check", {"train": train, "test": test})
    t2 = time.perf_counter()

    m = overview["metrics"]
    assert (m["rows"], m["cols"]) == (55_603, 13)  # 12 X columns + target
    columns = {c["column"] for c in _table(overview, "columns")}
    assert "target" in columns
    c = check["metrics"]
    # only_train = target, only_test = time_since_diagnosis
    assert (c["n_only_train"], c["n_only_test"]) == (1, 1)
    with capsys.disabled():
        print(
            f"\n[real data] dataset_overview {m} in {t1 - t0:.2f}s"
            f"\n[real data] train_test_check {c} in {t2 - t1:.2f}s"
        )


def _table(res, title):
    return next(t["records"] for t in res["tables"] if t["title"] == title)
