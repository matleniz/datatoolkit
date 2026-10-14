"""Workspace flow on local CSVs: synthetic (always) and real (opt-in via DTK_REAL_DATA_DIR)."""

import os
from pathlib import Path

import pytest

from dtk_engine import get_workspace, run_key, save_workspace

REAL_DIR = os.environ.get("DTK_REAL_DATA_DIR")
FILES = ("X_train_6ZIKlTY.csv", "X_test_oiZ2ukx.csv", "y_train_lXj6X5y.csv")


def build_parkinson_workspace(data: Path, name: str = "parkinson") -> dict:
    """Workspace over the CSVs in `data`: y_train joined to X_train by row order."""
    x_train, x_test, y_train = (data / f for f in FILES)
    return save_workspace(
        {
            "name": name,
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": str(x_train)},
                    "y": {"kind": "csv", "path": str(y_train)},
                },
                "test": {"x": {"kind": "csv", "path": str(x_test)}},
            },
            "label": {"mode": "order"},
        }
    )


def _run_flow(data: Path):
    build_parkinson_workspace(data)
    assert get_workspace("parkinson")["steps"] == []
    train = {"kind": "dataset", "workspace": "parkinson", "role": "train"}
    test = {"kind": "dataset", "workspace": "parkinson", "role": "test"}
    overview = run_key("dataset_overview", {"source": train})
    check = run_key("train_test_check", {"train": train, "test": test})
    return overview, check


def _synthetic_dir(path: Path, rows: int = 40) -> Path:
    head = "a,b,c"
    path.joinpath(FILES[0]).write_text(
        head + "\n" + "\n".join(f"{i},{i * 2},{i % 5}" for i in range(rows)) + "\n"
    )
    path.joinpath(FILES[1]).write_text(
        head + ",time_since_diagnosis\n"
        + "\n".join(f"{i},{i * 3},{i % 5},{i}" for i in range(10))
        + "\n"
    )
    path.joinpath(FILES[2]).write_text(
        "target\n" + "\n".join(str(i % 2) for i in range(rows)) + "\n"
    )
    return path


def test_synthetic_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    overview, check = _run_flow(_synthetic_dir(tmp_path))
    m = overview["metrics"]
    assert (m["rows"], m["cols"]) == (40, 4)  # 3 X columns + target
    assert "target" in {c["column"] for c in _table(overview, "columns")}
    c = check["metrics"]
    assert (c["n_only_train"], c["n_only_test"]) == (1, 1)


@pytest.mark.skipif(
    not (REAL_DIR and all((Path(REAL_DIR) / f).exists() for f in FILES)),
    reason="DTK_REAL_DATA_DIR unset or real data not available",
)
def test_parkinson_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    overview, check = _run_flow(Path(REAL_DIR))
    m = overview["metrics"]
    assert (m["rows"], m["cols"]) == (55_603, 13)  # 12 X columns + target
    assert "target" in {c["column"] for c in _table(overview, "columns")}
    c = check["metrics"]
    # only_train = target, only_test = time_since_diagnosis
    assert (c["n_only_train"], c["n_only_test"]) == (1, 1)


def _table(res, title):
    return next(t["records"] for t in res["tables"] if t["title"] == title)
