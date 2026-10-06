"""Export formats: csv, ipynb and py next to parquet (datatoolkit-issues#156)."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from dtk_engine import contract
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError

STEPS = [
    {"op": "impute", "target": "both", "params": {"columns": ["Fare"], "strategy": "median"},
     "note": "Median age: few missing values."},
    {"op": "formula", "target": "both", "params": {"name": "fare_pp", "expr": "Fare / (SibSp + 1)"}},
    {"op": "onehot", "target": "both", "params": {"columns": ["Embarked"]}},
    {"op": "filter_rows", "target": "train",
     "params": {"conditions": [{"column": "Fare", "op": "gt", "value": 0}], "combine": "and"}},
    {"op": "drop_columns", "target": "test", "params": {"columns": ["Cabin"]}},
    {"op": "rename", "target": "both", "params": {"mapping": {"Pclass": "pclass"}}},
]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))


@pytest.fixture
def exported(tmp_path) -> tuple[dict, object]:
    contract.save_workspace({
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": STEPS,
        "notes": {"workspace": "Titanic-like demo study.", "columns": {"Pclass": "ticket class"}},
    })
    out = tmp_path / "export"
    manifest = contract.export_workspace("demo", str(out), formats=["parquet", "csv", "ipynb", "py"])
    return manifest, out


def _parquet(out, role: str) -> pd.DataFrame:
    return pd.read_parquet(out / "processed" / f"{role}.parquet")


def _same(frame: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True).rename(columns=str), expected, check_dtype=False
    )


def test_outputs_are_listed_with_hashes(exported):
    manifest, out = exported
    assert manifest["formats"] == ["parquet", "csv", "ipynb", "py"]
    assert set(manifest["outputs"]) == {"train", "test", "train_csv", "test_csv", "notebook", "script"}
    for entry in manifest["outputs"].values():
        assert (out / entry["path"]).is_file() and len(entry["sha256"]) == 64
    assert manifest["outputs"]["notebook"]["path"] == "code/pipeline.ipynb"


def test_csv_round_trip(exported):
    _, out = exported
    for role in ("train", "test"):
        csv = pd.read_csv(out / "processed" / f"{role}.csv")
        expected = _parquet(out, role)
        pd.testing.assert_frame_equal(csv, expected, check_dtype=False)


def test_notebook_runs_and_reproduces_the_parquet(exported):
    _, out = exported
    nb = json.loads((out / "code" / "pipeline.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5
    assert all(c["id"] for c in nb["cells"])
    markdown = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "Titanic-like demo study." in markdown and "ticket class" in markdown
    assert "Median age: few missing values." in markdown
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)  # noqa: S102 - our own generated code
    _same(namespace["train"], _parquet(out, "train"))
    _same(namespace["test"], _parquet(out, "test"))


def test_script_runs_standalone(exported, tmp_path):
    _, out = exported
    script = (out / "code" / "pipeline.py").read_text()
    assert "# Median age: few missing values." in script
    check = tmp_path / "check.py"
    check.write_text(
        script.replace("print(train.shape", "(train.shape")
        + f"\ntrain.reset_index(drop=True).to_parquet({str(tmp_path / 'tr.parquet')!r})\n"
    )
    done = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    _same(pd.read_parquet(tmp_path / "tr.parquet"), _parquet(out, "train"))


def test_overwrite_removes_code_and_csv(exported):
    _, out = exported
    contract.export_workspace("demo", str(out), overwrite=True)
    assert not (out / "code" / "pipeline.ipynb").exists()
    assert not (out / "processed" / "train.csv").exists()
    assert (out / "processed" / "train.parquet").exists()


def test_unknown_format_is_refused(exported, tmp_path):
    with pytest.raises(KeyParamsError, match="formats"):
        contract.export_workspace("demo", str(tmp_path / "x"), formats=["xlsx"])
