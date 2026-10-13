"""Workspace ``label_last``: the label column stays last through replay / export
(datatoolkit-issues#196)."""

import pandas as pd
import pytest

from dtk_engine import export_workspace, save_workspace
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.workspace import dataset
from dtk_engine.workspace.dataset import workspace_frame

FORMULA = {"op": "formula", "target": "both", "params": {"name": "fare_pp", "expr": "Fare * 2"}}


def _ws(label_last=None) -> dict:
    ws = {
        "name": "ll",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": TRAIN_CSV}, "target_column": "Survived"},
            "test": {"x": {"kind": "csv", "path": TEST_CSV}},
        },
        "steps": [FORMULA],
    }
    if label_last is not None:
        ws["label_last"] = label_last
    return ws


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    dataset._FRAMES.clear()


def _train(ws_dict):
    from dtk_engine.workspace.models import Workspace

    return workspace_frame(Workspace.model_validate(ws_dict), "train")


def test_default_false_unchanged():
    cols = list(_train(_ws()).columns)
    assert cols[-1] == "fare_pp"
    assert cols != list(_train(_ws(True)).columns)
    assert cols[-1] != "Survived"


def test_true_keeps_label_last():
    cols = list(_train(_ws(True)).columns)
    assert cols[-1] == "Survived"
    assert cols[-2] == "fare_pp"


def test_cache_key_separates_label_last():
    assert list(_train(_ws(True)).columns) != list(_train(_ws(False)).columns)
    assert list(_train(_ws(True)).columns)[-1] == "Survived"


def test_export_honours_label_last(tmp_path):
    save_workspace(_ws(True))
    out = tmp_path / "out"
    export_workspace("ll", out)
    train = pd.read_parquet(out / "processed" / "train.parquet")
    assert train.columns[-1] == "Survived"
    assert "fare_pp" in train.columns
