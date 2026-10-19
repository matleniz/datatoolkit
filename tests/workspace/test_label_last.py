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


def _run_exports(out, tmp_path):
    """Frames from the exported script (run standalone) and notebook (exec'd)."""
    import json
    import subprocess
    import sys

    script = (out / "code" / "pipeline.py").read_text()
    check = tmp_path / "check.py"
    check.write_text(
        script.replace("print(train.shape", "(train.shape")
        + f"\ntrain.to_parquet({str(tmp_path / 'tr.parquet')!r})\n"
        + f"test.to_parquet({str(tmp_path / 'te.parquet')!r})\n"
    )
    done = subprocess.run(
        [sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    ns: dict = {}
    for cell in json.loads((out / "code" / "pipeline.ipynb").read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)  # noqa: S102 - our own generated code
    return [
        (pd.read_parquet(tmp_path / "tr.parquet"), pd.read_parquet(tmp_path / "te.parquet")),
        (ns["train"], ns["test"]),
    ]


def _check_exports(ws, tmp_path, expect_last):
    save_workspace(ws)
    out = tmp_path / "out"
    export_workspace(ws["name"], out, formats=["parquet", "py", "ipynb"])
    expected = {r: pd.read_parquet(out / "processed" / f"{r}.parquet") for r in ("train", "test")}
    if expect_last:
        assert expected["train"].columns[-1] == expect_last
    for train, test in _run_exports(out, tmp_path):
        pd.testing.assert_frame_equal(
            train.reset_index(drop=True), expected["train"], check_dtype=False
        )
        pd.testing.assert_frame_equal(
            test.reset_index(drop=True), expected["test"], check_dtype=False
        )
    return expected


@pytest.mark.parametrize("label_last", [True, False])
def test_exported_code_matches_parquet(tmp_path, label_last):
    ws = _ws(label_last)
    ws["steps"] = [
        FORMULA,
        {"op": "formula", "target": "both", "params": {"name": "fare_3", "expr": "Fare * 3"}},
    ]
    expected = _check_exports(ws, tmp_path, None)
    assert (expected["train"].columns[-1] == "Survived") is label_last


def _y_ws(tmp_path, mode):
    train = pd.read_csv(TRAIN_CSV).drop_duplicates("PassengerId")  # unique join key
    x_path, y_path = tmp_path / "x.csv", tmp_path / "y.csv"
    train.drop(columns="Survived").to_csv(x_path, index=False)
    y = train[["PassengerId", "Survived"]]
    y.to_csv(y_path, index=False)
    label = {"mode": "key", "key": "PassengerId"} if mode == "key" else {"mode": "order"}
    return {
        "name": "yws",
        "label_last": True,
        "label": label,
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(x_path)},
                      "y": {"kind": "csv", "path": str(y_path)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": [FORMULA],
    }


@pytest.mark.parametrize("mode", ["order", "key"])
def test_y_file_label_modes(tmp_path, mode):
    ws = _y_ws(tmp_path, mode)
    if mode == "order":  # order mode: y is the bare label column
        pd.read_csv(tmp_path / "y.csv")[["Survived"]].to_csv(tmp_path / "y.csv", index=False)
    expected = _check_exports(ws, tmp_path, "Survived")
    assert list(expected["train"].columns[-2:]) == ["fare_pp", "Survived"]


def test_zero_steps_and_groups(tmp_path):
    ws = _ws(True)
    ws["steps"] = []  # no step runs, so the replay moves nothing: script == parquet
    _check_exports(ws, tmp_path, None)
    ws = _ws(True)
    ws["name"] = "llg"
    ws["steps"] = [
        {**FORMULA, "id": "s0"},
        {"id": "s1", "op": "formula", "target": "both",
         "params": {"name": "fare_3", "expr": "Fare * 3"}},
    ]
    ws["groups"] = [
        {"id": "g1", "name": "one", "step_ids": ["s0"]},
        {"id": "g2", "name": "two", "step_ids": ["s1"]},
    ]
    expected = _check_exports(ws, tmp_path / "g", "Survived")
    assert list(expected["train"].columns[-3:]) == ["fare_pp", "fare_3", "Survived"]
