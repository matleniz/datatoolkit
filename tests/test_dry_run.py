"""preview_steps / evaluate: stateless multi-step dry run (datatoolkit-issues#155)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dtk_engine import contract
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, UnknownTransformError

DOUBLE = {"op": "formula", "target": "both", "params": {"name": "age2", "expr": "Age * 2"}}
FILL = {"op": "impute", "target": "both", "params": {"columns": ["age2"], "strategy": "median"}}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))


@pytest.fixture
def ws() -> dict:
    return contract.save_workspace({
        "name": "demo",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": [{"op": "log1p", "target": "both", "params": {"columns": ["Fare"]}}],
    })


def test_preview_steps_chains_and_saves_nothing(ws):
    out = contract.preview_steps(ws, [DOUBLE, FILL], "train")
    assert out["added_columns"] == ["age2"]
    assert [s["op"] for s in out["steps"]] == ["formula", "impute"]
    assert "age2" in out["steps"][1]["state"]["fill"]  # the second step saw the first's column
    assert out["shape"][0] == len(pd.read_csv(TRAIN_CSV))
    assert len(contract.get_workspace("demo")["steps"]) == 1
    test = contract.preview_steps(ws, [DOUBLE, FILL], "test")
    assert test["steps"][1]["fitted_on"] == "train"


def test_preview_steps_rejects_bad_lists(ws):
    with pytest.raises(KeyParamsError):
        contract.preview_steps(ws, [], "train")
    with pytest.raises(UnknownTransformError):
        contract.preview_steps(ws, [{"op": "nope", "target": "both"}], "train")


def test_evaluate_on_versions_drafts_and_rows(ws):
    raw = pd.read_csv(TRAIN_CSV)
    at0 = contract.evaluate(ws, "train", ["Age", "isna(Age)"], version=0)
    age, missing = at0["results"]
    assert age["mean"] == pytest.approx(raw["Age"].mean())
    assert age["missing"] == int(raw["Age"].isna().sum())
    assert missing["sum"] == age["missing"]
    fare = contract.evaluate(ws, "train", ["Fare"])["results"][0]  # latest: log1p applied
    assert fare["max"] == pytest.approx(float(np.log1p(raw["Fare"]).max()))
    drafted = contract.evaluate(ws, "train", ["age2"], steps=[DOUBLE, FILL])["results"][0]
    assert drafted["missing"] == 0 and drafted["median"] == pytest.approx(2 * raw["Age"].median())
    some = contract.evaluate(ws, "train", ["Age"], where="Pclass == 1", version=0)
    assert some["selected"] == int((raw["Pclass"] == 1).sum())
    assert some["results"][0]["mean"] == pytest.approx(raw.loc[raw["Pclass"] == 1, "Age"].mean())


def test_evaluate_errors(ws):
    with pytest.raises(KeyParamsError, match="expression"):
        contract.evaluate(ws, "train", ["nope_column + 1"])
    with pytest.raises(KeyParamsError):
        contract.evaluate(ws, "train", [])
    with pytest.raises(KeyParamsError, match="exclude"):
        contract.evaluate(ws, "train", ["Age"], steps=[DOUBLE], version=0)
