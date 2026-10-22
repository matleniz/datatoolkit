"""expected_schema as a workspace step: replay, preview, save, groups, export."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from dtk_engine import contract
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.workspace import Step
from dtk_engine.workspace.replay import replay, replay_fitted

SCHEMA = {
    "extra_columns": "ignore",
    "min_rows": 2,
    "columns": [
        {"name": "age", "dtype": "numeric", "nullable": False, "min": 0, "max": 120},
        {"name": "sex", "allowed": ["F", "M"]},
        {"name": "id", "unique": True},
    ],
}
TRAIN = pd.DataFrame(
    {"age": [30, 41, 150, None], "sex": ["F", "M", "X", "F"], "id": ["a", "b", "b", "c"], "z": 1}
)
TEST = pd.DataFrame({"age": [20, 30], "sex": ["F", "M"], "id": ["x", "y"], "z": 1})


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))


@pytest.fixture
def ws(tmp_path) -> dict:
    TRAIN.to_csv(tmp_path / "train.csv", index=False)
    TEST.to_csv(tmp_path / "test.csv", index=False)
    return {
        "name": "contract",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(tmp_path / "train.csv")}},
            "test": {"x": {"kind": "csv", "path": str(tmp_path / "test.csv")}},
        },
        "steps": [],
    }


def _step(target="both", **extra):
    return {"id": "s0", "op": "expected_schema", "target": target, "params": {**SCHEMA, **extra}}


def _checks(fitted_entry):
    return fitted_entry["checks"]


def test_both_checks_train_and_test_and_data_is_unchanged():
    steps = [Step(op="expected_schema", target="both", params=SCHEMA)]
    train, test, fitted = replay_fitted(steps, TRAIN.copy(), TEST.copy())
    pd.testing.assert_frame_equal(train, TRAIN)
    pd.testing.assert_frame_equal(test, TEST)
    by_split = {s: [c for c in _checks(fitted[0]) if c["split"] == s] for s in ("train", "test")}
    assert {c["check"] for c in by_split["train"]} == {
        "missing", "range", "allowed", "duplicates"
    }
    assert by_split["test"] == []
    assert fitted[0]["state"]["violations"]["error"] == 4  # counts of the fit frame, mirrored


def test_train_only_and_test_only_targets():
    _, _, fitted = replay_fitted(
        [Step(op="expected_schema", target="train", params=SCHEMA)], TRAIN, TEST
    )
    assert {c["split"] for c in _checks(fitted[0])} == {"train"}
    bad_test = pd.DataFrame({"age": [500], "sex": ["F"], "id": ["q"]})
    _, _, fitted = replay_fitted(
        [Step(op="expected_schema", target="test", params={**SCHEMA, "min_rows": 0})], TRAIN, bad_test
    )
    assert {c["split"] for c in _checks(fitted[0])} == {"test"}
    assert [c["check"] for c in _checks(fitted[0])] == ["range"]


def test_error_policy_fails_the_step_like_any_failing_step():
    steps = [Step(op="expected_schema", target="both", params={**SCHEMA, "on_violation": "error"})]
    for fn in (
        lambda: replay(steps, "train", TRAIN),
        lambda: replay(steps, "test", TEST.assign(age=-5), train=TRAIN),
        lambda: replay_fitted(steps, TRAIN, TEST),
    ):
        with pytest.raises(SourceError, match=r"step 0 .*expected schema violated.*age"):
            fn()
    # a clean frame passes; warnings alone never fail
    ok = pd.DataFrame({"age": [1, 2, 3, 4], "sex": ["F"] * 4, "id": list("abcd")})
    replay(steps, "train", ok)


def test_error_on_the_test_split_names_the_split():
    steps = [Step(op="expected_schema", target="both", params={
        "columns": [{"name": "age", "max": 100}], "on_violation": "error"})]
    with pytest.raises(SourceError, match="on the test set"):
        replay(steps, "test", pd.DataFrame({"age": [101]}), train=pd.DataFrame({"age": [1]}))


def test_other_steps_still_see_the_frame_after_a_report():
    steps = [
        Step(op="expected_schema", target="both", params=SCHEMA),
        Step(op="drop_columns", target="both", params={"columns": ["z"]}),
    ]
    train, _, fitted = replay_fitted(steps, TRAIN, TEST)
    assert "z" not in train.columns and "checks" not in fitted[1]


def test_preview_step_carries_checks(ws):
    out = contract.preview_step(ws, _step(), "train")
    assert out["checks"] and out["state"]["violations"]["error"] == 4
    assert {"severity", "check", "column", "message", "count", "examples", "split"} <= set(out["checks"][0])
    assert out["changed_total"] == 0 and out["added_columns"] == out["removed_columns"] == []
    json.dumps(out)
    plain = contract.preview_step(ws, {"op": "drop_columns", "target": "both",
                                       "params": {"columns": ["z"]}}, "train")
    assert plain["checks"] == []  # additive field, empty for other ops


def test_preview_steps_has_checks_per_step(ws):
    drop = {"id": "s1", "op": "drop_columns", "target": "both", "params": {"columns": ["z"]}}
    out = contract.preview_steps(ws, [_step(), drop], "train")
    assert out["steps"][0]["checks"] and out["steps"][1]["checks"] == []
    assert out["checks"] == []  # the last step's


def test_save_validates_params_and_round_trips(ws):
    ws["steps"] = [_step()]
    contract.save_workspace(ws)
    again = contract.get_workspace("contract")
    assert again["steps"][0]["params"] == SCHEMA
    again2 = contract.save_workspace(again)
    assert json.dumps(contract.get_workspace("contract")["steps"][0]["params"]) == json.dumps(SCHEMA)
    assert again2
    ws["steps"] = [_step(columns=[{"name": "a", "min": 3, "max": 1}])]
    with pytest.raises(KeyParamsError, match="transform 'expected_schema'"):
        contract.save_workspace(ws)


def test_group_and_export(ws, tmp_path):
    ws["steps"] = [_step(), {"id": "s1", "op": "drop_columns", "target": "both",
                             "params": {"columns": ["z"]}}]
    ws["groups"] = [{"id": "g1", "name": "Contract", "step_ids": ["s0"]},
                    {"id": "g2", "name": "Clean", "step_ids": ["s1"]}]
    contract.save_workspace(ws)
    out = tmp_path / "export"
    manifest = contract.export_workspace("contract", str(out), formats=["parquet", "ipynb", "py"])
    assert manifest["steps"][0]["op"] == "expected_schema"
    got = pd.read_parquet(out / "processed" / "train.parquet")
    assert list(got.columns) == ["age", "sex", "id"] and len(got) == 4
    script = (out / "code" / "pipeline.py").read_text()
    assert "DtkTransformer('expected_schema'" in script
    check = tmp_path / "check.py"
    check.write_text(script)
    done = subprocess.run([sys.executable, str(check)], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert "expected_schema" in done.stderr  # the report surfaces as a warning


def test_exported_script_raises_under_error_policy(ws, tmp_path):
    ws["steps"] = [_step(on_violation="error")]
    contract.save_workspace(ws)
    # the engine export replays the steps, so a failing contract fails it like any step
    with pytest.raises(SourceError, match="expected schema violated"):
        contract.export_workspace("contract", str(tmp_path / "e1"), formats=["parquet"])
    ws["steps"] = [_step(on_violation="report")]
    contract.save_workspace(ws)
    out = tmp_path / "e2"
    contract.export_workspace("contract", str(out), formats=["parquet", "py"])
    script = (out / "code" / "pipeline.py").read_text()
    assert "'on_violation': 'report'" in script
    strict = tmp_path / "strict.py"
    strict.write_text(script.replace("'on_violation': 'report'", "'on_violation': 'error'"))
    done = subprocess.run([sys.executable, str(strict)], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode != 0 and "expected schema violated" in done.stderr
