"""Export formats: csv, ipynb and py next to parquet (datatoolkit-issues#156)."""

from __future__ import annotations

import ast
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
    # a step note is now a comment in the group's code cell (was a markdown cell)
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert " · impute both\n# Median age: few missing values." in code
    assert "## impute both" in markdown  # no groups -> one singleton section per step
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


def test_onehot_missing_option_replays_in_script(tmp_path):
    contract.save_workspace({
        "name": "miss",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": [{"op": "onehot", "target": "both",
                   "params": {"columns": ["Embarked", "Cabin"], "missing": True}}],
    })
    out = tmp_path / "export"
    contract.export_workspace("miss", str(out), formats=["parquet", "py"])
    expected = _parquet(out, "train")
    assert "Cabin_missing" in expected.columns
    script = (out / "code" / "pipeline.py").read_text()
    check = tmp_path / "check.py"
    check.write_text(
        script.replace("print(train.shape", "(train.shape")
        + f"\ntrain.reset_index(drop=True).to_parquet({str(tmp_path / 'tr.parquet')!r})\n"
    )
    done = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    _same(pd.read_parquet(tmp_path / "tr.parquet"), expected)


# --- step groups (datatoolkit-issues#249) ---------------------------------

def _grouped_ws(groups: list[dict] | None) -> dict:
    steps = [{**s, "id": f"s{i}"} for i, s in enumerate(STEPS)]
    ws = {
        "name": "grp",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": steps,
    }
    if groups is not None:
        ws["groups"] = groups
    return ws


GROUPS = [
    {"id": "g1", "name": "Fare features", "intent": "Fare per person", "step_ids": ["s0", "s1"]},
    {"id": "g2", "name": "Encode and filter", "step_ids": ["s2", "s3", "s4"]},
    {"id": "g3", "name": "Rename", "step_ids": ["s5"]},
]


def _export_grouped(tmp_path, groups):
    contract.save_workspace(_grouped_ws(groups))
    out = tmp_path / "export"
    manifest = contract.export_workspace("grp", str(out), formats=["parquet", "ipynb", "py"])
    return manifest, out


def _run_cells(nb: dict) -> dict:
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)  # noqa: S102 - our own generated code
    return namespace


def test_grouped_notebook_has_one_pair_per_group(tmp_path):
    manifest, out = _export_grouped(tmp_path, GROUPS)
    nb = json.loads((out / "code" / "pipeline.ipynb").read_text())
    kinds = [c["cell_type"] for c in nb["cells"]]
    # intro, load, 3 x (markdown, code), final shapes
    assert kinds == ["markdown", "code"] + ["markdown", "code"] * 3 + ["code"]
    src = ["".join(c["source"]) for c in nb["cells"]]
    assert src[2].splitlines() == ["## Fare features", "Fare per person", "Steps: s0, s1"]
    assert src[4].splitlines() == ["## Encode and filter", "Steps: s2, s3, s4"]
    assert "# s0 · impute both" in src[3] and "# s1 · formula both" in src[3]
    assert "# Median age: few missing values." in src[3]
    assert src[3].index("# s0") < src[3].index("# s1")
    assert src[5].count("DtkTransformer(") == 3
    _same(_run_cells(nb)["train"], _parquet(out, "train"))
    assert [g["id"] for g in manifest["groups"]] == ["g1", "g2", "g3"]


def test_grouped_script_runs_like_the_replay(tmp_path):
    _, out = _export_grouped(tmp_path, GROUPS)
    script = (out / "code" / "pipeline.py").read_text()
    assert "# --- Fare features ---\n# Fare per person\n# s0 · impute both" in script
    assert "# --- Rename ---" in script
    check = tmp_path / "check.py"
    check.write_text(
        script.replace("print(train.shape", "(train.shape")
        + f"\ntrain.reset_index(drop=True).to_parquet({str(tmp_path / 'tr.parquet')!r})\n"
    )
    done = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    _same(pd.read_parquet(tmp_path / "tr.parquet"), _parquet(out, "train"))


def test_workspace_without_groups_exports_one_section_per_step(tmp_path):
    manifest, out = _export_grouped(tmp_path, None)
    nb = json.loads((out / "code" / "pipeline.ipynb").read_text())
    n = len(STEPS)
    assert len(nb["cells"]) == 2 + 2 * n + 1
    assert "".join(nb["cells"][2]["source"]).startswith("## impute both\nSteps: s0")
    assert len(manifest["groups"]) == n
    _same(_run_cells(nb)["train"], _parquet(out, "train"))


def test_manifest_maps_steps_to_groups(tmp_path):
    manifest, out = _export_grouped(tmp_path, GROUPS)
    assert [s["group_id"] for s in manifest["steps"]] == ["g1", "g1", "g2", "g2", "g2", "g3"]
    assert manifest["groups"][1] == {
        "id": "g2", "name": "Encode and filter", "intent": "", "steps": [2, 3, 4],
    }
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["groups"] == manifest["groups"]


PAYLOADS = [
    "x\nimport os; os.system('echo pwned')",
    "x\rimport os",
    "x\u2028import os",
    "x\u2029import os",
    "x\r\nimport os",
]


def _statements(script: str) -> list[str]:
    return [type(node).__name__ for node in ast.parse(script).body]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_free_text_cannot_inject_code(tmp_path, payload):
    clean_groups = [dict(g) for g in GROUPS]
    evil_groups = [dict(g) for g in GROUPS]
    evil_groups[0]["intent"] = payload
    ws = _grouped_ws(evil_groups)
    ws["steps"][1]["note"] = payload
    ws["notes"] = {"workspace": payload, "columns": {"Pclass": payload}}
    contract.save_workspace(ws)
    out = tmp_path / "evil"
    contract.export_workspace("grp", str(out), formats=["parquet", "ipynb", "py"])
    script = (out / "code" / "pipeline.py").read_text()

    contract.save_workspace(_grouped_ws(clean_groups))
    clean = tmp_path / "clean"
    contract.export_workspace("grp", str(clean), formats=["py"])
    assert _statements(script) == _statements((clean / "code" / "pipeline.py").read_text())
    assert not any(isinstance(n, ast.Import) and n.names[0].name == "os" for n in ast.walk(ast.parse(script)))

    nb = json.loads((out / "code" / "pipeline.ipynb").read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            assert not any(isinstance(n, ast.Import) and n.names[0].name == "os" for n in ast.walk(tree))
    _same(_run_cells(nb)["train"], _parquet(out, "train"))


@pytest.mark.parametrize("payload", PAYLOADS)
def test_group_name_must_be_one_line(payload):
    groups = [{**GROUPS[0], "name": payload}, *GROUPS[1:]]
    with pytest.raises(Exception, match="one line"):
        contract.save_workspace(_grouped_ws(groups))


def test_derive_step_exports_and_reproduces_the_parquet(tmp_path):
    """derive's param ``op`` (and ``name``) must not collide with DtkTransformer's argument (#253)."""
    contract.save_workspace({
        "name": "dv",
        "datasets": {"train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}}},
        "steps": [
            {"op": "derive", "target": "both",
             "params": {"a": "Fare", "b": "SibSp", "op": "product", "name": "fare_x_sib"}},
        ],
    })
    out = tmp_path / "export"
    contract.export_workspace("dv", str(out), formats=["parquet", "ipynb", "py"])
    expected = _parquet(out, "train")
    assert "fare_x_sib" in expected.columns
    nb = json.loads((out / "code" / "pipeline.ipynb").read_text())
    namespace: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)  # noqa: S102 - our own generated code
    _same(namespace["train"], expected)
    check = tmp_path / "check.py"
    check.write_text(
        (out / "code" / "pipeline.py").read_text()
        + f"\ntrain.reset_index(drop=True).to_parquet({str(tmp_path / 'tr.parquet')!r})\n"
    )
    done = subprocess.run([sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    _same(pd.read_parquet(tmp_path / "tr.parquet"), expected)


def test_test_only_step_without_a_test_set_is_skipped_in_the_script(tmp_path):
    contract.save_workspace({
        "name": "solo",
        "datasets": {"train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}}},
        "steps": [
            {"op": "drop_columns", "target": "test", "params": {"columns": ["Cabin"]}},
            {"op": "rename", "target": "train", "params": {"mapping": {"Pclass": "pclass"}}},
        ],
    })
    out = tmp_path / "export"
    manifest = contract.export_workspace("solo", str(out), formats=["parquet", "py"])
    script = (out / "code" / "pipeline.py").read_text()
    assert "'drop_columns'" not in script  # the replay skips it, so does the script
    assert "'rename'" in script
    assert [s["op"] for s in manifest["steps"]] == ["drop_columns", "rename"]
    assert "pclass" in pd.read_parquet(out / "processed" / "train.parquet").columns


# -- optional sklearn Pipeline section (datatoolkit-issues#295) --

BOTH_STEPS = [s for s in STEPS if s["target"] == "both"]


def _export_demo(tmp_path, name: str, steps, **kwargs):
    contract.save_workspace({
        "name": name,
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(TRAIN_CSV)}},
            "test": {"x": {"kind": "csv", "path": str(TEST_CSV)}},
        },
        "steps": steps,
    })
    out = tmp_path / name
    contract.export_workspace(name, str(out), **kwargs)
    return out


def test_pipeline_is_off_by_default(tmp_path):
    out = _export_demo(tmp_path, "w", STEPS, formats=["py"])
    script = (out / "code" / "pipeline.py").read_text()
    assert "PIPELINE" not in script and "sklearn" not in script
    again = _export_demo(tmp_path, "w2", STEPS, formats=["py"], pipeline=False)
    assert _statements(script) == _statements((again / "code" / "pipeline.py").read_text())


def test_pipeline_needs_the_py_format(tmp_path):
    with pytest.raises(KeyParamsError, match="pipeline needs the 'py' format"):
        contract.export_workspace("demo", str(tmp_path / "x"), formats=["ipynb"], pipeline=True)


def test_pipeline_section_runs_standalone_and_matches_the_replay(tmp_path):
    out = _export_demo(tmp_path, "pp", BOTH_STEPS, formats=["parquet", "py"], pipeline=True)
    script = (out / "code" / "pipeline.py").read_text()
    head, _, section = script.partition("# --- sklearn pipeline (optional) ---")
    assert section and "workspace_pipeline" not in script  # no stored workspace needed
    assert _statements(head) == _statements(
        (_export_demo(tmp_path, "pp2", BOTH_STEPS, formats=["py"]) / "code" / "pipeline.py").read_text()
    )
    check = tmp_path / "check.py"
    check.write_text(
        script
        + "\nraw_train, raw_test = workspace_frames(SOURCES)"
        + "\nPIPELINE.fit(raw_train)"
        + f"\nPIPELINE.transform(raw_train).to_parquet({str(tmp_path / 'tr.parquet')!r})"
        + f"\nPIPELINE.transform(raw_test).to_parquet({str(tmp_path / 'te.parquet')!r})\n"
    )
    done = subprocess.run(
        [sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    _same(pd.read_parquet(tmp_path / "tr.parquet"), _parquet(out, "train"))
    _same(pd.read_parquet(tmp_path / "te.parquet"), _parquet(out, "test"))


def _run_script(tmp_path, script: str, extra: str = ""):
    check = tmp_path / "check.py"
    check.write_text(script + "\n" + extra)
    done = subprocess.run(
        [sys.executable, str(check)], capture_output=True, text=True, timeout=120, check=False,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr


def test_pipeline_lists_skipped_steps_as_comments(tmp_path):
    out = _export_demo(tmp_path, "mix", STEPS, formats=["py"], pipeline=True)
    script = (out / "code" / "pipeline.py").read_text()
    section = script.split("PIPELINE = ")[1]
    assert "'0_impute'" in section and "'5_rename'" in section
    assert "DtkTransformer('filter_rows'" not in section
    assert "DtkTransformer('drop_columns'" not in section
    assert "# skipped: s5 drop_columns (test): test-only" in section
    assert "# skipped: s4 filter_rows (train): train-only" in section
    _run_script(tmp_path, script)


def test_pipeline_row_changing_both_step_is_skipped_with_a_reason(tmp_path):
    steps = [STEPS[0], {"op": "filter_rows", "target": "both", "params": STEPS[3]["params"]}]
    out = _export_demo(tmp_path, "rows", steps, formats=["py"], pipeline=True)
    script = (out / "code" / "pipeline.py").read_text()
    section = script.split("PIPELINE = ")[1]
    assert "# skipped: s2 filter_rows (both): changes the rows" in section
    assert "apply it to your data yourself before fitting" in section
    assert "DtkTransformer('filter_rows'" not in section
    _run_script(tmp_path, script)


def test_pipeline_label_reading_step_is_flagged(tmp_path):
    steps = [{"op": "select_k_best", "target": "both",
              "params": {"target": "Survived", "k": 2, "columns": ["Fare", "Pclass"]}}]
    out = _export_demo(tmp_path, "sel", steps, formats=["py"], pipeline=True)
    script = (out / "code" / "pipeline.py").read_text()
    assert "reads the label column 'Survived'" in script
    assert "must be in the frame passed to" in script
    assert "DtkTransformer('select_k_best'" in script
    assert "PIPELINE.fit(X_train)" not in script


def test_pipeline_header_states_what_is_outside(tmp_path):
    out = _export_demo(tmp_path, "hdr", BOTH_STEPS, formats=["py"], pipeline=True)
    script = (out / "code" / "pipeline.py").read_text()
    assert "merges and label joins are not in it" in script
    assert "label_last" not in script


def test_pipeline_without_both_steps_is_a_passthrough(tmp_path):
    out = _export_demo(tmp_path, "none", [STEPS[3]], formats=["py"], pipeline=True)
    assert "('passthrough', 'passthrough')" in (out / "code" / "pipeline.py").read_text()


def test_pipeline_leaves_other_formats_alone(tmp_path):
    plain = _export_demo(tmp_path, "a", BOTH_STEPS, formats=["parquet", "csv", "ipynb"])
    withp = _export_demo(tmp_path, "b", BOTH_STEPS, formats=["parquet", "csv", "ipynb", "py"], pipeline=True)
    for rel in ("processed/train.parquet", "processed/train.csv", "processed/test.csv"):
        assert (plain / rel).read_bytes() == (withp / rel).read_bytes()
    assert "PIPELINE" not in (withp / "code" / "pipeline.ipynb").read_text()
