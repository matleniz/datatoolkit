import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dtk_engine import api, export_workspace, save_workspace
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, SourceError
from dtk_engine.sources import load
from dtk_engine.sources.spec import DatasetSource
from dtk_engine.workspace import WorkspaceNotFoundError
from dtk_engine.workspace.export import export_workspace as export
from dtk_engine.workspace.export import load_state

STEPS = [
    {"op": "drop_duplicates", "target": "train", "params": {"keep": "none"}},
    {
        "op": "replace_sentinels",
        "target": "test",
        "params": {"sentinels": {"Age": ["unknown"]}},
    },
    {"op": "cast", "target": "test", "params": {"dtypes": {"Age": "float64"}}},
    {
        "op": "drop_columns",
        "target": "both",
        "params": {"columns": ["Name", "Ticket", "Cabin"]},
    },
    {"op": "impute_knn", "target": "both", "params": {"columns": ["Age", "Fare"]}},
    {"op": "onehot", "target": "both", "params": {"columns": ["Sex", "Embarked"]}},
    {"op": "scale", "target": "both", "params": {"columns": ["Fare"]}},
]


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    save_workspace(
        {
            "name": "demo",
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": TRAIN_CSV},
                    "target_column": "Survived",
                },
                "test": {"x": {"kind": "csv", "path": TEST_CSV}},
            },
            "steps": STEPS,
        }
    )
    return "demo"


def test_round_trip_equals_replay(workspace, tmp_path):
    raw = {p: _sha(p) for p in (TRAIN_CSV, TEST_CSV)}
    out = tmp_path / "export"
    returned = api.export_workspace(workspace, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == returned

    for role in ("train", "test"):
        replayed = load(DatasetSource(workspace=workspace, role=role))
        written = pd.read_parquet(out / "processed" / f"{role}.parquet")
        pd.testing.assert_frame_equal(written, replayed.reset_index(drop=True))
        assert manifest["outputs"][role]["rows"] == len(replayed)
        assert manifest["outputs"][role]["sha256"] == _sha(
            out / manifest["outputs"][role]["path"]
        )
    assert {p: _sha(p) for p in raw} == raw  # raw inputs untouched

    src = {s["role"]: s for s in manifest["sources"]}
    assert src["train"]["sha256"] == raw[TRAIN_CSV]
    assert src["train"]["size"] == Path(TRAIN_CSV).stat().st_size
    assert src["train"]["mtime"] and src["test"]["path"] == str(
        Path(TEST_CSV).resolve()
    )
    assert set(manifest["versions"]) >= {"dtk_engine", "pandas", "sklearn"}
    assert manifest["exported_at"]

    steps = manifest["steps"]
    assert [s["op"] for s in steps] == [s["op"] for s in STEPS]
    assert steps[2]["fitted_on"] == "test" and steps[4]["fitted_on"] == "train"
    assert set(steps[5]["state"]) == {"Sex", "Embarked"}  # small state: inline


def test_big_state_goes_to_side_file(workspace, tmp_path):
    out = tmp_path / "export"
    manifest = export_workspace(workspace, str(out))
    knn = manifest["steps"][4]
    assert "state" in knn  # 40 rows: small enough to inline

    # Tiny inline limit: the impute_knn state (train matrix) goes to a side file.
    manifest = export(workspace, out, overwrite=True, inline_state_bytes=256)
    knn = manifest["steps"][4]
    assert "state" not in knn and knn["state_file"] == "states/step_4_impute_knn.json"
    state = load_state(out, knn)
    assert state["columns"] == ["Age", "Fare"] and len(state["train"]) == 39
    assert knn["state_sha256"] == _sha(out / knn["state_file"])

    # Overwrite removes the files the previous export listed.
    export(workspace, out, overwrite=True)
    assert not (out / knn["state_file"]).exists()


def test_existing_export_needs_overwrite(workspace, tmp_path):
    export_workspace(workspace, str(tmp_path))
    with pytest.raises(KeyParamsError, match="overwrite"):
        export_workspace(workspace, str(tmp_path))
    assert (
        export_workspace(workspace, str(tmp_path), overwrite=True)["workspace"]
        == "demo"
    )


def _tamper(out, mutate):
    manifest = json.loads((out / "manifest.json").read_text())
    mutate(manifest)
    (out / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("bad", ["../victim.csv", "processed/../../victim.csv", "abs"])
def test_overwrite_refuses_unsafe_manifest_paths(workspace, tmp_path, bad):
    out = tmp_path / "export"
    export_workspace(workspace, str(out))
    victim = tmp_path / "victim.csv"
    victim.write_text("raw")
    bad = str(victim) if bad == "abs" else bad
    _tamper(out, lambda m: m["outputs"]["train"].update(path=bad))
    with pytest.raises(KeyParamsError, match="outside|non-relative"):
        export(workspace, out, overwrite=True)
    assert victim.exists() and (out / "processed" / "train.parquet").exists()


def test_overwrite_refuses_foreign_manifest(workspace, tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    keep = out / "processed" / "train.parquet"
    keep.parent.mkdir()
    keep.write_text("mine")
    (out / "manifest.json").write_text(
        json.dumps({"outputs": {"train": {"path": "processed/train.parquet"}}})
    )
    with pytest.raises(KeyParamsError, match="not written by"):
        export(workspace, out, overwrite=True)
    assert keep.exists()


def test_refuses_to_overwrite_a_raw_input(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    raw = tmp_path / "processed" / "train.parquet"
    raw.parent.mkdir()
    pd.DataFrame({"a": [1, 2]}).to_parquet(raw)
    save_workspace(
        {
            "name": "w",
            "datasets": {"train": {"x": {"kind": "parquet", "path": str(raw)}}},
        }
    )
    with pytest.raises(KeyParamsError, match="raw input"):
        export_workspace("w", str(tmp_path))
    manifest = export_workspace("w", str(tmp_path / "elsewhere"))
    assert "test" not in manifest["outputs"] and manifest["steps"] == []


def test_unknown_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    with pytest.raises(WorkspaceNotFoundError):
        export_workspace("nope", str(tmp_path))


# --- overwrite is staged: a failing export leaves the previous one intact ---


def _tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def copied_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    csv = tmp_path / "train.csv"
    csv.write_bytes(Path(TRAIN_CSV).read_bytes())
    save_workspace(
        {
            "name": "stage",
            "datasets": {"train": {"x": {"kind": "csv", "path": str(csv)}, "target_column": "Survived"}},
            "steps": [{"op": "scale", "target": "train", "params": {"columns": ["Fare"]}}],
        }
    )
    return csv


def _no_staging(parent: Path) -> bool:
    return not list(parent.rglob(".export-*"))


def test_overwrite_failing_replay_keeps_previous(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out, formats=["parquet", "csv", "py"])
    before = _tree(out)
    df = pd.read_csv(copied_source).drop(columns=["Fare"])
    df.to_csv(copied_source, index=False)
    with pytest.raises(Exception) as err:
        export("stage", out, overwrite=True, formats=["parquet", "csv", "py"])
    assert not isinstance(err.value, AssertionError)
    assert _tree(out) == before
    assert _no_staging(out)


def test_overwrite_failing_write_keeps_previous(copied_source, tmp_path, monkeypatch):
    out = tmp_path / "out"
    export("stage", out, formats=["parquet", "csv", "py"])
    before = _tree(out)

    def boom(self, *a, **k):
        raise ValueError("duplicate column names")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(Exception, match="cannot write train.parquet"):
        export("stage", out, overwrite=True, formats=["csv", "parquet"])
    assert _tree(out) == before
    assert _no_staging(out)


def test_overwrite_success_replaces_owned_keeps_foreign(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out, formats=["parquet", "csv", "py"])
    (out / "notes.txt").write_text("mine")
    (out / "processed" / "mine.txt").write_text("mine too")
    export("stage", out, overwrite=True, formats=["parquet"])
    assert (out / "notes.txt").read_text() == "mine"
    assert (out / "processed" / "mine.txt").read_text() == "mine too"
    assert not (out / "code" / "pipeline.py").exists()
    assert not (out / "processed" / "train.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["outputs"]["train"]["path"] == "processed/train.parquet"
    assert (out / "processed" / "train.parquet").exists()
    assert _no_staging(out)


def test_no_overwrite_unchanged_and_clean(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out)
    before = _tree(out)
    with pytest.raises(KeyParamsError, match="exists .*pass overwrite=True"):
        export("stage", out)
    assert _tree(out) == before
    assert _no_staging(out)


@pytest.mark.parametrize("fail_at", range(1, 9))
def test_overwrite_swap_failure_restores_previous(copied_source, tmp_path, monkeypatch, fail_at):
    from dtk_engine.workspace import export as mod

    out = tmp_path / "out"
    export("stage", out, formats=["parquet", "csv", "py"])
    before = _tree(out)
    real, calls = mod.os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == fail_at:  # 4 moves aside, then 4 installs
            raise OSError("disk gone")
        return real(src, dst)

    monkeypatch.setattr(mod.os, "replace", flaky)
    with pytest.raises(OSError, match="disk gone"):
        export("stage", out, overwrite=True, formats=["parquet", "csv", "py"])
    monkeypatch.undo()
    assert _tree(out) == before
    assert _no_staging(out)


def test_first_export_failure_removes_created_dir(copied_source, tmp_path, monkeypatch):
    from dtk_engine.workspace import export as mod

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(mod.os, "replace", boom)
    out = tmp_path / "a" / "b" / "out"
    with pytest.raises(OSError, match="disk gone"):
        export("stage", out)
    assert not (tmp_path / "a").exists()


def test_stale_staging_is_cleaned_only_when_ours_and_dead(copied_source, tmp_path):
    import os
    import subprocess
    import sys

    out = tmp_path / "out"
    export("stage", out)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    mine = {"dead": dead.pid, "live": os.getpid()}
    for name, pid in mine.items():
        d = out / f".export-staging-{name}"
        d.mkdir()
        (d / ".dtk-staging").write_text(str(pid))
    (out / ".export-staging-foreign").mkdir()  # no marker: not ours
    export("stage", out, overwrite=True)
    left = sorted(p.name for p in out.glob(".export-*"))
    assert left == [".export-staging-foreign", ".export-staging-live"]


def test_double_failure_keeps_previous_export_aside(copied_source, tmp_path, monkeypatch):
    from dtk_engine.workspace import export as mod

    out = tmp_path / "out"
    export("stage", out, formats=["parquet", "csv", "py"])
    before = _tree(out)
    real, calls = mod.os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        n = len(calls)
        if n == 6 or n == 8:  # an install move, then the restore of the 1st entry
            raise OSError("disk gone")
        return real(src, dst)

    monkeypatch.setattr(mod.os, "replace", flaky)
    with pytest.raises(OSError, match="previous export kept in") as err:
        export("stage", out, overwrite=True, formats=["parquet", "csv", "py"])
    monkeypatch.undo()
    assert isinstance(err.value.__cause__, OSError)
    (aside,) = out.glob(".export-old-*")
    assert str(aside) in str(err.value)
    kept = {str(p.relative_to(aside)): p.read_bytes() for p in aside.rglob("*") if p.is_file()}
    assert kept and all(before[k] == v for k, v in kept.items())


def test_stale_aside_is_restored_when_no_manifest(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out)
    before = _tree(out)
    aside = out / ".export-old-dead"
    aside.mkdir()
    for f in ("manifest.json", "processed/train.parquet"):
        (aside / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).replace(aside / f)
    with pytest.raises(KeyParamsError, match="pass overwrite=True"):
        export("stage", out)  # restored, hence seen as an existing export
    assert _tree(out) == before


def test_stale_aside_is_left_when_a_manifest_exists(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out)
    (out / ".export-old-dead").mkdir()
    (out / ".export-old-dead" / "keep").write_text("k")
    export("stage", out, overwrite=True)
    assert (out / ".export-old-dead" / "keep").read_text() == "k"


def test_parent_readonly_is_fine(copied_source, tmp_path):
    parent = tmp_path / "ro"
    out = parent / "out"
    out.mkdir(parents=True)
    parent.chmod(0o555)
    try:
        export("stage", out)
        export("stage", out, overwrite=True)
        assert (out / "manifest.json").exists()
    finally:
        parent.chmod(0o755)


def test_symlinked_owned_file_is_not_followed(copied_source, tmp_path):
    out = tmp_path / "out"
    export("stage", out)
    target = tmp_path / "precious.parquet"
    target.write_text("keep")
    (out / "processed" / "train.parquet").unlink()
    (out / "processed" / "train.parquet").symlink_to(target)
    with pytest.raises(KeyParamsError, match="outside"):
        export("stage", out, overwrite=True)
    assert target.read_text() == "keep"


def test_relative_out_dir(copied_source, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    export("stage", "rel_out")
    export("stage", "rel_out", overwrite=True)
    assert (tmp_path / "rel_out" / "manifest.json").exists()
    assert _no_staging(tmp_path / "rel_out")


# --- duplicate column names: parquet says so, the other formats are unchanged ---

DUP_STEPS = [{"op": "rename", "target": "both", "params": {"mapping": {"Fare": "Age"}}}]


@pytest.fixture
def dup_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    csv = tmp_path / "train.csv"
    csv.write_bytes(Path(TRAIN_CSV).read_bytes())
    save_workspace(
        {
            "name": "dup",
            "datasets": {"train": {"x": {"kind": "csv", "path": str(csv)}}},
            "steps": [],
        }
    )
    return csv


def _with_dup_step(csv):
    save_workspace(
        {
            "name": "dup",
            "datasets": {"train": {"x": {"kind": "csv", "path": str(csv)}}},
            "steps": DUP_STEPS,
        }
    )


def test_parquet_duplicate_columns_message(dup_source, tmp_path):
    _with_dup_step(dup_source)
    with pytest.raises(SourceError) as err:
        export("dup", tmp_path / "out", formats=["parquet"])
    msg = str(err.value)
    assert "duplicate column names ['Age']" in msg
    assert "train" in msg and "rename" in msg
    assert "cast" not in msg


def test_other_formats_unchanged_with_duplicate_columns(dup_source, tmp_path):
    _with_dup_step(dup_source)
    out = tmp_path / "out"
    export("dup", out, formats=["csv", "py", "ipynb"])
    header = (out / "processed" / "train.csv").read_text().splitlines()[0]
    assert header.split(",").count("Age") == 2
    assert (out / "code" / "pipeline.py").exists()
    assert (out / "code" / "pipeline.ipynb").exists()


def test_duplicate_columns_overwrite_keeps_previous(dup_source, tmp_path):
    out = tmp_path / "out"
    export("dup", out, formats=["parquet", "csv"])
    before = _tree(out)
    _with_dup_step(dup_source)
    with pytest.raises(SourceError, match="duplicate column names"):
        export("dup", out, overwrite=True, formats=["parquet", "csv"])
    assert _tree(out) == before
    assert _no_staging(out)


def _staging(tmp_path, pid, age=0):
    import os
    import time

    d = tmp_path / ".export-staging-x"
    d.mkdir()
    marker = d / ".dtk-staging"
    marker.write_text(str(pid))
    if age:
        t = time.time() - age
        os.utime(marker, (t, t))
    return d


def _win32_no_kill(monkeypatch):
    import os

    from dtk_engine.workspace import export as mod

    def boom(*a):
        raise AssertionError("os.kill must never be called on win32")

    monkeypatch.setattr(mod, "_IS_WINDOWS", True)
    monkeypatch.setattr(os, "kill", boom)
    return mod


def test_win32_recent_staging_is_kept_without_os_kill(tmp_path, monkeypatch):
    mod = _win32_no_kill(monkeypatch)
    assert not mod._staging_is_stale(_staging(tmp_path, 999999))


def test_win32_old_staging_is_stale_without_os_kill(tmp_path, monkeypatch):
    mod = _win32_no_kill(monkeypatch)
    assert mod._staging_is_stale(_staging(tmp_path, 999999, age=mod.STAGING_MAX_AGE + 60))


def _posix_kill(monkeypatch, exc):
    import os

    from dtk_engine.workspace import export as mod

    def fake(pid, sig):
        if exc is not None:
            raise exc

    monkeypatch.setattr(mod, "_IS_WINDOWS", False)
    monkeypatch.setattr(os, "kill", fake)
    return mod


@pytest.mark.parametrize(
    ("exc", "stale"),
    [
        (ProcessLookupError(), True),  # dead pid
        (None, False),  # live pid, also a recycled pid of an unrelated process
        (PermissionError(), False),  # EPERM: exists
        (OSError("odd"), False),  # unexpected: conservative, alive
    ],
)
def test_posix_staging_liveness(tmp_path, monkeypatch, exc, stale):
    mod = _posix_kill(monkeypatch, exc)
    assert mod._staging_is_stale(_staging(tmp_path, 4242)) is stale


def test_posix_old_staging_is_stale_even_if_pid_alive(tmp_path, monkeypatch):
    mod = _posix_kill(monkeypatch, None)
    assert mod._staging_is_stale(_staging(tmp_path, 4242, age=mod.STAGING_MAX_AGE + 60))


def test_provenance_of_a_partitioned_directory(tmp_path):
    from dtk_engine.workspace.export import file_provenance

    root = tmp_path / "parts"
    (root / "year=2020").mkdir(parents=True)
    (root / "year=2021").mkdir()
    (root / "year=2021" / "b.bin").write_bytes(b"BBBB")
    (root / "year=2020" / "a.bin").write_bytes(b"AA")
    prov = file_provenance(root)
    expected = hashlib.sha256()
    for rel, blob in (("year=2020/a.bin", b"AA"), ("year=2021/b.bin", b"BBBB")):
        expected.update(rel.encode() + b"\0" + blob)
    assert prov["sha256"] == expected.hexdigest()
    assert prov["n_files"] == 2 and prov["size"] == 6
    assert prov["path"] == str(root.resolve())
    (root / "year=2020" / "a.bin").write_bytes(b"AB")
    assert file_provenance(root)["sha256"] != prov["sha256"]


def test_empty_directory_provenance(tmp_path):
    from dtk_engine.workspace.export import file_provenance

    (tmp_path / "empty").mkdir()
    prov = file_provenance(tmp_path / "empty")
    assert prov["n_files"] == 0 and prov["size"] == 0
    assert prov["sha256"] == hashlib.sha256().hexdigest()


def _export_with_bystanders(workspace, out):
    export_workspace(workspace, str(out))
    (out / "notes.txt").write_text("mine")
    (out / "processed" / "mine.txt").write_text("mine too")
    return _tree(out)


@pytest.mark.parametrize(
    ("manifest_text", "match"),
    [
        (b"\xff\xfe not utf-8", "unreadable"),
        (b"{not json", "unreadable"),
        (json.dumps({"generator": "dtk_engine", "outputs": [1]}), "malformed"),
    ],
)
def test_overwrite_refuses_unreadable_or_malformed_manifest(
    workspace, tmp_path, manifest_text, match
):
    from dtk_engine.workspace.export import GENERATOR

    out = tmp_path / "export"
    before = _export_with_bystanders(workspace, out)
    if isinstance(manifest_text, str):
        manifest_text = manifest_text.replace("dtk_engine", GENERATOR).encode()
    (out / "manifest.json").write_bytes(manifest_text)
    before["manifest.json"] = manifest_text
    with pytest.raises(KeyParamsError, match=match):
        export(workspace, out, overwrite=True)
    assert _tree(out) == before  # nothing deleted, nothing added (no staging left)


def test_overwrite_refuses_step_entries_of_the_wrong_type(workspace, tmp_path):
    out = tmp_path / "export"
    before = _export_with_bystanders(workspace, out)
    _tamper(out, lambda m: m.update(steps=[7]))
    before["manifest.json"] = (out / "manifest.json").read_bytes()
    with pytest.raises(KeyParamsError, match="malformed"):
        export(workspace, out, overwrite=True)
    assert _tree(out) == before


def test_load_state_inline_needs_no_file(tmp_path):
    state = {"mean": [1.5]}
    assert load_state(tmp_path / "missing", {"state": state}) == state


def test_load_state_checks_the_side_file_sha256(tmp_path):
    side = tmp_path / "states" / "s0.json"
    side.parent.mkdir()
    side.write_text(json.dumps({"mean": [1.5]}))
    entry = {"state_file": "states/s0.json", "state_sha256": _sha(side)}
    assert load_state(tmp_path, entry) == {"mean": [1.5]}
    side.write_text(json.dumps({"mean": [9.9]}))
    with pytest.raises(SourceError, match="does not match its manifest sha256"):
        load_state(tmp_path, entry)
