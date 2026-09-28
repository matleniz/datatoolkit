import threading

import pytest
from pydantic import ValidationError

from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace import JsonWorkspaceStore, Workspace, WorkspaceNotFoundError
from dtk_engine.workspace.store import default_root


def _ws(name="demo", **kw):
    return Workspace.model_validate(
        {"name": name, "datasets": {"train": {"x": {"kind": "csv", "path": "x.csv"}}}}
        | kw
    )


def test_model_defaults():
    ws = _ws()
    assert ws.label.mode == "order" and ws.steps == [] and ws.datasets.test is None


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "../evil"},
        {"name": ".hidden"},
        {"label": {"mode": "key"}},
        {"steps": [{"op": "x", "target": "all"}]},
        {"extra": 1},
    ],
)
def test_model_is_strict(bad):
    with pytest.raises(ValidationError):
        _ws(**bad)


def test_y_and_target_column_exclusive():
    with pytest.raises(ValidationError, match="not both"):
        Workspace.model_validate(
            {
                "name": "w",
                "datasets": {
                    "train": {
                        "x": {"kind": "csv", "path": "x.csv"},
                        "y": {"kind": "csv", "path": "y.csv"},
                        "target_column": "t",
                    }
                },
            }
        )


def test_dataset_source_not_allowed_as_workspace_input():
    with pytest.raises(ValidationError):
        Workspace.model_validate(
            {
                "name": "w",
                "datasets": {"train": {"x": {"kind": "dataset", "workspace": "w"}}},
            }
        )


def test_store_roundtrip(tmp_path):
    store = JsonWorkspaceStore(tmp_path)
    assert store.list() == []
    store.save(_ws("b"))
    store.save(_ws("a", steps=[{"op": "noop", "target": "both"}]))
    assert store.list() == ["a", "b"]
    assert store.get("a").steps[0].target == "both"
    store.delete("b")
    assert store.list() == ["a"]
    with pytest.raises(WorkspaceNotFoundError):
        store.get("b")
    with pytest.raises(WorkspaceNotFoundError):
        store.delete("b")


def test_store_rejects_bad_name_and_corrupt_file(tmp_path):
    store = JsonWorkspaceStore(tmp_path)
    with pytest.raises(KeyParamsError):
        store.get("../x")
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(KeyParamsError, match="invalid workspace file"):
        store.get("bad")


def test_default_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    assert default_root() == tmp_path / "workspaces"
    assert JsonWorkspaceStore().root == tmp_path / "workspaces"
    monkeypatch.delenv("DTK_HOME")
    assert default_root().parts[-2:] == (".datatoolkit", "workspaces")


def test_store_rename_and_duplicate(tmp_path):
    store = JsonWorkspaceStore(tmp_path)
    store.save(
        _ws(
            "src",
            steps=[
                {
                    "op": "drop_columns",
                    "target": "both",
                    "params": {"columns": ["a"]},
                }
            ],
            variables=[{"name": "m", "stat": "mean", "column": "a"}],
        )
    )
    renamed = store.rename("src", "dst")
    assert renamed.name == "dst"
    assert store.list() == ["dst"]
    assert store.get("dst").steps[0].op == "drop_columns"
    assert store.get("dst").variables[0].name == "m"
    with pytest.raises(WorkspaceNotFoundError):
        store.get("src")

    copy = store.duplicate("dst", "copy")
    assert copy.name == "copy"
    assert store.list() == ["copy", "dst"]
    # Content-addressed source path stays shared (not rewritten / copied).
    assert (
        store.get("copy").datasets.train.x.path
        == store.get("dst").datasets.train.x.path
    )
    assert store.get("copy").steps == store.get("dst").steps

    with pytest.raises(KeyParamsError, match="already exists"):
        store.rename("dst", "copy")
    with pytest.raises(KeyParamsError, match="already exists"):
        store.duplicate("dst", "copy")
    with pytest.raises(KeyParamsError, match="must differ"):
        store.duplicate("dst", "dst")
    assert store.rename("dst", "dst").name == "dst"


def test_save_then_rename_preserves_put_content(tmp_path):
    """Under the store lock, a completed PUT is what rename moves (no lost update)."""
    store = JsonWorkspaceStore(tmp_path)
    store.save(_ws("w", steps=[]))
    save_done = threading.Event()
    errors: list[BaseException] = []

    def do_save():
        try:
            store.save(
                _ws(
                    "w",
                    steps=[
                        {
                            "op": "drop_columns",
                            "target": "both",
                            "params": {"columns": ["a"]},
                        }
                    ],
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            save_done.set()

    def do_rename():
        try:
            assert save_done.wait(timeout=5)
            store.rename("w", "w2")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=do_save)
    t2 = threading.Thread(target=do_rename)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert errors == []
    assert store.list() == ["w2"]
    assert store.get("w2").steps[0].params["columns"] == ["a"]


def test_rename_then_save_does_not_clobber_renamed_copy(tmp_path):
    """Rename first, then PUT to the old name: renamed copy keeps its content."""
    store = JsonWorkspaceStore(tmp_path)
    store.save(
        _ws(
            "w",
            steps=[
                {
                    "op": "drop_columns",
                    "target": "train",
                    "params": {"columns": ["old"]},
                }
            ],
        )
    )
    renamed_done = threading.Event()
    errors: list[BaseException] = []

    def do_rename():
        try:
            store.rename("w", "w2")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            renamed_done.set()

    def do_save():
        try:
            assert renamed_done.wait(timeout=5)
            store.save(
                _ws(
                    "w",
                    steps=[
                        {
                            "op": "drop_columns",
                            "target": "both",
                            "params": {"columns": ["new"]},
                        }
                    ],
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=do_rename)
    t2 = threading.Thread(target=do_save)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert errors == []
    assert set(store.list()) == {"w", "w2"}
    # Renamed workspace still has the pre-rename content (PUT did not overwrite it).
    assert store.get("w2").steps[0].params["columns"] == ["old"]
    assert store.get("w").steps[0].params["columns"] == ["new"]


def test_delete_under_concurrent_put_is_exclusive(tmp_path):
    store = JsonWorkspaceStore(tmp_path)
    store.save(_ws("w"))
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def do_delete():
        try:
            barrier.wait(timeout=5)
            store.delete("w")
        except WorkspaceNotFoundError:
            pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def do_save():
        try:
            barrier.wait(timeout=5)
            store.save(_ws("w", steps=[{"op": "noop", "target": "train"}]))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=do_delete)
    t2 = threading.Thread(target=do_save)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert errors == []
    if store.list():
        assert store.get("w").name == "w"
    else:
        with pytest.raises(WorkspaceNotFoundError):
            store.get("w")
