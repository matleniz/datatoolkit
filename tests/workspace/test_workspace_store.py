import pytest
from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace import JsonWorkspaceStore, Workspace, WorkspaceNotFoundError
from dtk_engine.workspace.store import default_root
from pydantic import ValidationError


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
