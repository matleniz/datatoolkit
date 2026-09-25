"""Smoke test of the Streamlit app (headless AppTest) with a workspace."""

from pathlib import Path

import dtk_streamlit
from dtk_engine import save_workspace
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from streamlit.testing.v1 import AppTest

APP = str(Path(dtk_streamlit.__file__).parent / "app.py")


def test_app_without_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.selectbox[0].value == "(none)"


def test_app_prefills_forms_from_active_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    save_workspace(
        {
            "name": "demo",
            "datasets": {
                "train": {"x": {"kind": "csv", "path": TRAIN_CSV}},
                "test": {"x": {"kind": "csv", "path": TEST_CSV}},
            },
        }
    )
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox[0].select("demo").run()
    assert not at.exception
    assert any("steps" in m.value for m in at.markdown)
    kinds = [s.value for s in at.selectbox if s.label.endswith("kind")]
    assert kinds and set(kinds) == {"dataset"}
    run = next(b for b in at.button if b.label == "Run")
    run.click().run()
    assert not at.exception and not at.error
    assert at.metric


def test_app_creates_workspace_through_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox[0].select("+ new workspace").run()
    fields = {t.label: t for t in at.text_input}
    fields["Name"].input("titanic")
    fields["X_train path"].input(TRAIN_CSV)
    fields["Target column (in X_train)"].input("Survived")
    fields["X_test path"].input(TEST_CSV)
    next(c for c in at.checkbox if c.label == "y already in X_train").check()
    next(b for b in at.button if b.label == "Save workspace").click().run()
    assert not at.exception and not at.error
    saved = (tmp_path / "workspaces" / "titanic.json").read_text()
    assert '"target_column": "Survived"' in saved
    assert at.session_state["active_workspace"] == "titanic"
    assert at.selectbox[0].value == "titanic"
