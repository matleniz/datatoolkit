"""Runtime file: dtk-api publishes its URL and per-run token under $DTK_HOME/agent."""

from __future__ import annotations

import json
import os
import stat

import pytest

from dtk_engine import http, ui_bridge
from dtk_engine.ui_bridge import (
    clear_runtime,
    read_runtime,
    runtime_path,
    write_runtime,
)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.delenv("DTK_UI_RUNTIME_FILE", raising=False)
    return tmp_path


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_write_read_round_trip(home):
    path = write_runtime("http://127.0.0.1:8765", "tok")
    assert path == home / "agent" / "runtime.json" == runtime_path()
    data = read_runtime()
    assert data is not None
    assert {k: data[k] for k in ("url", "token", "pid")} == {
        "url": "http://127.0.0.1:8765", "token": "tok", "pid": os.getpid(),
    }
    assert data["started"]


def test_owner_only_permissions(home):
    path = write_runtime("http://127.0.0.1:8765", "tok")
    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["runtime.json"]  # no temp left


def test_dead_pid_is_ignored(home, monkeypatch):
    write_runtime("http://127.0.0.1:8765", "tok")
    monkeypatch.setattr(ui_bridge, "_pid_alive", lambda pid: False)
    assert read_runtime() is None


def test_garbage_or_missing_file_is_ignored(home):
    assert read_runtime() is None
    runtime_path().parent.mkdir(parents=True)
    runtime_path().write_text("{not json")
    assert read_runtime() is None
    runtime_path().write_text(json.dumps({"url": "x", "token": "t"}))  # no pid
    assert read_runtime() is None


def test_clear_only_own_file(home):
    path = write_runtime("http://127.0.0.1:8765", "tok")
    clear_runtime(pid=os.getpid() + 1)
    assert path.exists()
    clear_runtime()
    assert not path.exists()
    clear_runtime()  # missing file: no error


def test_opt_out_writes_nothing(home, monkeypatch):
    monkeypatch.setenv("DTK_UI_RUNTIME_FILE", "0")
    assert write_runtime("http://127.0.0.1:8765", "tok") is None
    assert not runtime_path().exists()


@pytest.mark.parametrize(
    ("argv", "url"),
    [
        ([], "http://127.0.0.1:8765"),
        (["--host", "0.0.0.0", "--port", "9000"], "http://127.0.0.1:9000"),
        (["--host", "::1", "--port", "9001"], "http://[::1]:9001"),
    ],
)
def test_main_writes_then_clears(home, monkeypatch, argv, url):
    import uvicorn

    seen: dict = {}

    def fake_run(app, **kwargs):
        seen["runtime"] = read_runtime()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    http.main(argv)
    assert seen["runtime"]["url"] == url
    assert seen["runtime"]["token"] == http.app.state.ui_bridge.token
    assert not runtime_path().exists()


def test_main_clears_on_crash(home, monkeypatch):
    import uvicorn

    def boom(app, **kwargs):
        raise RuntimeError("bind failed")

    monkeypatch.setattr(uvicorn, "run", boom)
    with pytest.raises(RuntimeError):
        http.main([])
    assert not runtime_path().exists()


def test_one_dtk_home_resolver_expands_user(tmp_path, monkeypatch):
    from pathlib import Path

    from dtk_engine.agent import attachments, policy

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("DTK_UPLOAD_DIR", raising=False)
    monkeypatch.setenv("DTK_HOME", "~/dtk")
    home = tmp_path / "dtk"
    assert ui_bridge.dtk_home() == home
    assert runtime_path() == home / "agent" / "runtime.json"  # was left unexpanded
    assert attachments.upload_dir() == home / "uploads"
    assert Path(os.path.realpath(home)) in policy.allowed_roots()
    monkeypatch.delenv("DTK_HOME")
    assert ui_bridge.dtk_home() == tmp_path / ".datatoolkit"
