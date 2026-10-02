"""``dtk-mcp config`` (print / write / refuse / --http) and ``dtk-mcp doctor`` (real socket)."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from dtk_engine.agent import cli
from dtk_engine.http import create_app
from dtk_engine.ui_bridge import write_runtime

TOKEN = "config-test-token"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DTK_HOME", str(h))
    monkeypatch.setenv("DTK_UI_TOKEN", TOKEN)
    monkeypatch.delenv("DTK_UI_RUNTIME_FILE", raising=False)
    return h


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    try:
        cli.main(list(argv))
        code = 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    out, err = capsys.readouterr()
    return code, out, err


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def engine():
    port = _free_port()
    app = create_app()
    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error",
                       timeout_graceful_shutdown=1)
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield app, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(5)


# -- config ------------------------------------------------------------------


def test_config_prints_files_and_run_line_without_token(capsys, engine, home):
    write_runtime(engine[1], TOKEN)  # a running engine must not leak into stdio configs
    code, out, _ = _run(capsys, "config", "claude-code")
    assert code == 0
    assert out.startswith("# dtk.mcp.json\n")
    assert "claude --strict-mcp-config --mcp-config dtk.mcp.json --tools ''" in out
    assert "# tool policy:" in out and "# auth:" in out
    assert TOKEN not in out
    assert f'"DTK_HOME": "{home.resolve()}"' in out


def test_config_write_refuse_force(capsys, tmp_path):
    target = tmp_path / "agent"
    code, out, _ = _run(capsys, "config", "gemini", "--write", str(target))
    settings = target / ".gemini" / "settings.json"
    assert code == 0
    assert json.loads(settings.read_text())["mcp"] == {"allowed": ["dtk"]}
    assert f"cd {target} && gemini --skip-trust --allowed-mcp-server-names dtk" in out

    settings.write_text("{}")
    code, _, err = _run(capsys, "config", "gemini", "--write", str(target))
    assert code == 2
    assert "refusing to overwrite" in err
    assert settings.read_text() == "{}"

    code, _, _ = _run(capsys, "config", "gemini", "--write", str(target), "--force")
    assert code == 0
    assert "mcpServers" in json.loads(settings.read_text())


def test_config_unknown_pack_exits_2(capsys):
    code, _, err = _run(capsys, "config", "nope")
    assert code == 2
    assert "known: claude-code, gemini, opencode, stub" in err


def test_config_http_without_engine_exits_2(capsys, tmp_path):
    code, _, err = _run(capsys, "config", "opencode", "--http")
    assert code == 2
    assert "no running engine (start dtk-api)" in err
    code, _, _ = _run(capsys, "config", "opencode", "--http", "--write", str(tmp_path / "x"))
    assert code == 2
    assert not (tmp_path / "x").exists()


def test_config_http_uses_runtime_url_and_token(capsys, engine, tmp_path):
    write_runtime(engine[1], TOKEN)
    code, out, _ = _run(capsys, "config", "opencode", "--http", "--write", str(tmp_path / "oc"))
    assert code == 0
    assert "valid for this engine run only" in out
    written = tmp_path / "oc" / "opencode.json"
    entry = json.loads(written.read_text())["mcp"]["dtk"]
    assert entry["url"] == engine[1] + "/mcp/"
    assert entry["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert written.stat().st_mode & 0o777 == 0o600


# -- doctor ------------------------------------------------------------------


@pytest.fixture
def fake_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "claude"
    exe.write_text("#!/bin/sh\necho '1.2.3 (Fake)'\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    return bin_dir


def _doctor_json(capsys) -> tuple[int, dict]:
    code, out, _ = _run(capsys, "doctor", "--json")
    return code, json.loads(out)


def test_doctor_engine_down(capsys, fake_path):
    code, report = _doctor_json(capsys)
    assert code == 1
    assert report["ok"] is False
    assert report["runtime"]["found"] is False
    assert report["engine"] == {
        "reachable": False, "token_valid": None, "studio_connected": False, "sessions": []
    }
    packs = {p["id"]: p for p in report["packs"]}
    assert set(packs) == {"claude-code", "gemini", "opencode"}
    assert packs["claude-code"] == {
        "id": "claude-code", "cli": "claude", "detected": True,
        "path": str(fake_path / "claude"), "version": "1.2.3 (Fake)",
    }
    assert packs["gemini"]["detected"] is False and packs["gemini"]["version"] is None

    # A runtime file whose (live) pid serves nothing on that port: unreachable.
    write_runtime(f"http://127.0.0.1:{_free_port()}", TOKEN)
    code, report = _doctor_json(capsys)
    assert code == 1
    assert report["runtime"]["found"] is True
    assert report["runtime"]["pid"] == os.getpid()
    assert report["engine"]["reachable"] is False


def test_doctor_valid_token_and_studio(capsys, engine, fake_path):
    app, url = engine
    write_runtime(url, TOKEN)
    code, report = _doctor_json(capsys)
    assert code == 0
    assert report["engine"]["token_valid"] is True
    assert report["engine"]["studio_connected"] is False

    bridge = app.state.ui_bridge
    bridge.put_context({"session": "s1", "workspace": "w"})
    bridge.add_listener("s1")
    code, report = _doctor_json(capsys)
    assert code == 0
    assert report["engine"]["studio_connected"] is True
    assert report["engine"]["sessions"] == [
        {"session": "s1", "listening": True, "workspace": "w", "identity": None}
    ]

    code, out, _ = _run(capsys, "doctor")
    assert code == 0
    assert "studio connected ok" in out
    assert "pack claude-code" in out and "1.2.3 (Fake)" in out
    assert "pack gemini      not on PATH" in out
    assert TOKEN not in out


def test_doctor_invalid_token(capsys, engine, fake_path):
    write_runtime(engine[1], "wrong-token")
    code, report = _doctor_json(capsys)
    assert code == 1
    assert report["engine"]["reachable"] is True
    assert report["engine"]["token_valid"] is False
    code, out, _ = _run(capsys, "doctor")
    assert "token valid      NO" in out


def test_runtime_path_reported(capsys, home):
    _, report = _doctor_json(capsys)
    assert Path(report["runtime"]["path"]) == home / "agent" / "runtime.json"
