"""Pack registry: golden configs per pack (stdio / http), detect() on a fake PATH, stub script.

Regenerate the golden files with ``DTK_UPDATE_GOLDEN=1 uv run pytest tests/agent/test_packs.py``.
"""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

import pytest

from dtk_engine.agent.packs import PACKS, get_pack
from dtk_engine.agent.packs.stub import PACK as STUB
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.errors import KeyParamsError
from dtk_engine.ui_bridge import UiBridge

GOLDEN = Path(__file__).parent / "golden"
SERVERS = {
    "stdio": {"command": ["/py", "-m", "dtk_engine.agent.cli"]},
    "http": {"url": "http://127.0.0.1:8765/mcp/", "headers": {"Authorization": "Bearer T"}},
}
EXTERNAL = ["claude-code", "gemini", "opencode"]
UPDATE = os.environ.get("DTK_UPDATE_GOLDEN") == "1"


def _check(path: Path, text: str) -> None:
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.read_text() == text, f"{path} differs (DTK_UPDATE_GOLDEN=1 to regenerate)"


@pytest.mark.parametrize("variant", sorted(SERVERS))
@pytest.mark.parametrize("pack_id", EXTERNAL)
def test_golden_configs(pack_id, variant):
    pack = get_pack(pack_id)
    files = pack.files(SERVERS[variant])
    assert files
    root = GOLDEN / pack_id / variant
    for rel, content in files.items():
        _check(root / rel, json.dumps(content, indent=2) + "\n")
    written = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert written == set(files)


@pytest.mark.parametrize("pack_id", EXTERNAL)
def test_golden_launch_command(pack_id):
    command = get_pack(pack_id).launch_command(Path("/cfg"))
    _check(GOLDEN / pack_id / "launch.txt", shlex.join(command) + "\n")


def test_only_dtk_server_and_token_only_in_http():
    for pack_id in EXTERNAL:
        pack = get_pack(pack_id)
        stdio = json.dumps(pack.files(SERVERS["stdio"]))
        assert "Bearer" not in stdio
        assert "Bearer T" in json.dumps(pack.files(SERVERS["http"]))
        assert pack.panel == "external" and pack.tool_policy() and pack.auth and pack.cost


def test_stdio_env_is_passed_through():
    server = {**SERVERS["stdio"], "env": {"DTK_HOME": "/h"}}
    assert get_pack("claude-code").files(server)["dtk.mcp.json"]["mcpServers"]["dtk"]["env"] == {
        "DTK_HOME": "/h"
    }
    assert get_pack("gemini").files(server)[".gemini/settings.json"]["mcpServers"]["dtk"][
        "env"
    ] == {"DTK_HOME": "/h"}
    assert get_pack("opencode").files(server)["opencode.json"]["mcp"]["dtk"]["environment"] == {
        "DTK_HOME": "/h"
    }


def test_registry_and_unknown_pack():
    assert list(PACKS) == ["claude-code", "gemini", "opencode", "stub"]
    with pytest.raises(KeyParamsError, match="known: claude-code, gemini, opencode, stub"):
        get_pack("nope")


def _fake_bin(directory: Path, name: str) -> None:
    exe = directory / name
    exe.write_text("#!/bin/sh\necho 9.9.9\n")
    exe.chmod(0o755)


def test_detect_with_fake_path(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    full = tmp_path / "bin"
    full.mkdir()
    _fake_bin(full, "claude")
    assert get_pack("claude-code").detect(str(full))
    assert not get_pack("claude-code").detect(str(empty))
    assert not get_pack("gemini").detect(str(full))
    assert STUB.detect(str(empty))  # no CLI: always available
    assert STUB.files(SERVERS["stdio"]) == {}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_stub_run_script(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.delenv("DTK_AGENT_LOG", raising=False)
    server = build_server(LocalUiPort(UiBridge("t")), AuditLog())
    out = await STUB.run_script(server, [
        {"tool": "list_keys"},
        {"tool": "run_key", "args": {"key": "dataset_overview"}},
        {"tool": "nope", "args": {}},
    ])
    assert [(o["tool"], o["is_error"]) for o in out] == [
        ("list_keys", False), ("run_key", False), ("nope", True)
    ]
    keys = out[0]["result"]["data"]
    assert "dataset_overview" in json.dumps(keys)
    assert out[2]["result"]["message"] == "unknown tool: nope"
