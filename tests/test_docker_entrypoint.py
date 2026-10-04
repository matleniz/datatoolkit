"""``docker/entrypoint.sh``: the opt-in agent switch (DTK_AGENT) and its token guard.

Runs the script as the current (non-root) user with a fake ``dtk-api`` on PATH
that prints its argv and the agent env, so the privilege drop is not exercised
here (the CI ``smoke`` job covers it in the real image).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "entrypoint.sh"
CMD = ["dtk-api", "--host", "0.0.0.0", "--port", "8765"]
TOKEN = "s3cret-token-value"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("sh") is None or os.geteuid() == 0,
    reason="needs a POSIX sh, run as non-root",
)

FAKE_API = """#!/bin/sh
echo "argv: $*"
echo "pack: ${DTK_AGENT_PACK-<unset>}"
echo "terminal: ${DTK_AGENT_TERMINAL-<unset>}"
"""


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    fake = tmp_path / "dtk-api"
    fake.write_text(FAKE_API)
    fake.chmod(0o755)
    return tmp_path


def run(bin_dir: Path, env: dict[str, str], cmd: list[str] = CMD) -> subprocess.CompletedProcess:
    base = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        ["sh", str(ENTRYPOINT), *cmd], env={**base, **env},
        capture_output=True, text=True, timeout=10, check=False,
    )


def test_default_runs_cmd_unchanged(bin_dir):
    out = run(bin_dir, {})
    assert out.returncode == 0, out.stderr
    assert "argv: --host 0.0.0.0 --port 8765\n" in out.stdout
    assert "pack: <unset>" in out.stdout


def test_agent_off_ignores_a_stray_pack_and_terminal(bin_dir):
    out = run(bin_dir, {"DTK_AGENT": "0", "DTK_AGENT_PACK": "stub", "DTK_AGENT_TERMINAL": "1"})
    assert out.returncode == 0, out.stderr
    assert "argv: --host 0.0.0.0 --port 8765\n" in out.stdout
    assert "pack: <unset>" in out.stdout
    assert "terminal: <unset>" in out.stdout


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_agent_on_adds_the_default_api_pack(bin_dir, value):
    out = run(bin_dir, {"DTK_AGENT": value, "DTK_UI_TOKEN": TOKEN})
    assert out.returncode == 0, out.stderr
    assert "argv: --host 0.0.0.0 --port 8765 --agent api-anthropic\n" in out.stdout
    assert TOKEN not in out.stdout + out.stderr


@pytest.mark.parametrize("pack", ["api-openai", "stub"])
def test_agent_pack_override(bin_dir, pack):
    out = run(bin_dir, {"DTK_AGENT": "1", "DTK_UI_TOKEN": TOKEN, "DTK_AGENT_PACK": pack})
    assert out.returncode == 0, out.stderr
    assert f"--agent {pack}\n" in out.stdout


def test_agent_never_enables_the_terminal(bin_dir):
    env = {"DTK_AGENT": "1", "DTK_UI_TOKEN": TOKEN, "DTK_AGENT_TERMINAL": "1"}
    out = run(bin_dir, env)
    assert out.returncode == 0, out.stderr
    assert "terminal: <unset>" in out.stdout
    assert "--terminal" not in out.stdout


@pytest.mark.parametrize("token", [None, ""])
def test_agent_without_token_refuses_to_start(bin_dir, token):
    env = {"DTK_AGENT": "1"} if token is None else {"DTK_AGENT": "1", "DTK_UI_TOKEN": token}
    out = run(bin_dir, env)
    assert out.returncode == 64
    assert out.stdout == ""
    assert "DTK_UI_TOKEN" in out.stderr


@pytest.mark.parametrize("pack", ["agent-sdk", "claude-code", "bogus"])
def test_agent_refuses_a_pack_the_image_cannot_run(bin_dir, pack):
    out = run(bin_dir, {"DTK_AGENT": "1", "DTK_UI_TOKEN": TOKEN, "DTK_AGENT_PACK": pack})
    assert out.returncode == 64
    assert out.stdout == ""
    assert pack in out.stderr
    assert TOKEN not in out.stderr


def test_agent_leaves_another_command_alone(bin_dir):
    out = run(bin_dir, {"DTK_AGENT": "1"}, ["echo", "hello"])
    assert out.returncode == 0, out.stderr
    assert out.stdout == "hello\n"
