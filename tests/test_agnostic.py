"""Agnostic guard: the engine names no front (Studio, datatoolkit-web) nor UI framework.

Scans every ``src/dtk_engine`` file (code, docstrings, comments, param names).
The HTTP API (FastAPI, ``dtk-api``) is generic and allowed; so is the word
"front" for "any client of the contract". Known leaks that need a contract change
are frozen in ``KNOWN_LEAKS`` (file -> max hits): the list may only shrink, a new
hit anywhere fails.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "dtk_engine"

FORBIDDEN = re.compile(
    r"\bstudio\b|datatoolkit[-_]web|\breact\b|streamlit|gradio|\bflask\b|\bnicegui\b"
    r"|\bshiny\b|\bvue\b|\bsvelte\b|\btailwind\b|\bnext\.?js\b|\bdash ?apps?\b"
    r"|\bimport dash\b|\bfrom dash\b|\bfront-?ends?\b|\bbrowsers?\b|\bjavascript\b"
    r"|\btypescript\b",
    re.IGNORECASE,
)

# Open leaks (see the PR): the agent/chat layer, the UI bridge and the studio
# routes name Studio in their protocol (command names, error codes, route names).
# Fixing them changes public contracts: do it by lowering these numbers, never up.
# The plotly "browser" renderer (result.py) and HTTP Origin checks are technical.
KNOWN_LEAKS: dict[str, int] = {
    "agent/attachments.py": 2,
    "agent/chat.py": 8,
    "agent/cli.py": 1,
    "agent/commands.py": 33,
    "agent/digest.py": 8,
    "agent/doctor.py": 3,
    "agent/options.py": 1,
    "agent/packs/agent_sdk.py": 9,
    "agent/packs/base.py": 1,
    "agent/packs/chat_packs.py": 1,
    "agent/packs/claude_code.py": 2,
    "agent/packs/gemini.py": 2,
    "agent/packs/opencode.py": 2,
    "agent/packs/stub.py": 5,
    "agent/ports.py": 1,
    "agent/server.py": 3,
    "agent/terminal.py": 3,
    "agent/tool_meta.py": 5,
    "agent/tools.py": 18,
    "http.py": 9,
    "result.py": 6,
    "ui_bridge.py": 9,
}


def scan(root: Path) -> dict[str, list[str]]:
    """``{relative path: [offending lines]}`` for every file under ``root``."""
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        hits = [f"{n}: {line.strip()}" for n, line in enumerate(lines, 1) if FORBIDDEN.search(line)]
        if hits:
            found[path.relative_to(root).as_posix()] = hits
    return found


def test_scanner_flags_offending_file(tmp_path):
    (tmp_path / "bad.py").write_text(
        '"""Used by Studio."""\n# talks to datatoolkit-web\nimport streamlit\n'
        "def f(react_to=1): ...\n"
    )
    hits = scan(tmp_path)["bad.py"]
    assert len(hits) == 3  # 'react_to' is a param name, not the word React
    (tmp_path / "bad2.py").write_text("x = 1  # open it in the browser\n")
    assert "bad2.py" in scan(tmp_path)


def test_scanner_passes_clean_file(tmp_path):
    (tmp_path / "ok.py").write_text(
        '"""FastAPI server behind dtk-api: the HTTP API, any front or client."""\n'
        "# a front reads the JSON schema\n"
    )
    assert scan(tmp_path) == {}


def test_engine_is_agnostic():
    found = scan(ROOT)
    over = {
        f: hits
        for f, hits in found.items()
        if len(hits) > KNOWN_LEAKS.get(f, 0)
    }
    assert not over, "front/UI names in the engine (agnostic principle):\n" + "\n".join(
        f"{f}:{h}" for f, hits in over.items() for h in hits
    )
    stale = {f: n for f, n in KNOWN_LEAKS.items() if len(found.get(f, [])) < n}
    assert not stale, f"lower KNOWN_LEAKS (leaks fixed): {stale}"
