"""Drift check: the command table against Studio's parser (datatoolkit-web).

Reads ``src/state/agentCommands.ts`` and the enums it uses from a web checkout
(``DTK_WEB_DIR``, else ``~/datatoolkit-web``); skipped when there is none. The
reverse check (Studio's parser against ``GET /api/ui/commands/schema``) lives
in the web repo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from dtk_engine.agent import commands
from dtk_engine.agent.commands import UI_COMMANDS

WEB = Path(os.environ.get("DTK_WEB_DIR") or Path.home() / "datatoolkit-web")
STATE = WEB / "src" / "state"

pytestmark = pytest.mark.skipif(
    not (STATE / "agentCommands.ts").is_file(), reason=f"no datatoolkit-web checkout at {WEB}"
)


def _read(name: str) -> str:
    return (STATE / name).read_text(encoding="utf-8")


def _const_list(source: str, name: str) -> list[str]:
    """Strings of ``const NAME = [ ... ]`` (``as const`` arrays)."""
    match = re.search(rf"const {name}\b[^=]*=\s*\[(.*?)\]", source, re.DOTALL)
    assert match, f"{name} not found"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_command_types_match_parse_command():
    source = _read("agentCommands.ts")
    body = source[source.index("export function parseCommand"):]
    body = body[: body.index("default:")]
    assert set(re.findall(r'case "([a-z_]+)":', body)) == set(UI_COMMANDS)


def test_enums_match_studio():
    assert _const_list(_read("dockTypes.ts"), "TOOL_IDS") == commands.WINDOWS
    chart = _read("chartDraft.ts")
    assert _const_list(chart, "CHART_TYPES") == commands.CHART_TYPES
    assert _const_list(chart, "CHART_AGGS") == commands.CHART_AGGS
    source = _read("agentCommands.ts")
    assert _const_list(source, "VARIABLE_STATS") == commands.VARIABLE_STATS
    ops = re.findall(r'\{\s*op:\s*"([a-z]+)"', _read("gridView.ts"))
    assert ops == commands.FILTER_OPS
    cap = re.search(r"const CHART_NAME_MAX = (\d+);", source)
    assert cap and int(cap.group(1)) == commands.CHART_NAME_MAX


def test_chart_param_keys_match_studio():
    source = _read("agentCommands.ts")
    rules = source[source.index("const CHART_PARAM_RULES"):]
    rules = rules[: rules.index("};")]
    keys = set(re.findall(r"^\s{2}(\w+): \[", rules, re.MULTILINE))
    keys |= set(_const_list(source, "CHART_COLUMN_KEYS"))
    keys |= set(_const_list(source, "CHART_BOOL_KEYS"))
    schema = UI_COMMANDS["draft_chart"].input_schema["properties"]["params"]
    assert set(schema["properties"]) == keys
