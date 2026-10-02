"""Gemini CLI, external: project ``.gemini/settings.json`` with only ``dtk``.

Verified against gemini-cli 0.50.0 (``gemini --help`` and its bundled source)
and its docs:
- ``mcpServers.<name>`` (``command`` / ``args`` / ``env``, ``httpUrl`` +
  ``headers`` for streamable HTTP, ``trust``), ``mcp.allowed``,
  ``tools.exclude``, project file ``.gemini/settings.json``:
  https://geminicli.com/docs/reference/configuration/
- project settings and MCP servers are skipped in an untrusted folder
  (``--skip-trust`` trusts the cwd for one session):
  https://geminicli.com/docs/cli/trusted-folders/
- ``--allowed-mcp-server-names``: ``gemini --help``.

``tools.core: []`` is *not* used: 0.50.0 turns a set ``tools.core`` into a
``*`` deny rule ranked above trusted MCP servers, which would deny the dtk
tools too. ``tools.exclude`` lists the built-ins instead (tool names from the
bundled ``*_TOOL_NAME`` constants).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dtk_engine.agent.packs.base import SERVER_NAME, Pack, ServerRef, is_http

SETTINGS_FILE = ".gemini/settings.json"
# Built-ins that touch the shell, files, the network or spawn agents.
EXCLUDED_TOOLS = [
    "run_shell_command",
    "write_file",
    "replace",
    "read_file",
    "read_many_files",
    "list_directory",
    "glob",
    "grep_search",
    "web_fetch",
    "google_web_search",
    "invoke_agent",
    "activate_skill",
    "write_todos",
]


def _server_entry(server: ServerRef) -> dict:
    if is_http(server):
        entry = {"httpUrl": server["url"], "headers": server["headers"]}
    else:
        command, *args = server["command"]
        entry = {"command": command, "args": args}
        if server.get("env"):
            entry["env"] = server["env"]
    return {**entry, "trust": True}


@dataclass(frozen=True)
class GeminiPack(Pack):
    def files(self, server: ServerRef) -> dict[str, dict]:
        settings = {
            "mcpServers": {SERVER_NAME: _server_entry(server)},
            "mcp": {"allowed": [SERVER_NAME]},
            "tools": {"exclude": EXCLUDED_TOOLS},
        }
        return {SETTINGS_FILE: settings}

    def launch_command(self, directory: Path) -> list[str]:
        return ["gemini", "--skip-trust", "--allowed-mcp-server-names", SERVER_NAME]

    def tool_policy(self) -> str:
        return (
            "Run it from the config directory: gemini reads .gemini/settings.json from the "
            "cwd, and --skip-trust trusts that folder for the session (an untrusted folder "
            "loads no project settings and no MCP server). mcp.allowed and "
            "--allowed-mcp-server-names keep only the dtk server; tools.exclude removes the "
            "shell, file read / write / edit, web fetch / search, subagent, skill and todo "
            "built-ins; trust: true runs dtk tools without a prompt (destructive Studio "
            "edits still wait for your review in Studio). Built-ins outside that list "
            "(ask_user, plan mode, task tracker) and your user settings / GEMINI.md stay on: "
            "that part is your own gemini session."
        )


PACK = GeminiPack(
    id="gemini",
    title="Gemini CLI",
    panel="external",
    cli="gemini",
    auth="your gemini login (Google account) or GEMINI_API_KEY",
    cost="billed to your Google account / Gemini API key by Google",
)
