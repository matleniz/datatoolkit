"""Claude Code, external: ``claude`` with only the ``dtk`` server and no built-in tool.

Verified against Claude Code 2.1.287 (``claude --help``) and its docs:
- MCP config file format (``mcpServers``, ``type`` stdio / http, ``headers``):
  https://code.claude.com/docs/en/mcp
- ``--strict-mcp-config``, ``--mcp-config``, ``--tools ""`` (no built-in tool;
  MCP tools are not affected), ``--allowedTools``:
  https://code.claude.com/docs/en/cli-reference
- ``mcp__<server>__*`` allow rule: https://code.claude.com/docs/en/permissions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dtk_engine.agent.packs.base import SERVER_NAME, Pack, ServerRef, is_http

CONFIG_FILE = "dtk.mcp.json"


def _server_entry(server: ServerRef) -> dict:
    if is_http(server):
        return {"type": "http", "url": server["url"], "headers": server["headers"]}
    command, *args = server["command"]
    entry = {"type": "stdio", "command": command, "args": args}
    if server.get("env"):
        entry["env"] = server["env"]
    return entry


@dataclass(frozen=True)
class ClaudeCodePack(Pack):
    def files(self, server: ServerRef) -> dict[str, dict]:
        return {CONFIG_FILE: {"mcpServers": {SERVER_NAME: _server_entry(server)}}}

    def launch_command(self, directory: Path) -> list[str]:
        return [
            "claude",
            "--strict-mcp-config",
            "--mcp-config", str(directory / CONFIG_FILE),
            "--tools", "",
            "--allowedTools", f"mcp__{SERVER_NAME}__*",
        ]

    def tool_policy(self) -> str:
        return (
            "--strict-mcp-config loads only the dtk server (your other MCP servers are "
            "ignored); --tools \"\" removes every built-in tool (shell, file read / edit, "
            "web fetch / search, subagents); --allowedTools mcp__dtk__* lets the dtk tools "
            "run without a prompt (destructive Studio edits still wait for your review in "
            "Studio). Your CLAUDE.md, hooks and settings still load: that part is your own "
            "Claude Code session."
        )


PACK = ClaudeCodePack(
    id="claude-code",
    title="Claude Code",
    panel="external",
    cli="claude",
    auth="your Claude Code login (claude auth login) or ANTHROPIC_API_KEY",
    cost="billed to your Claude subscription / API key by Anthropic",
)
