"""opencode, external: project ``opencode.json`` with ``dtk`` and every other tool denied.

Verified against opencode 1.17.18 (``opencode --help``; the resolved config
checked with ``opencode debug agent build``: every built-in tool ends up
disabled, ``dtk_*`` allowed) and its docs:
- ``mcp.<name>`` (``type`` local + ``command`` array + ``environment``, or
  remote + ``url`` + ``headers``; ``enabled``), tools named ``<server>_<tool>``:
  https://opencode.ai/docs/mcp-servers/
- ``permission`` (``allow`` / ``ask`` / ``deny``, wildcards, last match wins;
  ``edit`` covers write / patch; the boolean ``tools`` key is deprecated):
  https://opencode.ai/docs/permissions/ and https://opencode.ai/docs/tools/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dtk_engine.agent.packs.base import SERVER_NAME, Pack, ServerRef, is_http

CONFIG_FILE = "opencode.json"


def _server_entry(server: ServerRef) -> dict:
    if is_http(server):
        return {"type": "remote", "url": server["url"], "headers": server["headers"],
                "enabled": True}
    entry = {"type": "local", "command": list(server["command"]), "enabled": True}
    if server.get("env"):
        entry["environment"] = server["env"]
    return entry


@dataclass(frozen=True)
class OpencodePack(Pack):
    def files(self, server: ServerRef) -> dict[str, dict]:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "mcp": {SERVER_NAME: _server_entry(server)},
            "permission": {"*": "deny", f"{SERVER_NAME}_*": "allow"},
        }
        return {CONFIG_FILE: config}

    def launch_command(self, directory: Path) -> list[str]:
        return ["opencode"]

    def tool_policy(self) -> str:
        return (
            "Run it from the config directory: opencode reads opencode.json from the "
            "project it starts in. permission \"*\": \"deny\" turns off every built-in "
            "tool (bash, read, edit / write / patch, grep, glob, webfetch, websearch, task, "
            "skill, todo) and the tools of your other MCP servers; \"dtk_*\": \"allow\" "
            "runs dtk tools without a prompt (destructive Studio edits still wait for your "
            "review in Studio). Your global opencode config (providers, plugins, other MCP "
            "servers, which still start) stays yours."
        )


PACK = OpencodePack(
    id="opencode",
    title="opencode",
    panel="external",
    cli="opencode",
    auth="the provider you configured in opencode (opencode auth login)",
    cost="billed by the model provider you configured in opencode",
)
