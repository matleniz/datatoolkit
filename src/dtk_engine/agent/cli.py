"""``dtk-mcp``: the MCP server over stdio.

Contract calls run in this process against the same ``$DTK_HOME`` store; UI
tools go through the running ``dtk-api`` found in the runtime file (else
``no_studio``). Subcommands beyond ``serve`` (``config`` / ``doctor``, #66)
register in ``SUBCOMMANDS``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import anyio
from mcp.server.stdio import stdio_server

from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import RemoteUiPort
from dtk_engine.agent.server import build_server


async def _serve_stdio() -> None:
    server = build_server(RemoteUiPort(), AuditLog())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def serve(args: argparse.Namespace) -> None:
    anyio.run(_serve_stdio)


SUBCOMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "serve": ("run the MCP server over stdio (default)", serve),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dtk-mcp", description="datatoolkit MCP server")
    commands = parser.add_subparsers(dest="command")
    for name, (help_text, _) in SUBCOMMANDS.items():
        commands.add_parser(name, help=help_text)
    args = parser.parse_args(argv)
    SUBCOMMANDS[args.command or "serve"][1](args)


if __name__ == "__main__":
    main()
