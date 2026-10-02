"""``dtk-mcp``: the MCP server over stdio.

Contract calls run in this process against the same ``$DTK_HOME`` store; UI
tools go through the running ``dtk-api`` found in the runtime file (else
``no_studio``). Subcommands beyond ``serve`` (``config`` / ``doctor``, #66)
register in ``SUBCOMMANDS``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

import anyio
from mcp.server.stdio import stdio_server

from dtk_engine.agent import configure, doctor
from dtk_engine.agent.packs import PACKS, get_pack
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import RemoteUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.errors import KeyParamsError


async def _serve_stdio() -> None:
    server = build_server(RemoteUiPort(), AuditLog())
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def serve(args: argparse.Namespace) -> None:
    anyio.run(_serve_stdio)


def config(args: argparse.Namespace) -> None:
    try:
        pack = get_pack(args.pack)
        text = configure.run(pack, write=args.write, force=args.force, http=args.http)
    except (configure.ConfigError, KeyParamsError) as exc:
        print(f"dtk-mcp config: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    sys.stdout.write(text)


def doctor_command(args: argparse.Namespace) -> None:
    report = doctor.diagnose()
    sys.stdout.write(json.dumps(report, indent=2) + "\n" if args.json else doctor.render(report))
    raise SystemExit(0 if report["ok"] else 1)


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pack", help=f"one of: {', '.join(PACKS)}")
    parser.add_argument("--write", metavar="DIR", help="write the files under DIR")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--http", action="store_true",
        help="use the running engine's /mcp URL + token (valid for this run only)",
    )


def _doctor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print the report as JSON")


SUBCOMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "serve": ("run the MCP server over stdio (default)", serve),
    "config": ("print or write an agent CLI's config wired to dtk only", config),
    "doctor": ("check the engine, its token, Studio and the agent CLIs", doctor_command),
}
ARGUMENTS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "config": _config_args,
    "doctor": _doctor_args,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dtk-mcp", description="datatoolkit MCP server")
    commands = parser.add_subparsers(dest="command")
    for name, (help_text, _) in SUBCOMMANDS.items():
        sub = commands.add_parser(name, help=help_text)
        if name in ARGUMENTS:
            ARGUMENTS[name](sub)
    args = parser.parse_args(argv)
    SUBCOMMANDS[args.command or "serve"][1](args)


if __name__ == "__main__":
    main()
