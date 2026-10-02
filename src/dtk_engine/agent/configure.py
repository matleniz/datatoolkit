"""``dtk-mcp config <pack>``: print or write a pack's config wired to ``dtk`` only.

Default wiring is stdio (``<this python> -m dtk_engine.agent.cli``): no token
in any file, and the config survives ``dtk-api`` restarts (the stdio server
finds the running engine through the runtime file on each call). ``--http``
points at the running ``/mcp`` with the per-run token instead.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import textwrap
from pathlib import Path

from dtk_engine.agent.packs import Pack, ServerRef
from dtk_engine.agent.policy import DEFAULT_ROWS, MAX_ROWS
from dtk_engine.ui_bridge import read_runtime

NO_ENGINE = "no running engine (start dtk-api)"
HTTP_NOTE = "--http: the config holds a token valid for this engine run only"


class ConfigError(Exception):
    """A user-facing refusal; ``main`` prints it and exits 2."""


def stdio_server() -> ServerRef:
    """This interpreter (the env with the ``agent`` extra), plus ``DTK_HOME`` if set."""
    server: ServerRef = {"command": [sys.executable, "-m", "dtk_engine.agent.cli"]}
    home = os.environ.get("DTK_HOME")
    if home:
        server["env"] = {"DTK_HOME": str(Path(home).expanduser().resolve())}
    return server


def http_server() -> ServerRef:
    runtime = read_runtime()
    if runtime is None or not runtime.get("url"):
        raise ConfigError(NO_ENGINE)
    url = runtime["url"].rstrip("/") + "/mcp/"
    return {"url": url, "headers": {"Authorization": f"Bearer {runtime.get('token', '')}"}}


def _dump(content: dict) -> str:
    return json.dumps(content, indent=2) + "\n"


def write_files(files: dict[str, dict], directory: Path, *, force: bool, secret: bool) -> list[Path]:
    """Write every file under ``directory``; refuse (before writing any) to overwrite."""
    targets = {directory / rel: content for rel, content in files.items()}
    existing = [str(p) for p in targets if p.exists()]
    if existing and not force:
        raise ConfigError(f"refusing to overwrite {', '.join(existing)} (use --force)")
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(content))
        if secret:  # holds the per-run token
            path.chmod(0o600)
    return list(targets)


def render(pack: Pack, server: ServerRef, directory: Path | None) -> str:
    """The text ``config`` prints: files (when not written), run line, policy notes."""
    lines: list[str] = []
    if "url" in server:
        lines.append(f"# {HTTP_NOTE}")
    if directory is None:
        for rel, content in pack.files(server).items():
            lines += [f"# {rel}", _dump(content).rstrip("\n"), ""]
        command = pack.launch_command(Path("."))
        lines += ["# run (from the directory holding these files):", shlex.join(command)]
    else:
        command = pack.launch_command(directory)
        lines += ["# run:", f"cd {shlex.quote(str(directory))} && {shlex.join(command)}"]
    notes = [
        f"tool policy: {pack.tool_policy() or 'n/a'}",
        f"auth: {pack.auth}",
        (
            f"cost / data: {pack.cost}; rows and profiles the agent reads go to that "
            f"provider ({DEFAULT_ROWS} rows per call by default, {MAX_ROWS} max)"
        ),
    ]
    lines.append("")
    lines += [
        textwrap.fill(
            note, 88, initial_indent="# ", subsequent_indent="#   ", break_on_hyphens=False
        )
        for note in notes
    ]
    return "\n".join(lines) + "\n"


def run(pack: Pack, *, write: str | None, force: bool, http: bool) -> str:
    server = http_server() if http else stdio_server()
    if write is None:
        return render(pack, server, None)
    directory = Path(write).expanduser().resolve()
    written = write_files(pack.files(server), directory, force=force, secret=http)
    header = "".join(f"# wrote {path}\n" for path in written)
    return header + render(pack, server, directory)
