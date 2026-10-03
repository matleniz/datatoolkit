"""Model discovery results and the CLI-side listers (``opencode models``).

``ModelList`` is what a pack's ``models()`` answers: a closed list, or free
text when the provider cannot say (``models_error`` tells why when a lookup
failed). Discovery never raises: a failure is a free-text list with an error,
so a pack stays usable. The hub caches one answer per lister per run.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from dataclasses import dataclass, field

DISCOVERY_TIMEOUT = 30.0


@dataclass(frozen=True)
class ModelList:
    """``models``: ``[{id, label?, description?, resolved?}]``; ``free_text``: any id is accepted."""

    models: list[dict] = field(default_factory=list)
    free_text: bool = False
    error: str | None = None

    def ids(self) -> list[str]:
        return [m["id"] for m in self.models]

    def to_json(self) -> dict:
        out: dict = {"models": self.models, "model_free_text": self.free_text}
        if self.error:
            out["models_error"] = self.error
        return out


def failed(error: str) -> ModelList:
    return ModelList([], True, error)


FREE_TEXT = ModelList([], True)


async def opencode_models() -> ModelList:
    """``opencode models``: one ``provider/model`` per stdout line."""
    cli = shutil.which("opencode")
    if cli is None:
        return failed("opencode not found on PATH")
    try:
        proc = await asyncio.create_subprocess_exec(
            cli, "models", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), DISCOVERY_TIMEOUT)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return failed(f"opencode models timed out after {DISCOVERY_TIMEOUT:.0f} s")
    except OSError as exc:
        return failed(f"opencode models failed: {exc}")
    ids = [line.strip() for line in out.decode(errors="replace").splitlines() if line.strip()]
    if proc.returncode != 0 or not ids:
        return failed(f"opencode models gave no list (exit {proc.returncode})")
    return ModelList([{"id": i, "label": i} for i in ids])


async def gemini_models() -> ModelList:
    """gemini has no list command: free text (``-m``)."""
    return FREE_TEXT
