"""Layer contract: http -> contract -> api | pipeline -> workspace -> keys -> ops -> sources.

A module may import its own layer or a lower one, never a higher one. Shared
foundation modules (errors, params, result, registry, cache, ...) are not ranked.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "dtk_engine"
PACKAGE = "dtk_engine"

# Higher rank = higher layer; api and pipeline are siblings.
RANK = {
    "http": 5,
    "contract": 4,
    "api": 3,
    "pipeline": 3,
    "workspace": 2,
    "keys": 1,
    "ops": 0,
    "sources": -1,
}


def _layer(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] == PACKAGE and len(parts) > 1 and parts[1] in RANK:
        return parts[1]
    return None


def _imports(path: Path, root: Path) -> set[str]:
    rel = path.relative_to(root).with_suffix("").parts
    package = [PACKAGE, *rel[:-1]]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            if module == PACKAGE:  # `from dtk_engine import api` names a submodule
                found.update(f"{module}.{alias.name}" for alias in node.names)
            else:
                found.add(module)
    return found


def layer_violations(root: Path = ROOT) -> list[str]:
    out = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("").parts
        src = _layer(".".join([PACKAGE, *rel]))
        if src is None:
            continue
        for target in sorted(_imports(path, root)):
            dst = _layer(target)
            if dst is not None and RANK[dst] > RANK[src]:
                out.append(f"{path.relative_to(root)}: {src} -> {dst} ({target})")
    return out


def test_layer_contract():
    assert layer_violations() == []


def test_detects_forbidden_edge(tmp_path):
    pkg = tmp_path / "ops"
    pkg.mkdir()
    (pkg / "bad.py").write_text("from dtk_engine.keys import chart\n")
    assert layer_violations(tmp_path) == ["ops/bad.py: ops -> keys (dtk_engine.keys)"]


def test_ui_bridge_is_isolated_and_only_http_and_agent_import_it():
    bridge = ROOT / "ui_bridge.py"
    imported = {m for m in _imports(bridge, ROOT) if m.startswith(PACKAGE)}
    assert imported == set(), "ui_bridge must stay pure asyncio + stdlib"
    importers = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(ROOT.rglob("*.py"))
        if path != bridge and f"{PACKAGE}.ui_bridge" in _imports(path, ROOT)
    ]
    assert importers == [
        "agent/chat.py", "agent/configure.py", "agent/doctor.py", "agent/packs/chat_packs.py",
        "agent/ports.py", "agent/server.py", "http.py",
    ]


_AGENT_ALLOWED = {
    f"{PACKAGE}.contract",
    f"{PACKAGE}.ui_bridge",
    f"{PACKAGE}.errors",
}


def test_agent_layer_imports():
    agent = ROOT / "agent"
    for path in sorted(agent.rglob("*.py")):
        for target in _imports(path, ROOT):
            if not target.startswith(PACKAGE) or target == PACKAGE:
                continue
            ok = target in _AGENT_ALLOWED or target.startswith(f"{PACKAGE}.agent")
            assert ok, f"{path.relative_to(ROOT)} imports {target}"
    importers = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(ROOT.rglob("*.py"))
        if agent not in path.parents
        and any(m.startswith(f"{PACKAGE}.agent") for m in _imports(path, ROOT))
    ]
    assert importers == ["http.py"]
