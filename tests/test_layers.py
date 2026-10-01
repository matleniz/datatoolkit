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
