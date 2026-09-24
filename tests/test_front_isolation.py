import ast
from pathlib import Path

import dtk_streamlit

FRONT = Path(dtk_streamlit.__file__).parent


def _imports_engine(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "dtk_engine" for a in node.names):
                return True
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] == "dtk_engine"
        ):
            return True
    return False


def test_front_never_imports_engine_except_client():
    modules = [p for p in FRONT.rglob("*.py") if p.name != "client.py"]
    assert modules
    offenders = [p.name for p in modules if _imports_engine(ast.parse(p.read_text()))]
    assert offenders == []
