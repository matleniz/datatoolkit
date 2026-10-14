"""``dtk-api`` / ``dtk-mcp`` without their extra: one line, exit 1, no traceback."""

import builtins

import pytest

from dtk_engine import _launch

REPO = "git+https://github.com/matleniz/datatoolkit"


@pytest.fixture
def without(monkeypatch):
    def hide(*names):
        real = builtins.__import__

        def fake(name, *args, **kwargs):
            if name.split(".")[0] in names:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name.split(".")[0])
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake)
        for mod in ("dtk_engine.http", "dtk_engine.agent.cli"):
            monkeypatch.delitem(__import__("sys").modules, mod, raising=False)

    return hide


@pytest.mark.parametrize(
    ("entry", "script", "extra", "missing"),
    [("api_main", "dtk-api", "api", "fastapi"), ("mcp_main", "dtk-mcp", "agent", "anyio")],
)
def test_missing_extra_message(without, entry, script, extra, missing):
    without(missing)
    with pytest.raises(SystemExit) as exc:
        getattr(_launch, entry)()
    assert exc.value.code == f'{script} needs the "{extra}" extra: uv pip install "dtk-engine[{extra}] @ {REPO}"'


def test_unrelated_missing_module_is_not_masked(monkeypatch):
    def boom(name):
        raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

    monkeypatch.setattr(_launch.importlib, "import_module", boom)
    with pytest.raises(ModuleNotFoundError):
        _launch.api_main()


def test_entry_runs_when_extra_present(monkeypatch):
    called = []
    monkeypatch.setattr("dtk_engine.http.main", lambda: called.append(1))
    _launch.api_main()
    assert called == [1]
