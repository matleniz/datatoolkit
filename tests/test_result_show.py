"""Result.show() never blocks outside a notebook."""

import os
import subprocess
import sys
import types

import plotly.io
import pytest

from dtk_engine import Result

SCRIPT = """
from dtk_engine import Result
r = Result(metrics={"rows": 3})
r.add_figure("f", {"data": [{"type": "bar", "x": [1], "y": [2]}], "layout": {}})
r.show()
print("returned")
"""


def test_show_returns_without_kernel():
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        stdin=subprocess.DEVNULL,
        env={k: v for k, v in os.environ.items() if k != "PLOTLY_RENDERER"},
    )
    assert out.returncode == 0, out.stderr
    assert "rows: 3" in out.stdout
    assert "1 figure not displayed" in out.stdout
    assert out.stdout.strip().endswith("returned")


# In-process unit tests: plotly's display is stubbed, nothing is opened.
def _result():
    r = Result(metrics={"rows": 3})
    r.add_figure("f", {"data": [{"type": "bar", "x": [1], "y": [2]}], "layout": {}})
    return r


def _stub_ipython(monkeypatch, shell):
    mod = types.ModuleType("IPython")
    mod.get_ipython = lambda: shell
    monkeypatch.setitem(sys.modules, "IPython", mod)


@pytest.fixture
def shown(monkeypatch):
    import plotly.basedatatypes

    calls = []
    monkeypatch.setattr(
        plotly.basedatatypes.BaseFigure, "show", lambda self, *a, **k: calls.append(1)
    )
    monkeypatch.delenv("PLOTLY_RENDERER", raising=False)
    monkeypatch.setattr(plotly.io.renderers, "default", "browser")
    _stub_ipython(monkeypatch, None)
    return calls


def test_env_renderer_shows_figures(shown, monkeypatch, capsys):
    monkeypatch.setenv("PLOTLY_RENDERER", "json")
    _result().show()
    assert shown == [1]
    assert "not displayed" not in capsys.readouterr().out


def test_configured_default_renderer_shows_figures(shown, monkeypatch):
    monkeypatch.setattr(plotly.io.renderers, "default", "json")
    _result().show()
    assert shown == [1]


def test_kernel_shows_figures(shown, monkeypatch):
    _stub_ipython(monkeypatch, types.SimpleNamespace(kernel=object()))
    _result().show()
    assert shown == [1]


def test_ipython_terminal_prints_one_line(shown, monkeypatch, capsys):
    _stub_ipython(monkeypatch, types.SimpleNamespace(kernel=None))
    _result().show()
    assert shown == []
    assert "1 figure not displayed" in capsys.readouterr().out
