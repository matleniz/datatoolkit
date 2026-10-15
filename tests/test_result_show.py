"""Result.show() never blocks outside a notebook."""

import os
import subprocess
import sys

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
