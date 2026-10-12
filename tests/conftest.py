"""Shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_reconnect_grace(monkeypatch):
    """The UI bridge fails at once by default in tests; grace tests opt in explicitly."""
    monkeypatch.setenv("DTK_UI_RECONNECT_GRACE", "0")
