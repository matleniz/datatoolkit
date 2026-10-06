"""Agent tests share the process-wide step tracker: start each test empty."""

import pytest

from dtk_engine.agent.digest import TRACKER


@pytest.fixture(autouse=True)
def _fresh_step_tracker():
    TRACKER.clear()
    yield
    TRACKER.clear()
