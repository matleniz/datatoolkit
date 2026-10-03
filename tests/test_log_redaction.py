"""The UI token (``?token=``) never reaches uvicorn's console lines."""

from __future__ import annotations

import logging
import logging.config

import pytest

from dtk_engine import http
from dtk_engine.ui_bridge import RedactTokenFilter, redact_token

TOKEN = "s3cr3t-T0ken_value"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"/api/ui/events?token={TOKEN}", "/api/ui/events?token=***"),
        (
            f"/api/ui/events?session=x&token={TOKEN}&y=1",
            "/api/ui/events?session=x&token=***&y=1",
        ),
        (f'"WebSocket /api/ui/terminal?pack=c&TOKEN={TOKEN}"', '"WebSocket /api/ui/terminal?pack=c&TOKEN=***"'),
        ("/api/keys?mytoken=1", "/api/keys?mytoken=1"),
    ],
)
def test_redact_token(text, expected):
    assert redact_token(text) == expected


@pytest.fixture
def uvicorn_logging():
    """Apply ``uvicorn_log_config`` then restore the uvicorn loggers."""
    names = ("uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {
        n: (logging.getLogger(n).handlers[:], logging.getLogger(n).level,
            logging.getLogger(n).propagate)
        for n in names
    }
    yield lambda: logging.config.dictConfig(http.uvicorn_log_config())
    for n, (handlers, level, propagate) in saved.items():
        logger = logging.getLogger(n)
        logger.handlers[:] = handlers
        logger.setLevel(level)
        logger.propagate = propagate


def test_uvicorn_lines_are_redacted(uvicorn_logging, capsys):
    uvicorn_logging()  # after capsys: handlers bind the captured streams
    # Same shapes as uvicorn's httptools access line and websockets accept line.
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5000", "GET", f"/api/ui/events?session=x&token={TOKEN}", "1.1", 200,
    )
    logging.getLogger("uvicorn.error").info(
        '%s - "WebSocket %s" [accepted]',
        "127.0.0.1:5001", f"/api/ui/terminal?session=x&pack=claude&token={TOKEN}",
    )
    out = capsys.readouterr()
    logs = out.out + out.err
    assert TOKEN not in logs
    assert "/api/ui/events?session=x&token=*** HTTP/1.1" in logs
    assert "WebSocket /api/ui/terminal?session=x&pack=claude&token=***" in logs


def test_filter_keeps_access_args_shape():
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("c", "GET", f"/x?token={TOKEN}", "1.1", 200), None,
    )
    assert RedactTokenFilter().filter(record)
    assert record.args == ("c", "GET", "/x?token=***", "1.1", 200)


def test_main_passes_redacting_log_config(monkeypatch, tmp_path):
    import uvicorn

    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    http.main([])
    config = seen["log_config"]
    assert all("redact_token" in h["filters"] for h in config["handlers"].values())
