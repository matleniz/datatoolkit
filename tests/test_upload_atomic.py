"""Uploads: atomic write, and the 'already exists' shortcut never trusts a partial file."""

from __future__ import annotations

import os
import threading

import pytest

from dtk_engine import http

DATA = b"a,b\n" + b"1,2\n" * 5000


def _files(root):
    return [p for p in root.rglob("*") if p.is_file()]


def test_normal_put_unchanged(tmp_path):
    p = http._save_upload("t.csv", DATA, tmp_path)
    assert p.read_bytes() == DATA
    assert p.parent.parent == tmp_path.resolve()
    assert http._save_upload("t.csv", DATA, tmp_path) == p
    assert _files(tmp_path) == [p]


def test_truncated_existing_file_is_repaired(tmp_path):
    p = http._save_upload("t.csv", DATA, tmp_path)
    p.write_bytes(DATA[: len(DATA) // 2])
    assert http._save_upload("t.csv", DATA, tmp_path) == p
    assert p.read_bytes() == DATA


def test_same_size_corrupt_file_is_repaired(tmp_path):
    p = http._save_upload("t.csv", DATA, tmp_path)
    p.write_bytes(b"x" * len(DATA))
    http._save_upload("t.csv", DATA, tmp_path)
    assert p.read_bytes() == DATA


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        http._save_upload("t.csv", DATA, tmp_path)
    assert _files(tmp_path) == []


def test_interrupted_write_keeps_previous_complete_file(tmp_path, monkeypatch):
    p = http._save_upload("t.csv", DATA, tmp_path)
    p.write_bytes(b"partial")
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        http._save_upload("t.csv", DATA, tmp_path)
    assert _files(tmp_path) == [p]


def test_concurrent_puts_end_with_complete_file(tmp_path):
    barrier = threading.Barrier(8)
    paths, errors = [], []

    def run():
        try:
            barrier.wait()
            paths.append(http._save_upload("t.csv", DATA, tmp_path))
        except OSError as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(set(paths)) == 1
    assert paths[0].read_bytes() == DATA
    assert _files(tmp_path) == [paths[0]]


def test_file_mode_follows_umask(tmp_path):
    old = os.umask(0o022)
    try:
        p = http._save_upload("t.csv", DATA, tmp_path)
    finally:
        os.umask(old)
    assert p.stat().st_mode & 0o777 == 0o644
