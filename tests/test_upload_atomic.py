"""Uploads: atomic write, and the 'already exists' shortcut never trusts a partial file."""

from __future__ import annotations

import os
import threading

import pytest

from dtk_engine import http

DATA = b"a,b\n" + b"1,2\n" * 5000


def _save(name, data, root):
    sink = http._UploadSink(root)
    for i in range(0, len(data), 4096):
        sink.write(data[i : i + 4096])
    return sink.commit(name)


def _files(root):
    return [p for p in root.rglob("*") if p.is_file()]


def test_normal_put_unchanged(tmp_path):
    p = _save("t.csv", DATA, tmp_path)
    assert p.read_bytes() == DATA
    assert p.parent.parent == tmp_path.resolve()
    assert _save("t.csv", DATA, tmp_path) == p
    assert _files(tmp_path) == [p]


def test_truncated_existing_file_is_repaired(tmp_path):
    p = _save("t.csv", DATA, tmp_path)
    p.write_bytes(DATA[: len(DATA) // 2])
    assert _save("t.csv", DATA, tmp_path) == p
    assert p.read_bytes() == DATA


def test_same_size_corrupt_file_is_repaired(tmp_path):
    p = _save("t.csv", DATA, tmp_path)
    p.write_bytes(b"x" * len(DATA))
    _save("t.csv", DATA, tmp_path)
    assert p.read_bytes() == DATA


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    def boom(src, dst):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        _save("t.csv", DATA, tmp_path)
    assert _files(tmp_path) == []


def test_interrupted_write_keeps_previous_complete_file(tmp_path, monkeypatch):
    p = _save("t.csv", DATA, tmp_path)
    p.write_bytes(b"partial")
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        _save("t.csv", DATA, tmp_path)
    assert _files(tmp_path) == [p]


def test_concurrent_puts_end_with_complete_file(tmp_path):
    barrier = threading.Barrier(8)
    paths, errors = [], []

    def run():
        try:
            barrier.wait()
            paths.append(_save("t.csv", DATA, tmp_path))
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
        p = _save("t.csv", DATA, tmp_path)
    finally:
        os.umask(old)
    assert p.stat().st_mode & 0o777 == 0o644


def test_identical_reupload_is_reused_not_rewritten(tmp_path):
    p = _save("t.csv", DATA, tmp_path)
    inode = p.stat().st_ino
    assert _save("t.csv", DATA, tmp_path) == p
    assert p.stat().st_ino == inode
    assert _files(tmp_path) == [p]


def test_differing_content_is_rewritten(tmp_path):
    p = _save("t.csv", DATA, tmp_path)
    p.write_bytes(DATA[:-1] + b"X")  # same size, last byte differs
    assert _save("t.csv", DATA, tmp_path) == p
    assert p.read_bytes() == DATA


def test_abort_removes_the_temp_file(tmp_path):
    sink = http._UploadSink(tmp_path)
    sink.write(DATA)
    assert len(_files(tmp_path)) == 1
    sink.abort()
    assert _files(tmp_path) == []
