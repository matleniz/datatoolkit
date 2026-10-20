"""X-Dtk-Write-Seq: an older PUT never overwrites a newer one (datatoolkit-issues#263)."""

from __future__ import annotations

import random
import threading
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.http import create_app
from dtk_engine.workspace import store as store_mod

H = "X-Dtk-Write-Seq"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(tmp_path / "uploads"))
    return tmp_path


def _workspace(name, **extra):
    train = {"x": {"kind": "csv", "path": TRAIN_CSV}, "target_column": "Survived"}
    test = {"x": {"kind": "csv", "path": str(Path(TRAIN_CSV).with_name("test.csv"))}}
    return {"name": name, "datasets": {"train": train, "test": test}, **extra}


@pytest.fixture
def client(home):
    store_mod._write_seq.clear()
    return TestClient(create_app())


def _put(client, seq=None, name="w", note="x"):
    ws = _workspace(name, notes={"workspace": note})
    headers = {} if seq is None else {H: str(seq)}
    return client.put(f"/api/workspaces/{name}", json=ws, headers=headers)


def _note(client, name="w"):
    return client.get(f"/api/workspaces/{name}").json()["notes"]["workspace"]


def test_older_put_refused_409(client):
    assert _put(client, 6, note="six").status_code == 200
    r = _put(client, 5, note="five")
    assert r.status_code == 409
    body = r.json()
    assert body["type"] == "StaleWriteError"
    assert body["detail"] == {"rule": "stale_write", "last_seq": 6}
    assert "message" in body
    assert _note(client) == "six"


def test_equal_and_ascending_accepted(client):
    for seq, note in [(1, "a"), (1, "b"), (2, "c"), (10, "d")]:
        assert _put(client, seq, note=note).status_code == 200
        assert _note(client) == note


def test_no_header_unguarded_and_leaves_seq(client):
    assert _put(client, 9, note="nine").status_code == 200
    assert _put(client, None, note="free").status_code == 200
    assert _note(client) == "free"
    assert _put(client, 8, note="eight").status_code == 409
    assert _note(client) == "free"


def test_no_header_never_creates_seq(client):
    assert _put(client, None, note="a").status_code == 200
    assert _put(client, 3, note="b").status_code == 200


@pytest.mark.parametrize("value", ["abc", "-1", "", "1.5", "+5", " 5", "5_0", str(2**63), "٥", "1" * 5000, "1" * 20])
def test_invalid_header_422(client, value):
    r = client.put(
        "/api/workspaces/w", json=_workspace("w"), headers={H: value.encode("utf-8")}
    )
    assert r.status_code == 422
    assert r.json()["type"] == "KeyParamsError"
    assert H in r.json()["message"]
    assert client.get("/api/workspaces/w").status_code == 404


def test_max_value_accepted(client):
    assert _put(client, 2**63 - 1).status_code == 200


def test_repeated_header_422(client):
    r = client.put(
        "/api/workspaces/w", json=_workspace("w"), headers=[(H, "1"), (H, "2")]
    )
    assert r.status_code == 422
    assert H in r.json()["message"]


def test_delete_resets(client):
    assert _put(client, 9, note="a").status_code == 200
    assert client.delete("/api/workspaces/w").status_code == 204
    assert _put(client, 1, note="b").status_code == 200


def test_rename_resets_both_names(client):
    assert _put(client, 9, name="a").status_code == 200
    assert _put(client, 7, name="b").status_code == 200
    client.delete("/api/workspaces/b")
    assert _put(client, 9, name="b").status_code == 200  # b has seq 9 again
    client.delete("/api/workspaces/b")
    r = client.post("/api/workspaces/a/rename", json={"new_name": "b"})
    assert r.status_code == 200, r.text
    assert _put(client, 1, name="b").status_code == 200
    assert _put(client, 1, name="a").status_code == 200


def test_duplicate_leaves_source_seq(client):
    assert _put(client, 9, name="a").status_code == 200
    r = client.post("/api/workspaces/a/duplicate", json={"new_name": "c"})
    assert r.status_code == 200, r.text
    assert _put(client, 8, name="a").status_code == 409
    assert _put(client, 1, name="c").status_code == 200


def test_lru_bound(client, monkeypatch):
    monkeypatch.setattr(store_mod, "_WRITE_SEQ_NAMES", 3)
    for i in range(4):
        assert _put(client, 9, name=f"w{i}").status_code == 200
    assert len(store_mod._write_seq) == 3
    assert _put(client, 1, name="w0").status_code == 200  # evicted: forgotten
    assert _put(client, 1, name="w3").status_code == 409  # kept


def test_failed_validation_does_not_advance_seq(client):
    assert _put(client, 1).status_code == 200
    bad = client.put("/api/workspaces/w", json={"datasets": 3}, headers={H: "50"})
    assert bad.status_code == 422
    assert _put(client, 2).status_code == 200


def test_concurrent_puts_keep_highest(home):
    app = create_app()
    for i in range(40):
        store_mod._write_seq.clear()
        name = f"c{i}"
        seqs = [10, 11]
        random.shuffle(seqs)
        codes: dict[int, int] = {}

        def go(seq, name=name, codes=codes):
            c = TestClient(app)
            ws = _workspace(name, notes={"workspace": f"s{seq}"})
            codes[seq] = c.put(
                f"/api/workspaces/{name}", json=ws, headers={H: str(seq)}
            ).status_code

        threads = [threading.Thread(target=go, args=(s,)) for s in seqs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert codes[11] == 200
        assert codes[10] in (200, 409)
        got = TestClient(app).get(f"/api/workspaces/{name}").json()
        assert got["notes"]["workspace"] == "s11"


def test_cors_preflight_allows_header(client):
    r = client.options(
        "/api/workspaces/w",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": f"content-type,{H.lower()}",
        },
    )
    assert r.status_code == 200
    assert H.lower() in r.headers["access-control-allow-headers"].lower()


def test_leading_zeros_accepted(client):
    assert _put(client, "0" * 5000 + "7").status_code == 200
    assert _put(client, 6).status_code == 409


@pytest.mark.parametrize(
    ("guarded", "code", "winner"), [(True, 409, "six"), (False, 200, "five")]
)
def test_late_older_write_deterministic(home, monkeypatch, guarded, code, winner):
    """The older write (5) takes the lock only after the newer one (6) is done:
    with the header it is refused; without, the older content wins (the bug)."""
    from dtk_engine import contract
    from dtk_engine.errors import StaleWriteError

    store_mod._write_seq.clear()
    newer_done = threading.Event()
    real = store_mod.JsonWorkspaceStore._exclusive

    def exclusive(self):
        if threading.current_thread().name == "late":
            assert newer_done.wait(10)
        return real(self)

    monkeypatch.setattr(store_mod.JsonWorkspaceStore, "_exclusive", exclusive)
    out = {}

    def run(label, seq, note):
        ws = _workspace("w", notes={"workspace": note})
        try:
            contract.save_workspace(ws, seq if guarded else None)
            out[label] = 200
        except StaleWriteError:
            out[label] = 409

    late = threading.Thread(target=run, args=("late", 5, "five"), name="late")
    late.start()
    run("early", 6, "six")
    newer_done.set()
    late.join(10)
    assert out == {"early": 200, "late": code}
    assert contract.get_workspace("w")["notes"]["workspace"] == winner


def test_no_header_never_creates_or_advances_seq_table(client):
    assert _put(client, 9, note="nine").status_code == 200
    assert _put(client, None, note="free").status_code == 200
    assert _put(client, 8, note="eight").status_code == 409
    assert _put(client, 9, note="again").status_code == 200
    assert _put(client, None, name="other").status_code == 200
    assert not any(k[1] == "other" for k in store_mod._write_seq)
