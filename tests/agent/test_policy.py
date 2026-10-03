import json

import pytest

from dtk_engine import contract
from dtk_engine.agent import policy
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("DTK_HOME", str(h))
    monkeypatch.delenv("DTK_UPLOAD_DIR", raising=False)
    monkeypatch.delenv("DTK_AGENT_LOG", raising=False)
    monkeypatch.delenv("DTK_AGENT_MAX_CHARS", raising=False)
    return h


def test_policy_error_is_params_error():
    assert issubclass(policy.PolicyError, KeyParamsError)


def test_path_under_home_passes(home):
    (home / "a.csv").write_text("x\n1\n")
    policy.check_args({"source": {"kind": "csv", "path": str(home / "a.csv")}})


def test_upload_dir_allowed(home, tmp_path, monkeypatch):
    up = tmp_path / "up"
    up.mkdir()
    monkeypatch.setenv("DTK_UPLOAD_DIR", str(up))
    policy.check_args({"path": str(up / "f.csv")})


@pytest.mark.parametrize(
    "make",
    [
        lambda home, tmp: str(home / ".." / "x.csv"),
        lambda home, tmp: "/etc/passwd",
        lambda home, tmp: "relative/outside.csv",
        lambda home, tmp: "C:\\Users\\x\\f.csv",
    ],
)
def test_outside_paths_refused(home, tmp_path, monkeypatch, make):
    monkeypatch.chdir(tmp_path)
    path = make(home, tmp_path)
    with pytest.raises(policy.PolicyError, match="path not allowed for the agent"):
        policy.check_args({"source": {"kind": "csv", "path": path}})


def test_symlink_escape_refused(home, tmp_path):
    outside = tmp_path / "secret.csv"
    outside.write_text("x\n1\n")
    link = home / "link.csv"
    link.symlink_to(outside)
    with pytest.raises(policy.PolicyError):
        policy.check_args({"path": str(link)})


def test_nested_params_source_checked(home):
    args = {
        "key": "train_test_check",
        "params": {
            "train": {"kind": "csv", "path": str(home / "ok.csv")},
            "test": {"kind": "csv", "path": "/etc/hosts"},
        },
    }
    with pytest.raises(policy.PolicyError, match="/etc/hosts"):
        policy.check_args(args)


def test_sql_refused_and_dataset_passes(home):
    with pytest.raises(policy.PolicyError, match="sql sources are not available"):
        policy.check_args({"source": {"kind": "sql", "query": "select 1"}})
    policy.check_args({"source": {"kind": "dataset", "workspace": "w"}})


def test_workspace_referenced_path_passes(home):
    contract.save_workspace(
        {
            "name": "w",
            "datasets": {
                "train": {
                    "x": {"kind": "csv", "path": TRAIN_CSV},
                    "target_column": "Survived",
                },
                "test": {"x": {"kind": "csv", "path": TEST_CSV}},
            },
        }
    )
    assert TRAIN_CSV in policy.workspace_paths()
    policy.check_args({"source": {"kind": "csv", "path": TRAIN_CSV}})
    with pytest.raises(policy.PolicyError):
        policy.check_args({"path": "/etc/hosts"})


def test_row_limit():
    assert policy.row_limit(None) == 50
    assert policy.row_limit(1) == policy.row_limit(1) and policy.row_limit(500) == 500
    for bad in (0, 501, -3):
        with pytest.raises(policy.PolicyError, match="limit must be between 1 and 500"):
            policy.row_limit(bad)


def test_compact_result_drops_plotly(home):
    res = contract.run_key("dataset_overview", {})
    assert any("plotly" in f for f in res["figures"])
    slim = policy.compact_result(res)
    assert all(set(f) == {"title", "group", "main"} for f in slim["figures"])
    assert len(slim["figures"]) == len(res["figures"])
    assert policy.compact_result(res, include_figures=True) is res


def test_frame_small_and_truncated(home, monkeypatch):
    assert policy.frame({"rows": [1]}, identity="me") == {
        "identity": "me",
        "data": {"rows": [1]},
    }
    monkeypatch.setenv("DTK_AGENT_MAX_CHARS", "500")
    big = {"rows": [{"a": "x" * 20, "i": i} for i in range(200)]}
    out = policy.frame(big)
    assert out["truncated"] is True and "kept" in out["note"]
    assert 0 < len(out["data"]["rows"]) < 200
    assert len(json.dumps(out["data"])) <= 500
    assert len(big["rows"]) == 200
    other = policy.frame({"blob": "y" * 2000})
    assert other["truncated"] and len(other["data"]["preview"]) == 500


def test_check_command():
    policy.check_command("propose_steps")
    with pytest.raises(policy.PolicyError):
        policy.check_command("export")


def test_audit_summary_has_no_values_or_tokens(home, monkeypatch):
    log = policy.AuditLog()
    entry = log.record(
        "run_key",
        {
            "key": "k",
            "token": "SECRET-TOKEN",
            "rows": [{"name": "Alice"}],
            "source": {"kind": "csv", "path": "/x.csv", "data": "Alice"},
            "params": {"note": "n" * 300, "api_key": "KKK", "cols": ["a"]},
        },
        status="ok",
        identity="me",
        duration_ms=1.5,
    )
    dumped = json.dumps(entry)
    for leak in ("Alice", "SECRET-TOKEN", "KKK"):
        assert leak not in dumped
    assert entry["args"]["source"] == {"kind": "csv", "path": "/x.csv"}
    assert entry["args"]["rows"] == "<list 1>"
    assert len(entry["args"]["params"]["note"]) == 120
    assert log.entries() == [entry]


def test_audit_ring_and_file_sink(home, monkeypatch):
    log = policy.AuditLog()
    for i in range(510):
        log.record("t", {"i": i}, status="ok")
    assert len(log.entries()) == 500
    monkeypatch.setenv("DTK_AGENT_LOG", "1")
    log.record("t", {}, status="ok")
    lines = (home / "agent" / "log.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["tool"] == "t"
    # An unwritable sink never fails the call.
    (home / "agent" / "log.jsonl").unlink()
    (home / "agent" / "log.jsonl").mkdir()
    log.record("t", {}, status="ok")


@pytest.mark.parametrize(
    "rel",
    ["agent/runtime.json", "agent/log.jsonl", "agent/terminal/claude/settings.json"],
)
def test_control_files_refused(home, rel):
    target = home / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"token": "secret"}')
    with pytest.raises(policy.PolicyError, match="control files"):
        policy.check_args({"source": {"kind": "json", "path": str(target)}})


def test_control_files_refused_through_symlink(home):
    (home / "agent").mkdir()
    (home / "agent" / "runtime.json").write_text('{"token": "secret"}')
    uploads = home / "uploads"
    uploads.mkdir()
    link = uploads / "innocent.json"
    link.symlink_to(home / "agent" / "runtime.json")
    with pytest.raises(policy.PolicyError, match="control files"):
        policy.check_args({"path": str(link)})
    (uploads / "data.csv").write_text("x\n1\n")
    policy.check_args({"path": str(uploads / "data.csv")})


def test_control_files_refused_even_if_workspace_references_them(home):
    runtime = home / "agent" / "runtime.json"
    runtime.parent.mkdir()
    runtime.write_text('{"token": "secret"}')
    contract.save_workspace(
        {
            "name": "leak",
            "datasets": {"train": {"x": {"kind": "json", "path": str(runtime)}}},
        }
    )
    assert str(runtime) in policy.workspace_paths()
    with pytest.raises(policy.PolicyError):
        policy.check_args({"path": str(runtime)})
