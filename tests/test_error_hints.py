"""Error messages name the valid values (datatoolkit-issues #231, #230)."""

import pytest

from dtk_engine import contract
from dtk_engine.errors import KeyParamsError, UnknownKeyError


def test_unknown_key_lists_available():
    with pytest.raises(UnknownKeyError) as e:
        contract.run_key("nope", {})
    msg = str(e.value)
    assert msg.startswith("unknown key 'nope'; available: ")
    assert "missing_values" in msg


def test_unknown_param_lists_valid_params():
    with pytest.raises(KeyParamsError) as e:
        contract.run_key("missing_values", {"max_pct": 1})
    msg = str(e.value)
    assert msg.startswith("max_pct: Extra inputs are not permitted")
    assert "valid params:" in msg and "columns" in msg
    assert e.value.details[0]["loc"] == ["max_pct"]


def test_unknown_target_lists_columns_and_close_match():
    with pytest.raises(KeyParamsError) as e:
        contract.run_key("target_analysis", {"target": "Survive"})
    msg = str(e.value)
    assert "target 'Survive' not in the frame" in msg
    assert "did you mean 'Survived'?" in msg and "columns:" in msg


def test_columns_list_is_capped():
    import pandas as pd

    from dtk_engine.keys.target_analysis import target_result

    df = pd.DataFrame({f"c{i}": [1, 2] for i in range(30)})
    with pytest.raises(KeyParamsError, match=r"\.\.\. \(30 total\)"):
        target_result(df, "zzzz")


def test_policy_error_says_how_to_fix(tmp_path, monkeypatch):
    from dtk_engine.agent.policy import PolicyError, check_args

    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    csv = tmp_path / "d.csv"
    csv.write_text("a\n1\n")
    with pytest.raises(PolicyError) as e:
        check_args({"source": {"kind": "csv", "path": str(csv)}})
    assert "$DTK_HOME/uploads" in str(e.value)
    assert "/api/uploads/" in str(e.value)
