import pytest
from pydantic import ValidationError

from dtk_engine.contract import get_workspace, save_workspace
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.errors import KeyParamsError
from dtk_engine.workspace.models import VARIABLE_STATS, VariableSpec, Workspace


def _base_ws(name="w_vars", **kw):
    return {
        "name": name,
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": TRAIN_CSV},
                "target_column": "Survived",
            }
        },
        **kw,
    }


@pytest.mark.parametrize("stat", VARIABLE_STATS)
def test_variable_spec_valid_stats(stat):
    spec = VariableSpec(name="v_stat", stat=stat, column="Age")
    assert spec.name == "v_stat"
    assert spec.stat == stat
    assert spec.column == "Age"


@pytest.mark.parametrize(
    "bad_name",
    ["", "123num", "has space", "kebab-case", "@var", "foo.bar", "a+b"],
)
def test_variable_spec_invalid_name(bad_name):
    with pytest.raises(ValidationError):
        VariableSpec(name=bad_name, stat="mean", column="Age")


@pytest.mark.parametrize("bad_stat", ["variance", "sum", "mode", "avg", ""])
def test_variable_spec_invalid_stat(bad_stat):
    with pytest.raises(ValidationError):
        VariableSpec(name="my_var", stat=bad_stat, column="Age")


def test_variable_spec_strict_extra_forbidden():
    with pytest.raises(ValidationError):
        VariableSpec(name="my_var", stat="mean", column="Age", extra_field=123)


def test_workspace_variables_unique_names():
    ws_dict = _base_ws(
        variables=[
            {"name": "mean_age", "stat": "mean", "column": "Age"},
            {"name": "median_age", "stat": "median", "column": "Age"},
            {"name": "count_fare", "stat": "count", "column": "Fare"},
        ]
    )
    ws = Workspace.model_validate(ws_dict)
    assert len(ws.variables) == 3
    assert [v.name for v in ws.variables] == ["mean_age", "median_age", "count_fare"]


def test_workspace_variables_duplicate_names_refused():
    ws_dict = _base_ws(
        variables=[
            {"name": "age_stat", "stat": "mean", "column": "Age"},
            {"name": "age_stat", "stat": "median", "column": "Age"},
        ]
    )
    with pytest.raises(ValidationError, match="duplicate variable name"):
        Workspace.model_validate(ws_dict)


def test_save_workspace_rejects_duplicate_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    ws_dict = _base_ws(
        variables=[
            {"name": "v1", "stat": "mean", "column": "Age"},
            {"name": "v1", "stat": "std", "column": "Fare"},
        ]
    )
    with pytest.raises(KeyParamsError, match="duplicate variable name"):
        save_workspace(ws_dict)


def test_workspace_variables_backward_compatible():
    # Workspace without 'variables' field defaults to empty list
    ws = Workspace.model_validate(_base_ws())
    assert ws.variables == []
    dumped = ws.model_dump(mode="json")
    assert dumped["variables"] == []


def test_save_and_get_workspace_with_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    vars_list = [
        {"name": "mean_age", "stat": "mean", "column": "Age"},
        {"name": "q25_fare", "stat": "q25", "column": "Fare"},
        {"name": "q75_fare", "stat": "q75", "column": "Fare"},
    ]
    saved = save_workspace(_base_ws(variables=vars_list))
    assert saved["variables"] == vars_list

    fetched = get_workspace("w_vars")
    assert fetched["variables"] == vars_list
