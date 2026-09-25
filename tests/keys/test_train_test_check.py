import pytest

from dtk_engine import run_key
from dtk_engine.errors import KeyParamsError, SourceError


def _by_title(res):
    return {t["title"]: t["records"] for t in res["tables"]}


def test_defaults_find_demo_inconsistencies():
    res = run_key("train_test_check", {})
    m = res["metrics"]
    assert (m["n_common"], m["n_only_train"], m["n_only_test"]) == (11, 1, 0)
    assert m["n_dtype_mismatch"] == 1
    tables = _by_title(res)
    issues = tables["issues"]
    assert m["n_issues"] == len(issues)
    found = {(i["severity"], i["check"], i["column"]) for i in issues}
    assert ("info", "schema", "Survived") in found  # target only in train
    assert ("error", "schema", "Age") in found  # float vs str
    unseen = [i for i in issues if i["check"] == "categorical"]
    assert any(i["column"] == "Embarked" and "Q" in i["message"] for i in unseen)
    assert issues[0]["severity"] == "error"

    columns = {c["column"]: c for c in tables["columns"]}
    assert len(columns) == 12
    assert columns["Survived"]["in_test"] is False
    assert columns["Embarked"]["n_unseen_categories"] == 1
    assert columns["Age"]["pct_missing_delta"] < 0

    overlap = tables["overlap"]
    assert overlap[0]["kind"] == "rows" and overlap[0]["n_test_in_train"] == 0
    assert "PassengerId" in {o["column"] for o in overlap if o["kind"] == "id"}
    assert len(res["figures"]) >= 2  # % missing + drift histograms
    assert res["figures"][0]["title"] == "% missing train vs test"
    assert all(
        f["title"].endswith("train vs test") or ": train vs test" in f["title"]
        for f in res["figures"]
    )
    assert len(res["figures"]) <= 7
    assert {"numeric_drift", "categorical_drift"} <= set(tables)
    assert m["n_drifted"] >= 1
    assert {d["column"] for d in tables["numeric_drift"]} >= {"Fare"}
    assert all(d["diff"] is not None for d in tables["categorical_drift"])
    assert any(i["check"] == "drift" for i in issues)


def test_explicit_id_columns_detect_leak(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    train.write_text("patient_id,x,y\n101,1,0\n205,2,1\n309,3,0\n")
    test.write_text("patient_id,x\n309,4\n412,5\n")
    res = run_key(
        "train_test_check",
        {
            "train": {"kind": "csv", "path": str(train)},
            "test": {"kind": "csv", "path": str(test)},
            "id_columns": ["patient_id", "nope"],
        },
    )
    issues = _by_title(res)["issues"]
    leak = [i for i in issues if i["column"] == "patient_id"]
    assert leak[0]["severity"] == "error" and "50.0%" in leak[0]["message"]
    assert any(i["column"] == "nope" and i["severity"] == "error" for i in issues)
    assert res["metrics"]["n_errors"] == 2


def test_missing_test_file_surfaces_source_error(tmp_path):
    with pytest.raises(SourceError):
        run_key(
            "train_test_check", {"test": {"kind": "csv", "path": str(tmp_path / "x")}}
        )


def test_bad_id_columns_rejected():
    with pytest.raises(KeyParamsError):
        run_key("train_test_check", {"id_columns": "PassengerId"})
