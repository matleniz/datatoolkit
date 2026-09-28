"""Studio grid contract: workspace_rows, column_profiles, preview_step, align_report."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dtk_engine import (
    align_report,
    column_profiles,
    preview_step,
    workspace_rows,
)
from dtk_engine.demo_data import TEST_CSV, TRAIN_CSV
from dtk_engine.errors import KeyParamsError, SourceError, UnknownTransformError
from dtk_engine.workspace.inspect import column_kind


def _workspace(name="w", **extra):
    ws = {
        "name": name,
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": TRAIN_CSV},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": TEST_CSV}},
        },
    }
    ws.update(extra)
    return ws


def test_workspace_rows_basic_json_and_rids():
    out = workspace_rows(_workspace(), "train", limit=10)
    assert json.dumps(out)
    assert out["total"] == 41
    assert out["version"] == 0
    assert len(out["rows"]) == 10
    assert [r["_rid"] for r in out["rows"]] == list(range(10))
    kinds = {c["kind"] for c in out["columns"]}
    assert kinds <= {"number", "binary", "text", "date", "identifier", "bool"}
    assert {"name", "dtype", "kind"} <= set(out["columns"][0])


def test_workspace_rows_pagination_and_version():
    ws = _workspace(
        steps=[
            {
                "op": "drop_columns",
                "target": "both",
                "params": {"columns": ["Name"]},
            }
        ]
    )
    raw = workspace_rows(ws, "train", version=0, offset=0, limit=5)
    assert raw["version"] == 0
    assert any(c["name"] == "Name" for c in raw["columns"])
    after = workspace_rows(ws, "train", version=1, offset=5, limit=3)
    assert after["version"] == 1
    assert "Name" not in [c["name"] for c in after["columns"]]
    assert [r["_rid"] for r in after["rows"]] == [5, 6, 7]
    all_steps = workspace_rows(ws, "train")
    assert all_steps["version"] == 1


def test_workspace_rows_rid_survives_drop_duplicates(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("id,x,Survived\n1,a,0\n1,a,0\n2,b,1\n3,c,0\n")
    test.write_text("id,x\n1,a\n2,b\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
        "steps": [
            {
                "op": "drop_duplicates",
                "target": "train",
                "params": {
                    "subset": ["id", "x"],
                    "keep": "first",
                    "sort_by": ["id"],
                },
            }
        ],
    }
    before = workspace_rows(ws, "train", version=0)
    assert before["total"] == 4
    after = workspace_rows(ws, "train", version=1)
    assert after["total"] == 3
    assert [r["_rid"] for r in after["rows"]] == [0, 2, 3]


def test_workspace_rows_filter_preserves_rid(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("x,Survived\n1,0\n2,1\n3,0\n4,1\n")
    test.write_text("x\n1\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
        "steps": [
            {
                "op": "filter_rows",
                "target": "train",
                "params": {
                    "conditions": [{"column": "Survived", "op": "eq", "value": 1}]
                },
            }
        ],
    }
    out = workspace_rows(ws, "train")
    assert [r["_rid"] for r in out["rows"]] == [1, 3]


def test_workspace_rows_errors():
    with pytest.raises(KeyParamsError, match="role"):
        workspace_rows(_workspace(), "val")
    with pytest.raises(KeyParamsError, match="version"):
        workspace_rows(_workspace(), "train", version=-1)
    with pytest.raises(KeyParamsError, match="offset"):
        workspace_rows(_workspace(), "train", offset=-1)


def test_column_profiles_shape_and_helpers(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text(
        "Age,city,code,Survived\n"
        "1,Paris,1.5,0\n"
        "2,paris,2.0,1\n"
        "3,PARIS ,3.5,0\n"
        "4,Lyon,4.0,1\n"
        "-999,Lyon,5.5,0\n"
        "5,Lyon,6.0,1\n"
        "6,Lyon,7.5,0\n"
        "7,Lyon,8.0,1\n"
        "8,Lyon,9.5,0\n"
        "9,Lyon,10.0,1\n"
    )
    test.write_text("Age,city,code\n1,Paris,1.0\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    out = column_profiles(ws, "train")
    assert json.dumps(out)
    assert out["version"] == 0
    by = {c["name"]: c for c in out["columns"]}
    age = by["Age"]
    assert age["kind"] == "number"
    assert age["histogram"] is not None
    assert "edges" in age["histogram"] and "counts" in age["histogram"]
    assert age["top_values"] is None
    assert age["iqr_bounds"] is not None
    assert age["sentinel_candidates"]
    city = by["city"]
    assert city["kind"] == "text"
    assert city["histogram"] is None
    assert city["top_values"]
    assert city["variants"] == {"raw": 4, "normalized": 2}


def test_column_profiles_numbers_as_text_and_dates(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text(
        "amt,when,Survived\n"
        '"1.5","2020-01-01",0\n'
        '"2.0","2020-02-01",1\n'
        '"3.5","01/03/2020",0\n'
    )
    test.write_text("amt,when\n1.0,2020-01-01\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {
                    "kind": "csv",
                    "path": str(train),
                    "dtype": {"amt": "str"},
                },
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    by = {c["name"]: c for c in column_profiles(ws, "train")["columns"]}
    assert by["amt"]["numbers_as_text"] is True
    assert by["when"]["looks_like_dates"] is True


def test_column_profiles_comma_decimal_numbers_as_text(tmp_path):
    """European decimals ('41,0') must flag numbers_as_text on profiles + align."""
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("monthly_spend,Survived\n10.5,0\n20.0,1\n")
    test.write_text('monthly_spend\n"41,0"\n"9,5"\n"-3,25"\n')
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {
                "x": {
                    "kind": "csv",
                    "path": str(test),
                    "dtype": {"monthly_spend": "str"},
                }
            },
        },
    }
    by = {c["name"]: c for c in column_profiles(ws, "test")["columns"]}
    assert by["monthly_spend"]["numbers_as_text"] is True
    align = align_report(ws)
    spend = next(
        r for r in align["columns"] if r["train"] and r["train"]["name"] == "monthly_spend"
    )
    assert spend["numbers_as_text"] is True


def test_column_profiles_id_name_heuristic(tmp_path):
    """Text column named *_id with >=90% distinct non-null values -> identifier."""
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    # 18 distinct ids, 2 duplicate rows -> 18/20 = 0.9; below ID_MIN_NON_NULL
    # so semantic_type stays text, but the name heuristic promotes kind.
    ids = [f"C{i:03d}" for i in range(1, 19)] + ["C001", "C002"]
    train.write_text(
        "customer_id,Survived\n" + "".join(f"{i},0\n" for i in ids)
    )
    test.write_text("customer_id\nC001\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    by = {c["name"]: c for c in column_profiles(ws, "train")["columns"]}
    assert by["customer_id"]["kind"] == "identifier"
    rows = workspace_rows(ws, "train", limit=1)
    assert next(c["kind"] for c in rows["columns"] if c["name"] == "customer_id") == (
        "identifier"
    )


def test_preview_step_drop_columns_diff():
    ws = _workspace()
    out = preview_step(
        ws,
        {"op": "drop_columns", "target": "both", "params": {"columns": ["Name"]}},
        "train",
    )
    assert json.dumps(out)
    assert "Name" in out["removed_columns"]
    assert "Name" not in out["columns"]
    assert out["added_columns"] == []
    assert out["removed_rids"] == []
    assert out["shape"][0] == 41
    assert out["fitted_on"] == "train"
    assert isinstance(out["state"], dict)
    assert out["changed_total"] == 0


def test_preview_step_impute_reports_changed_and_state(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("Age,id\n10,1\n20,2\n,3\n")
    test.write_text("Age,id\n,1\n100,2\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(train)}},
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    out = preview_step(
        ws,
        {
            "op": "impute",
            "target": "both",
            "params": {"columns": ["Age"], "strategy": "median"},
        },
        "train",
    )
    assert out["fitted_on"] == "train"
    assert out["state"]
    assert out["changed_total"] >= 1
    assert any(c["column"] == "Age" and c["_rid"] == 2 for c in out["changed"])


def test_preview_step_removed_rids(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("x,Survived\n1,0\n2,\n3,1\n")
    test.write_text("x\n1\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    out = preview_step(
        ws,
        {
            "op": "drop_missing_target",
            "target": "train",
            "params": {"target": "Survived"},
        },
        "train",
    )
    assert out["removed_rids"] == [1]
    assert out["shape"] == [2, 2]


def test_preview_step_invalid_and_data_error():
    with pytest.raises(UnknownTransformError):
        preview_step(
            _workspace(), {"op": "nope", "target": "both", "params": {}}, "train"
        )
    with pytest.raises(KeyParamsError):
        preview_step(
            _workspace(),
            {"op": "drop_columns", "target": "both", "params": {}},
            "train",
        )
    with pytest.raises(SourceError, match="step"):
        preview_step(
            _workspace(),
            {
                "op": "drop_columns",
                "target": "both",
                "params": {"columns": ["does_not_exist"]},
            },
            "train",
        )


def test_preview_step_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    before = list(tmp_path.rglob("*"))
    preview_step(
        _workspace(),
        {"op": "drop_columns", "target": "both", "params": {"columns": ["Name"]}},
        "train",
    )
    assert list(tmp_path.rglob("*")) == before


def test_align_report_match_label_and_mismatch(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("a,b,Survived\n1,x,0\n2,y,1\n")
    test.write_text("a,b\n1,x\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    out = align_report(ws)
    assert json.dumps(out)
    by = {
        (r["train"]["name"] if r["train"] else r["test"]["name"]): r
        for r in out["columns"]
    }
    assert by["a"]["status"] == "match"
    assert by["b"]["status"] == "match"
    assert by["Survived"]["status"] == "label"
    assert by["Survived"]["test"] is None
    assert by["a"]["train_mean"] is not None


def test_align_report_missing_extra_similar_type_mismatch(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("support_calls,age,Survived\n1,10,0\n2,20,1\n")
    test.write_text("nb_support_calls,age,promo_code\n1,10,x\n2,20,y\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {
                "x": {
                    "kind": "csv",
                    "path": str(test),
                    "dtype": {"age": "str"},
                }
            },
        },
    }
    out = align_report(ws)
    rows = out["columns"]
    by_train = {r["train"]["name"]: r for r in rows if r["train"]}
    by_test_only = [r for r in rows if r["status"] == "extra_in_test"]
    assert by_train["support_calls"]["status"] == "missing_in_test"
    assert "nb_support_calls" in by_train["support_calls"]["similar"]
    assert by_train["age"]["status"] == "type_mismatch"
    assert by_train["age"]["numbers_as_text"] is True
    assert by_train["Survived"]["status"] == "label"
    assert {r["test"]["name"] for r in by_test_only} == {
        "nb_support_calls",
        "promo_code",
    }


def test_align_report_after_rename_step(tmp_path):
    train = tmp_path / "t.csv"
    test = tmp_path / "e.csv"
    train.write_text("support_calls,Survived\n1,0\n")
    test.write_text("nb_support_calls\n1\n")
    ws = {
        "name": "w",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "Survived",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
        "steps": [
            {
                "op": "rename",
                "target": "test",
                "params": {"mapping": {"nb_support_calls": "support_calls"}},
            }
        ],
    }
    out = align_report(ws)
    by = {r["train"]["name"]: r for r in out["columns"] if r["train"]}
    assert by["support_calls"]["status"] == "match"


def test_align_report_value_mismatch_adult_like_labels(tmp_path):
    """Train labels without trailing '.' vs test with '.' → value_mismatch + hint."""
    train = tmp_path / "adult_train.csv"
    test = tmp_path / "adult_test.csv"
    train.write_text(
        "age,income\n"
        "25,<=50K\n30,<=50K\n40,>50K\n50,>50K\n"
        "28,<=50K\n35,<=50K\n45,>50K\n55,>50K\n"
    )
    test.write_text(
        "age,income\n"
        "22,<=50K.\n33,>50K.\n44,<=50K.\n"
        "26,<=50K.\n38,>50K.\n"
    )
    ws = {
        "name": "adult",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": str(train)},
                "target_column": "income",
            },
            "test": {"x": {"kind": "csv", "path": str(test)}},
        },
    }
    out = align_report(ws)
    assert json.dumps(out)
    by = {r["train"]["name"]: r for r in out["columns"] if r["train"]}
    assert by["age"]["status"] == "match"
    assert by["age"]["only_in_test"] is None
    income = by["income"]
    assert income["status"] == "value_mismatch"
    assert income["pct_test_rows_unseen"] == 100.0
    assert {r["value"] for r in income["only_in_test"]} == {"<=50K.", ">50K."}
    assert sum(r["count"] for r in income["only_in_test"]) == 5
    assert income["near_matches"] == [
        {"test": "<=50K.", "train": "<=50K"},
        {"test": ">50K.", "train": ">50K"},
    ]
    assert "standardize_text" in income["near_match_hint"]


def test_column_kind_mapping():
    assert column_kind(pd.Series([1.0, 2.0, 3.0])) == "number"
    assert column_kind(pd.Series([0, 1, 0, 1])) == "bool"
    assert column_kind(pd.Series(["a", "b", "a"])) == "text"
    assert (
        column_kind(pd.Series(pd.to_datetime(["2020-01-01", "2020-02-01"]))) == "date"
    )
    ids = pd.Series([f"C{i:03d}" for i in range(1, 19)] + ["C001", "C002"])
    assert column_kind(ids, "customer_id") == "identifier"
    assert column_kind(ids, "userId") == "identifier"
    assert column_kind(ids, "id") == "identifier"
    assert column_kind(ids, "city") == "text"
    # Below 90% distinct -> stay text even with an id-like name.
    low = pd.Series(["A", "A", "B", "B", "C"])
    assert column_kind(low, "customer_id") == "text"
