import pytest

from dtk_engine import run_key
from dtk_engine.errors import KeyParamsError, SourceError


def test_defaults_on_demo_train():
    res = run_key("dataset_overview", {})
    m = res["metrics"]
    assert (m["rows"], m["cols"]) == (41, 12)
    assert m["n_duplicate_rows"] == 1
    assert 0 < m["pct_missing_cells"] < 100
    assert m["memory_mb"] > 0
    tables = {t["title"]: t for t in res["tables"]}
    columns, head = tables["columns"], tables["head"]
    assert len(columns["records"]) == 12
    assert len(head["records"]) == 5
    types = {r["column"]: r["semantic_type"] for r in columns["records"]}
    assert types["PassengerId"] == "id_like"
    assert types["Survived"] == "boolean"
    assert types["Sex"] == "categorical"
    assert {f["title"] for f in res["figures"]} >= {"% missing per column"}
    groups = [t["group"] for t in res["tables"]] + [f["group"] for f in res["figures"]]
    assert None not in groups and set(groups) >= {"Overview", "Numeric", "Categorical"}
    assert {"numeric stats", "category summary", "category values"} <= set(tables)
    assert "distributions" in {f["title"] for f in res["figures"]}


def test_only_groups_with_columns(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("x\n" + "\n".join(str(i % 7 + i * 0.5) for i in range(30)) + "\n")
    res = run_key("dataset_overview", {"source": {"kind": "csv", "path": str(path)}})
    groups = {t["group"] for t in res["tables"]}
    assert groups == {"Overview", "Numeric"}


def test_custom_source_and_head_rows(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("x;y\n1;a\n2;\n3;c\n")
    res = run_key(
        "dataset_overview",
        {"source": {"kind": "csv", "path": str(path)}, "head_rows": 2},
    )
    assert res["metrics"]["rows"] == 3
    assert res["metrics"]["pct_missing_cells"] == pytest.approx(16.67)
    head = next(t for t in res["tables"] if t["title"] == "head")
    assert len(head["records"]) == 2


def test_missing_file_surfaces_source_error(tmp_path):
    with pytest.raises(SourceError):
        run_key(
            "dataset_overview", {"source": {"kind": "csv", "path": str(tmp_path / "x")}}
        )


def test_unknown_source_kind_rejected():
    with pytest.raises(KeyParamsError):
        run_key("dataset_overview", {"source": {"kind": "xml", "path": "x"}})
