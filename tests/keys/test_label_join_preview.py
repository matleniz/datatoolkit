from pathlib import Path

import pandas as pd
import pytest

from dtk_engine import api, list_keys, run_key
from dtk_engine.demo_data import TRAIN_CSV

FIX = Path(__file__).parent.parent / "fixtures" / "label_join_preview"


def _csv(name):
    return {"kind": "csv", "path": str(FIX / name)}


def _tables(res):
    return {t["title"]: t["records"] for t in res["tables"]}


def _candidates(res):
    return {(c["mode"], c["key"]): c for c in _tables(res)["candidates"]}


def test_key_is_listed():
    assert "label_join_preview" in {k["id"] for k in list_keys()}


def test_defaults_prefer_aligned_order_over_duplicated_key():
    # The demo train CSV carries one duplicated row, so its id is not a clean key.
    res = run_key("label_join_preview", {})
    m = res["metrics"]
    assert m["x_rows"] == m["y_rows"] and m["y_columns"] == 2
    cands = _candidates(res)
    assert cands[("order", "PassengerId")]["would_join"] is True
    assert cands[("order", "PassengerId")]["aligned"] is True
    assert cands[("key", "PassengerId")]["would_join"] is False
    assert cands[("key", "PassengerId")]["x_duplicated"] == 1
    assert (m["recommended_mode"], m["recommended_key"]) == ("order", "PassengerId")
    assert _tables(res)["head"][0].keys() >= {"X: PassengerId", "y: Survived"}


def test_demo_xy_without_shared_id_recommends_order(tmp_path):
    train = pd.read_csv(TRAIN_CSV)
    train.drop(columns=["Survived"]).drop(columns=["PassengerId"]).to_csv(
        tmp_path / "x.csv", index=False
    )
    train[["Survived"]].to_csv(tmp_path / "y.csv", index=False)
    res = run_key(
        "label_join_preview",
        {
            "x": {"kind": "csv", "path": str(tmp_path / "x.csv")},
            "y": {"kind": "csv", "path": str(tmp_path / "y.csv")},
        },
    )
    assert res["metrics"]["recommended_mode"] == "order"
    cands = _candidates(res)
    assert cands[("order", None)]["would_join"] is True
    assert cands[("order", None)]["aligned"] is None
    assert any(i["check"] == "unverified_order" for i in _tables(res)["issues"])


def test_mismatched_ids_report_match_rates_and_rows():
    res = run_key(
        "label_join_preview",
        {"x": _csv("x_mismatch.csv"), "y": _csv("y_mismatch.csv")},
    )
    c = _candidates(res)[("key", "id")]
    assert c["would_join"] is False
    assert c["match_x_to_y"] == pytest.approx(4 / 6)
    assert c["match_y_to_x"] == pytest.approx(4 / 5)
    assert c["result_rows"] == 6
    assert res["metrics"]["recommended_mode"] == "none"
    issues = _tables(res)["issues"]
    assert issues[0]["severity"] == "error"
    assert {i["check"] for i in issues} >= {"unmatched", "row_count"}


def test_duplicate_keys_multiplying_rows_are_flagged():
    res = run_key(
        "label_join_preview",
        {"x": _csv("x_mismatch.csv"), "y": _csv("y_dups.csv")},
    )
    c = _candidates(res)[("key", "id")]
    assert c["y_unique"] is False
    assert c["y_duplicated"] == 1
    assert c["extra_rows"] == 1
    assert c["result_rows"] == 7
    dup = [i for i in _tables(res)["issues"] if i["check"] == "duplicates"]
    assert dup and "multiply" in dup[0]["message"]


def test_shuffled_order_is_misaligned_but_key_is_clean():
    res = run_key(
        "label_join_preview",
        {"x": _csv("x_small.csv"), "y": _csv("y_shuffled.csv")},
    )
    cands = _candidates(res)
    assert cands[("order", "id")]["aligned"] is False
    assert cands[("order", "id")]["would_join"] is False
    assert res["metrics"]["recommended_mode"] == "key"
    assert any(i["check"] == "misaligned" for i in _tables(res)["issues"])


def test_explicit_missing_key_column_is_an_error():
    res = run_key(
        "label_join_preview",
        {
            "x": _csv("x_small.csv"),
            "y": _csv("y_shuffled.csv"),
            "key_columns": ["nope"],
        },
    )
    assert any(i["check"] == "key_missing" for i in _tables(res)["issues"])


def test_notebook_api_on_dataframes():
    x = pd.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3]})
    y = pd.DataFrame({"id": [1, 2, 4], "label": [0, 1, 1]})
    res = api.label_join_preview(x, y)
    assert res.metrics["recommended_mode"] == "none"
    assert res.metrics["n_key_candidates"] == 1
