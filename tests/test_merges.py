from pathlib import Path

import pandas as pd
import pytest

from dtk_engine.contract import (
    export_workspace,
    get_workspace,
    preview_workspace,
    save_workspace,
)
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.join import merge_match_stats, merge_table
from dtk_engine.sources.dataset import workspace_frame
from dtk_engine.workspace.models import MergeSpec, Workspace

# --- Ops: merge_table & merge_match_stats ---


def test_merge_table_preserves_rows_order_and_index():
    x = pd.DataFrame(
        {"id": [3, 1, 2], "val": ["x3", "x1", "x2"]},
        index=[100, 200, 300],
    )
    merge_df = pd.DataFrame(
        {"id": [1, 2, 3], "extra": ["e1", "e2", "e3"], "note": ["n1", "n2", "n3"]}
    )

    out = merge_table(x, merge_df, key="id")
    assert list(out.index) == [100, 200, 300]
    assert list(out["id"]) == [3, 1, 2]
    assert list(out["extra"]) == ["e3", "e1", "e2"]
    assert list(out["note"]) == ["n3", "n1", "n2"]
    assert len(out) == len(x)


def test_merge_table_many_to_one():
    # Multiple rows in X with the same key
    x = pd.DataFrame(
        {"id": [1, 2, 1, 1], "name": ["a", "b", "c", "d"]},
        index=[10, 20, 30, 40],
    )
    merge_df = pd.DataFrame({"id": [1, 2], "category": ["cat1", "cat2"]})

    out = merge_table(x, merge_df, key="id")
    assert len(out) == 4
    assert list(out.index) == [10, 20, 30, 40]
    assert list(out["category"]) == ["cat1", "cat2", "cat1", "cat1"]


def test_merge_table_unmatched_rows_get_nan():
    x = pd.DataFrame({"id": [1, 2, 99], "val": ["a", "b", "c"]})
    merge_df = pd.DataFrame({"id": [1, 2], "extra": [10.0, 20.0]})

    out = merge_table(x, merge_df, key="id")
    assert len(out) == 3
    assert out.loc[0, "extra"] == 10.0
    assert out.loc[1, "extra"] == 20.0
    assert pd.isna(out.loc[2, "extra"])


def test_merge_table_column_selection():
    x = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    merge_df = pd.DataFrame(
        {"id": [1, 2], "col_a": [10, 20], "col_b": [100, 200], "col_c": [1, 2]}
    )

    out = merge_table(x, merge_df, key="id", columns=["col_b"])
    assert list(out.columns) == ["id", "val", "col_b"]
    assert "col_a" not in out.columns and "col_c" not in out.columns


def test_merge_table_matched_counts_reporting():
    x = pd.DataFrame({"id": [1, 2, 3, 99], "val": ["a", "b", "c", "d"]})
    merge_df = pd.DataFrame({"id": [1, 2, 3], "extra": ["e1", "e2", "e3"]})

    stats = merge_match_stats(x, merge_df, key="id")
    assert stats == {"matched": 3, "total": 4}

    out, stats_out = merge_table(x, merge_df, key="id", return_stats=True)
    assert stats_out == {"matched": 3, "total": 4}
    assert len(out) == 4


def test_merge_table_missing_key_in_x():
    x = pd.DataFrame({"other_id": [1, 2]})
    merge_df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})

    with pytest.raises(KeyParamsError, match="column 'id' not in X"):
        merge_table(x, merge_df, key="id")
    with pytest.raises(KeyParamsError, match="column 'id' not in X"):
        merge_match_stats(x, merge_df, key="id")


def test_merge_table_missing_key_in_merge_df():
    x = pd.DataFrame({"id": [1, 2]})
    merge_df = pd.DataFrame({"other_id": [1, 2], "val": [10, 20]})

    with pytest.raises(KeyParamsError, match="column 'id' not in merge table"):
        merge_table(x, merge_df, key="id")
    with pytest.raises(KeyParamsError, match="column 'id' not in merge table"):
        merge_match_stats(x, merge_df, key="id")


def test_merge_table_duplicated_keys_in_merge_df_refused():
    x = pd.DataFrame({"id": [1, 2]})
    merge_df = pd.DataFrame({"id": [1, 2, 2, 3, 3], "val": [10, 20, 21, 30, 31]})

    with pytest.raises(KeyParamsError, match="duplicated keys") as exc:
        merge_table(x, merge_df, key="id")
    assert "2" in str(exc.value) and "3" in str(exc.value)


def test_merge_table_null_keys_in_merge_df_refused():
    x = pd.DataFrame({"id": [1, 2]})
    merge_df = pd.DataFrame({"id": [1, None], "val": [10, 20]})

    with pytest.raises(KeyParamsError, match="null value"):
        merge_table(x, merge_df, key="id")


def test_merge_table_column_clash_refused():
    x = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    merge_df = pd.DataFrame({"id": [1, 2], "name": ["x", "y"]})

    with pytest.raises(KeyParamsError, match="already exist in X"):
        merge_table(x, merge_df, key="id")


def test_merge_table_missing_requested_column():
    x = pd.DataFrame({"id": [1, 2]})
    merge_df = pd.DataFrame({"id": [1, 2], "col_a": [10, 20]})

    with pytest.raises(KeyParamsError, match="not in merge table"):
        merge_table(x, merge_df, key="id", columns=["col_a", "non_existent"])


# --- Workspace Integration: Merges ---


@pytest.fixture
def workspace_files(tmp_path):
    train_path = tmp_path / "train.csv"
    train_path.write_text("id,age,target\n1,25,0\n2,30,1\n3,35,0\n4,40,1\n")

    test_path = tmp_path / "test.csv"
    test_path.write_text("id,age\n3,35\n4,40\n5,45\n")

    extra_path = tmp_path / "extra.csv"
    extra_path.write_text("id,city,tier\n1,Paris,gold\n2,Lyon,silver\n3,Paris,bronze\n4,Marseille,gold\n")

    dup_extra_path = tmp_path / "dup_extra.csv"
    dup_extra_path.write_text("id,region\n1,IDF\n1,PACA\n2,ARA\n")

    return {
        "train": str(train_path),
        "test": str(test_path),
        "extra": str(extra_path),
        "dup_extra": str(dup_extra_path),
    }


def _make_ws(files, *, apply_to="both", columns=None, merge_file=None):
    mf = merge_file or files["extra"]
    return {
        "name": "demo_ws",
        "datasets": {
            "train": {
                "x": {"kind": "csv", "path": files["train"]},
                "target_column": "target",
            },
            "test": {"x": {"kind": "csv", "path": files["test"]}},
        },
        "merges": [
            {
                "source": {"kind": "csv", "path": mf},
                "key": "id",
                "apply_to": apply_to,
                "columns": columns,
            }
        ],
    }


def test_workspace_merge_train_only(workspace_files, tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    ws_dict = _make_ws(workspace_files, apply_to="train")
    ws = Workspace.model_validate(ws_dict)

    train_df = workspace_frame(ws, "train")
    assert "city" in train_df.columns and "tier" in train_df.columns
    assert list(train_df["city"]) == ["Paris", "Lyon", "Paris", "Marseille"]

    test_df = workspace_frame(ws, "test")
    assert "city" not in test_df.columns and "tier" not in test_df.columns


def test_workspace_merge_both(workspace_files, tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    ws_dict = _make_ws(workspace_files, apply_to="both")
    ws = Workspace.model_validate(ws_dict)

    train_df = workspace_frame(ws, "train")
    assert "city" in train_df.columns and "tier" in train_df.columns

    test_df = workspace_frame(ws, "test")
    assert "city" in test_df.columns and "tier" in test_df.columns
    # Row 5 was unmatched in extra.csv -> NaN
    assert list(test_df["id"]) == [3, 4, 5]
    assert test_df.loc[test_df["id"] == 3, "city"].iloc[0] == "Paris"
    assert test_df.loc[test_df["id"] == 4, "city"].iloc[0] == "Marseille"
    assert pd.isna(test_df.loc[test_df["id"] == 5, "city"].iloc[0])


def test_workspace_merge_duplicated_keys_refused_at_save_and_replay(
    workspace_files, tmp_path, monkeypatch
):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    ws_dict = _make_ws(workspace_files, merge_file=workspace_files["dup_extra"])

    # At save: refused with KeyParamsError listing duplicated keys
    with pytest.raises(KeyParamsError, match="duplicated keys") as exc:
        save_workspace(ws_dict)
    assert "1" in str(exc.value)

    # At preview: refused with KeyParamsError
    with pytest.raises(KeyParamsError, match="duplicated keys"):
        preview_workspace(ws_dict, "train")

    # If bypassed model validation into labeled_frame directly: refused with KeyParamsError
    clean_dict = _make_ws(workspace_files)
    raw_ws = Workspace.model_construct(
        name="bypass",
        datasets=Workspace.model_validate(clean_dict).datasets,
        merges=[
            MergeSpec(
                source={"kind": "csv", "path": workspace_files["dup_extra"]},
                key="id",
                apply_to="both",
            )
        ],
    )
    with pytest.raises(KeyParamsError, match="duplicated keys"):
        workspace_frame(raw_ws, "train")


def test_workspace_merge_clash_refused(workspace_files, tmp_path):
    # Merge source introduces a column named 'age' which already exists in train X
    clash_file = tmp_path / "clash.csv"
    clash_file.write_text("id,age\n1,99\n2,99\n")
    ws_dict = _make_ws(workspace_files, merge_file=str(clash_file))

    ws = Workspace.model_validate(ws_dict)
    with pytest.raises(KeyParamsError, match="already exist in X"):
        workspace_frame(ws, "train")


def test_workspace_merge_roundtrip_save_and_get(workspace_files, tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    ws_dict = _make_ws(workspace_files, columns=["city"])
    saved = save_workspace(ws_dict)

    assert len(saved["merges"]) == 1
    assert saved["merges"][0]["key"] == "id"
    assert saved["merges"][0]["columns"] == ["city"]
    assert saved["merges"][0]["apply_to"] == "both"

    fetched = get_workspace("demo_ws")
    assert fetched == saved


def test_preview_workspace_shows_merged_columns(workspace_files):
    ws_dict = _make_ws(workspace_files, apply_to="both", columns=["city"])
    preview = preview_workspace(ws_dict, "train")

    assert "city" in preview["columns"]
    assert len(preview["head"]) == 4
    assert [r["city"] for r in preview["head"]] == ["Paris", "Lyon", "Paris", "Marseille"]

    preview_test = preview_workspace(ws_dict, "test")
    assert "city" in preview_test["columns"]
    assert len(preview_test["head"]) == 3


def test_export_manifest_records_merges_and_variables(workspace_files, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DTK_HOME", str(home))
    ws_dict = _make_ws(workspace_files, columns=["city", "tier"])
    ws_dict["variables"] = [
        {"name": "mean_age", "stat": "mean", "column": "age"},
    ]
    save_workspace(ws_dict)

    export_dir = tmp_path / "export"
    manifest = export_workspace("demo_ws", str(export_dir))

    assert "merges" in manifest
    assert len(manifest["merges"]) == 1
    merge_entry = manifest["merges"][0]
    assert merge_entry["key"] == "id"
    assert merge_entry["apply_to"] == "both"
    assert merge_entry["columns"] == ["city", "tier"]
    assert "sha256" in merge_entry and merge_entry["size"] > 0
    assert merge_entry["path"] == str(Path(workspace_files["extra"]).resolve())

    assert "variables" in manifest
    assert manifest["variables"] == [{"name": "mean_age", "stat": "mean", "column": "age"}]

    # Check that processed parquet contains merged columns
    train_df = pd.read_parquet(export_dir / "processed" / "train.parquet")
    assert "city" in train_df.columns and "tier" in train_df.columns

    test_df = pd.read_parquet(export_dir / "processed" / "test.parquet")
    assert "city" in test_df.columns and "tier" in test_df.columns


def test_export_refuses_to_overwrite_merge_source(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("DTK_HOME", str(home))
    raw_merge = tmp_path / "processed" / "extra.csv"
    raw_merge.parent.mkdir(parents=True, exist_ok=True)
    raw_merge.write_text("id,city\n1,Paris\n")

    train_path = tmp_path / "train.csv"
    train_path.write_text("id,val\n1,10\n")

    ws_dict = {
        "name": "w_merge_overwrite",
        "datasets": {
            "train": {"x": {"kind": "csv", "path": str(train_path)}}
        },
        "merges": [
            {
                "source": {"kind": "csv", "path": str(raw_merge)},
                "key": "id",
            }
        ],
    }
    save_workspace(ws_dict)

    with pytest.raises(KeyParamsError, match="would overwrite the raw input"):
        export_workspace("w_merge_overwrite", str(tmp_path))

