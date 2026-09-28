"""MAT-175: keys honour DatasetSource.version (time travel), same semantics as
``workspace_rows`` / ``column_profiles``. Workspace: raw -> v1 Impute -> v2 Scale.
"""

import pytest

from dtk_engine import run_key, save_workspace
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.outliers import univariate_outliers
from dtk_engine.sources import DatasetSource, load


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("DTK_HOME", str(tmp_path / "home"))
    train = tmp_path / "train.csv"
    # "f" has one missing value and a clear high outlier once imputed.
    train.write_text("id,f\n1,1\n2,2\n3,3\n4,4\n5,\n6,100\n")
    ws = {
        "name": "w",
        "datasets": {"train": {"x": {"kind": "csv", "path": str(train)}}},
        "steps": [
            {
                "op": "impute",
                "target": "both",
                "params": {"columns": ["f"], "strategy": "median"},
            },
            {
                "op": "scale",
                "target": "both",
                "params": {"columns": ["f"], "method": "standard"},
            },
        ],
    }
    return save_workspace(ws)


def test_outliers_fences_follow_version(workspace):
    imputed = load(DatasetSource(workspace="w", version=1))
    unscaled_table = univariate_outliers(imputed, ["f"])
    unscaled_lo = float(unscaled_table.iloc[0]["lower_fence"])
    unscaled_hi = float(unscaled_table.iloc[0]["upper_fence"])

    at_v1 = run_key(
        "outliers",
        {"source": {"kind": "dataset", "workspace": "w", "version": 1}, "columns": ["f"]},
    )
    table_v1 = at_v1["tables"][0]["records"][0]
    assert table_v1["lower_fence"] == pytest.approx(unscaled_lo)
    assert table_v1["upper_fence"] == pytest.approx(unscaled_hi)

    scaled = load(DatasetSource(workspace="w"))  # version=None -> every step
    scaled_table = univariate_outliers(scaled, ["f"])
    scaled_lo = float(scaled_table.iloc[0]["lower_fence"])
    scaled_hi = float(scaled_table.iloc[0]["upper_fence"])
    assert scaled_lo != pytest.approx(unscaled_lo)

    at_latest = run_key(
        "outliers",
        {"source": {"kind": "dataset", "workspace": "w"}, "columns": ["f"]},
    )
    table_latest = at_latest["tables"][0]["records"][0]
    assert table_latest["lower_fence"] == pytest.approx(scaled_lo)
    assert table_latest["upper_fence"] == pytest.approx(scaled_hi)
    assert table_latest["lower_fence"] != pytest.approx(table_v1["lower_fence"])


def test_missing_values_at_raw_version_shows_missing_count(workspace):
    raw = run_key(
        "missing_values",
        {"source": {"kind": "dataset", "workspace": "w", "version": 0}},
    )
    assert raw["metrics"]["n_missing_cells"] == 1

    after_impute = run_key(
        "missing_values",
        {"source": {"kind": "dataset", "workspace": "w", "version": 1}},
    )
    assert after_impute["metrics"]["n_missing_cells"] == 0

    latest = run_key(
        "missing_values", {"source": {"kind": "dataset", "workspace": "w"}}
    )
    assert latest["metrics"]["n_missing_cells"] == 0


def test_version_out_of_range_raises_key_params_error(workspace):
    with pytest.raises(KeyParamsError, match="version 3 exceeds workspace step count"):
        run_key(
            "missing_values",
            {"source": {"kind": "dataset", "workspace": "w", "version": 3}},
        )
