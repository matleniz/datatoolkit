"""scikit-learn's estimator checks on DtkTransformer (sample of ops).

``EXPECTED_FAILED_CHECKS`` is the one place listing the checks that cannot
apply to a DataFrame-in / DataFrame-out transformer, with the reason each.
"""

import pandas as pd
from sklearn.utils.estimator_checks import parametrize_with_checks

from dtk_engine import DtkTransformer, save_workspace, workspace_pipeline
from dtk_engine.demo_data import TRAIN_CSV
from dtk_engine.transform_registry import get_transform

# Sample ops. Arrays reach the op as columns x0..xN, so params name ``x0`` only
# (every check feeds at least one feature).
SAMPLE = [
    DtkTransformer("impute", columns=["x0"]),
    DtkTransformer("scale", columns=["x0"]),
    DtkTransformer("onehot", columns=["x0"]),
    DtkTransformer("clip", columns=["x0"]),
    DtkTransformer("formula", name="z", expr="x0 * 2"),
    DtkTransformer("drop_columns", columns=["x0"]),
]

EXPECTED_FAILED_CHECKS = {
    "check_dtype_object": (
        "ops validate their own columns (numeric-only ops raise their own message, "
        "text-friendly ops accept object data); no generic TypeError wording"
    ),
    "check_estimators_nan_inf": (
        "NaN is legitimate input (impute, onehot missing=...); the op decides, "
        "there is no blanket finite-value check"
    ),
    "check_transformer_preserve_dtypes": (
        "transform always returns a DataFrame (pandas in / pandas out), not an "
        "ndarray with the input dtype"
    ),
}


def _expected_failed_checks(estimator):
    return EXPECTED_FAILED_CHECKS


@parametrize_with_checks(SAMPLE, expected_failed_checks=_expected_failed_checks)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


def test_workspace_replay_frames_unchanged(tmp_path, monkeypatch):
    """Replaying through DtkTransformer == chaining the ops directly (as on main)."""
    monkeypatch.setenv("DTK_HOME", str(tmp_path))
    steps = [
        {"op": "impute", "target": "both", "params": {"columns": ["Age"]}},
        {"op": "scale", "target": "both", "params": {"columns": ["Age", "Fare"]}},
        {"op": "drop_columns", "target": "both", "params": {"columns": ["Name"]}},
    ]
    save_workspace(
        {
            "name": "replay",
            "datasets": {"train": {"x": {"kind": "csv", "path": TRAIN_CSV}}},
            "steps": steps,
        }
    )
    df = pd.read_csv(TRAIN_CSV)
    expected = df
    for s in steps:
        t = get_transform(s["op"])
        p = t.parse(s["params"])
        expected = t.apply(expected, p, t.fit(expected, p))
    got = workspace_pipeline("replay").fit_transform(df)
    pd.testing.assert_frame_equal(got, expected)
