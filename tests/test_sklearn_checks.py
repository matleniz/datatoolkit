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


def test_pickle_from_before_params_property_loads():
    """Main's layout kept ``params`` in the instance __dict__ (set in __init__)."""
    import pickle

    from sklearn.base import clone

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    fresh = DtkTransformer("scale", columns=["a"]).fit(df)
    old_state = {k: v for k, v in fresh.__getstate__().items() if k != "_params"}
    old_state["params"] = {"columns": ["a"]}  # main's layout
    old = DtkTransformer.__new__(DtkTransformer)
    old.__setstate__(old_state)
    pd.testing.assert_frame_equal(old.transform(df), fresh.transform(df))
    # the new layout round-trips through pickle and clone as before
    again = pickle.loads(pickle.dumps(fresh))
    pd.testing.assert_frame_equal(again.transform(df), fresh.transform(df))
    assert clone(fresh).get_params() == fresh.get_params()
