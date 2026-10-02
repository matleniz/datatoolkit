import pandas as pd
import pytest

from dtk_engine.ops.join import (
    LabelJoinError,
    join_labels,
    key_candidates,
    key_diagnostics,
    order_diagnostics,
)


@pytest.mark.parametrize(
    "y",
    [
        pd.DataFrame({"id": [1, 2, 3], "t": [0, 1, 0]}),
        pd.DataFrame({"id": [1, 2, 4], "t": [0, 1, 0]}),
        pd.DataFrame({"id": [1, 2, 2], "t": [0, 1, 0]}),
        pd.DataFrame({"id": [1, 2], "t": [0, 1]}),
    ],
)
def test_would_join_mirrors_join_labels(y):
    x = pd.DataFrame({"id": [1, 2, 3], "a": [5, 6, 7]})
    for mode, diag in (
        ("key", key_diagnostics(x, y, "id")),
        ("order", order_diagnostics(x, y, "id")),
    ):
        try:
            join_labels(x, y, mode, "id")
            ok = True
        except LabelJoinError:
            ok = False
        assert diag["would_join"] is ok


def test_key_diagnostics_counts():
    x = pd.DataFrame({"id": [1, 2, 3, 3], "a": range(4)})
    y = pd.DataFrame({"id": [3, 3, 9], "t": [0, 1, 1]})
    d = key_diagnostics(x, y, "id")
    assert (d["x_unmatched"], d["y_unmatched"]) == (2, 1)
    assert d["result_rows"] == 2 + 4  # ids 1, 2 unmatched; each id-3 row matches twice
    assert d["extra_rows"] == 2
    assert d["match_x_to_y"] == 0.5


def test_key_candidates_auto_and_explicit():
    x = pd.DataFrame({"id": [10, 20, 30, 40], "flag": [0, 0, 0, 0], "a": [1, 2, 3, 4]})
    y = pd.DataFrame({"id": [10, 20, 30, 40], "flag": [1, 1, 1, 1], "t": [0, 1, 0, 1]})
    assert key_candidates(x, y) == ["id"]
    assert key_candidates(x, y, ["flag", "zz"]) == ["flag", "zz"]


def test_order_diagnostics_never_raises_on_multi_value_y():
    x = pd.DataFrame({"a": [1, 2]})
    y = pd.DataFrame({"t": [0, 1], "u": [1, 1], "v": [2, 2]})
    d = order_diagnostics(x, y)
    assert d["would_join"] is False
    assert d["value_error"]


def test_key_diagnostics_dtype_mismatch_does_not_raise():
    x = pd.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3]})
    y = pd.DataFrame({"id": ["1", "2", "3"], "label": [0, 1, 0]})
    d = key_diagnostics(x, y, "id")
    assert d["dtype_mismatch"] is True
    assert d["match_x_to_y"] == 0 and d["match_y_to_x"] == 0
    assert d["result_rows"] == 3 and d["would_join"] is False


def test_key_diagnostics_result_rows_counts_duplicates_and_nan():
    x = pd.DataFrame({"id": [1, 2, None], "a": [1, 2, 3]})
    y = pd.DataFrame({"id": [1, 1, None, 5.0], "label": [0, 1, 0, 1]})
    d = key_diagnostics(x, y, "id")
    assert d["result_rows"] == len(x.merge(y[["id"]], on="id", how="left"))
