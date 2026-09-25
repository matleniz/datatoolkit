import pandas as pd
import pytest
from dtk_engine.ops.join import (
    LabelJoinError,
    is_index_like,
    join_labels,
    label_columns,
)


def _x(n=3):
    return pd.DataFrame({"Index": range(n), "id": list("abcdefgh")[:n], "f": range(n)})


def test_order_single_column():
    out = join_labels(_x(), pd.DataFrame({"target": [1.0, 2.0, 3.0]}), "order")
    assert out["target"].tolist() == [1.0, 2.0, 3.0]
    assert list(out.columns) == ["Index", "id", "f", "target"]


def test_order_drops_index_column_and_checks_alignment():
    y = pd.DataFrame({"Index": [0, 1, 2], "target": [5, 6, 7]})
    out = join_labels(_x(), y, "order")
    assert list(out.columns) == ["Index", "id", "f", "target"]
    shuffled = pd.DataFrame({"Index": [2, 1, 0], "target": [5, 6, 7]})
    with pytest.raises(LabelJoinError, match="not aligned"):
        join_labels(_x(), shuffled, "order")


def test_order_uses_key_column_when_given():
    y = pd.DataFrame({"id": list("abc"), "target": [1, 2, 3]})
    assert join_labels(_x(), y, "order", key="id")["target"].tolist() == [1, 2, 3]
    with pytest.raises(LabelJoinError, match="not aligned"):
        join_labels(_x(), y.iloc[::-1], "order", key="id")


def test_order_row_count_mismatch():
    with pytest.raises(LabelJoinError, match="X has 3 rows, y has 2 rows"):
        join_labels(_x(), pd.DataFrame({"target": [1, 2]}), "order")


def test_order_too_many_value_columns():
    y = pd.DataFrame({"a": [0.5, 1, 2], "b": [1.5, 2, 3], "c": [4, 5, 6]})
    with pytest.raises(LabelJoinError, match="exactly one value column"):
        join_labels(_x(), y, "order")


def test_order_clash_with_x_column():
    with pytest.raises(LabelJoinError, match="already exist in X"):
        join_labels(_x(), pd.DataFrame({"f": [1, 2, 3]}), "order")


def test_index_like():
    assert is_index_like(pd.Series([7, 3], name="Unnamed: 0"))
    assert is_index_like(pd.Series([1, 2, 3], name="row"))
    assert not is_index_like(pd.Series([1, 3, 2], name="row"))
    assert not is_index_like(pd.Series([0.0, 1.0], name="target"))
    assert label_columns(pd.DataFrame({"target": [3.2, 1.1], "idx": [0, 1]})) == (
        "idx",
        "target",
    )


def test_key_join_keeps_x_order_and_index():
    x = _x().set_index(pd.Index([10, 11, 12]))
    y = pd.DataFrame({"id": ["c", "a", "b"], "target": [3, 1, 2]})
    out = join_labels(x, y, "key", key="id")
    assert out["target"].tolist() == [1, 2, 3]
    assert out.index.tolist() == [10, 11, 12]


def test_key_join_refuses_row_loss_with_counts():
    y = pd.DataFrame({"id": ["a", "b", "z"], "target": [1, 2, 9]})
    with pytest.raises(LabelJoinError) as err:
        join_labels(_x(), y, "key", key="id")
    assert "x_unmatched=1" in str(err.value) and "y_unmatched=1" in str(err.value)


def test_key_join_refuses_duplicates():
    y = pd.DataFrame({"id": ["a", "a", "b", "c"], "target": [1, 1, 2, 3]})
    with pytest.raises(LabelJoinError, match="y_duplicated_keys=1"):
        join_labels(_x(), y, "key", key="id")


def test_key_join_errors():
    y = pd.DataFrame({"id": list("abc"), "target": [1, 2, 3]})
    with pytest.raises(LabelJoinError, match="not in X"):
        join_labels(_x().drop(columns="id"), y, "key", key="id")
    with pytest.raises(LabelJoinError, match="key column is required"):
        join_labels(_x(), y, "key")
    with pytest.raises(LabelJoinError, match="no column besides"):
        join_labels(_x(), y[["id"]], "key", key="id")
    with pytest.raises(LabelJoinError, match="unknown label join mode"):
        join_labels(_x(), y, "zip")
