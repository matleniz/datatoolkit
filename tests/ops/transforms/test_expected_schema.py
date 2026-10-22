"""expected_schema: params validation, every rule, all violations at once."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from dtk_engine import DtkTransformer
from dtk_engine.errors import KeyParamsError
from dtk_engine.ops.compare.issues import ISSUE_FIELDS
from dtk_engine.ops.schema_check import VALUE_CHARS
from dtk_engine.transform_registry import get_transform


def _check(df, **params):
    t = get_transform("expected_schema")
    p = t.parse(params)
    return t.check(df, p, t.fit(df, p))


def _by(found, check, column=None):
    return [f for f in found if f["check"] == check and f["column"] == column]


def _col(**rule):
    return {"columns": [rule], "extra_columns": "ignore"}


def test_registered_and_passes_data_through_unchanged():
    t = get_transform("expected_schema")
    df = pd.DataFrame({"a": [1, 2, None]})
    before = df.copy()
    p = t.parse({"columns": [{"name": "a", "min": 5}]})
    out = t.fit_apply(df, p)
    pd.testing.assert_frame_equal(out, before)
    pd.testing.assert_frame_equal(df, before)  # input untouched
    assert out is not df


def test_findings_use_the_issue_vocabulary_and_are_json():
    found = _check(pd.DataFrame({"a": [1]}), columns=[{"name": "b"}])
    assert found and set(ISSUE_FIELDS) <= set(found[0])
    json.dumps(found)


def test_required_column():
    df = pd.DataFrame({"a": [1]})
    assert _by(_check(df, **_col(name="b")), "schema", "b")
    assert not _check(df, **_col(name="b", required=False))
    assert not _check(df, **_col(name="a"))


def test_dtype_each_kind():
    df = pd.DataFrame(
        {
            "n": [1.5, 2.0],
            "i": [1, 2],
            "s": ["x", "y"],
            "b": [True, False],
            "d": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "c": pd.Categorical(["u", "v"]),
            "mixed": ["x", 3],
        }
    )
    ok = [("n", "numeric"), ("i", "numeric"), ("s", "string"), ("b", "bool"),
          ("d", "datetime"), ("c", "category"), ("s", "category")]
    for name, dtype in ok:
        assert not _check(df, **_col(name=name, dtype=dtype)), (name, dtype)
    bad = [("s", "numeric"), ("b", "numeric"), ("n", "string"), ("n", "datetime"),
           ("mixed", "string"), ("i", "bool"), ("n", "category")]
    for name, dtype in bad:
        found = _check(df, **_col(name=name, dtype=dtype))
        assert len(found) == 1 and found[0]["check"] == "schema", (name, dtype)


def test_nullable_false_boundary():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    found = _check(df, **_col(name="a", nullable=False))
    assert found[0]["check"] == "missing" and found[0]["count"] == 1
    assert found[0]["examples"] == [{"row": 1, "value": "nan"}]
    assert not _check(pd.DataFrame({"a": [1.0, 2.0]}), **_col(name="a", nullable=False))


def test_max_missing_pct_boundary():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})  # 25 %
    assert not _check(df, **_col(name="a", max_missing_pct=25))
    assert _check(df, **_col(name="a", max_missing_pct=24.9))


def test_min_max_boundaries_are_inclusive_and_nan_ignored():
    df = pd.DataFrame({"a": [0, 5, 10, np.nan]})
    assert not _check(df, **_col(name="a", min=0, max=10))
    found = _check(df, **_col(name="a", min=1, max=9))
    assert found[0]["check"] == "range" and found[0]["count"] == 2
    assert [e["row"] for e in found[0]["examples"]] == [0, 2]
    assert _check(df, **_col(name="a", max=9.99))


def test_datetime_bounds():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2021-06-01", None])})
    assert not _check(df, **_col(name="d", min="2020-01-01", max="2021-06-01"))
    found = _check(df, **_col(name="d", dtype="datetime", max="2021-01-01"))
    assert found[0]["count"] == 1 and found[0]["examples"][0]["row"] == 1


def test_timezone_mismatch_is_a_finding_not_a_crash():
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01"]).tz_localize("UTC")})
    assert _check(df, **_col(name="d", min="2019-01-01"))


def test_bounds_on_a_non_numeric_column_say_dtype_not_range():
    found = _check(pd.DataFrame({"a": ["x"]}), **_col(name="a", min=0))
    assert [f["check"] for f in found] == ["schema"]


def test_allowed():
    df = pd.DataFrame({"s": ["F", "M", "X", None, "Y"]})
    assert not _check(df, **_col(name="s", allowed=["F", "M", "X", "Y"]))
    found = _check(df, **_col(name="s", allowed=["F", "M"]))
    assert found[0]["check"] == "allowed" and found[0]["count"] == 2
    assert "'X'" in found[0]["message"]


def test_allowed_numbers_and_categoricals():
    assert not _check(pd.DataFrame({"a": [1, 2]}), **_col(name="a", allowed=[1, 2, 3]))
    cat = pd.DataFrame({"c": pd.Categorical(["u", "z"])})
    assert _check(cat, **_col(name="c", allowed=["u"]))[0]["count"] == 1


def test_unique_ignores_missing():
    assert not _check(pd.DataFrame({"a": [1, 2, None, None]}), **_col(name="a", unique=True))
    found = _check(pd.DataFrame({"a": [1, 2, 1, 1]}), **_col(name="a", unique=True))
    assert found[0]["count"] == 2 and [e["row"] for e in found[0]["examples"]] == [2, 3]


def test_row_bounds_and_extra_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "z": [0, 0, 0]})
    assert not _check(df, min_rows=3, max_rows=3, extra_columns="ignore")
    assert len(_check(df, min_rows=4, extra_columns="ignore")) == 1
    assert len(_check(df, max_rows=2, extra_columns="ignore")) == 1
    warn = _check(df, columns=[{"name": "a"}])
    assert [(f["severity"], f["check"]) for f in warn] == [("warning", "schema")]
    assert "'z'" in warn[0]["message"]


def test_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    assert not _check(df, **_col(name="a", dtype="numeric", nullable=False, min=0, unique=True))
    assert _check(df, min_rows=1, extra_columns="ignore")
    assert _check(pd.DataFrame(), **_col(name="a"))


def test_all_violations_at_once_errors_first():
    df = pd.DataFrame({"a": [1, 1, 99], "b": ["x", None, "z"], "extra": [0, 0, 0]})
    found = _check(
        df,
        min_rows=10,
        columns=[
            {"name": "a", "max": 50, "unique": True},
            {"name": "b", "nullable": False, "allowed": ["x"]},
            {"name": "gone"},
        ],
    )
    kinds = {(f["check"], f["column"]) for f in found}
    assert kinds == {
        ("schema", None), ("range", "a"), ("duplicates", "a"), ("missing", "b"),
        ("allowed", "b"), ("schema", "gone"),
    } | {("schema", None)}
    severities = [f["severity"] for f in found]
    assert severities == sorted(severities, key=["error", "warning", "info"].index)
    assert severities[-1] == "warning"


def test_examples_are_capped_and_values_truncated():
    df = pd.DataFrame({"a": ["x" * 500] * 30})
    found = _check(df, max_examples=3, **_col(name="a", allowed=["ok"]))
    ex = found[0]["examples"]
    assert found[0]["count"] == 30 and len(ex) == 3
    assert all(len(e["value"]) <= VALUE_CHARS and set(e) == {"row", "value"} for e in ex)
    assert "x" * 100 not in json.dumps(found)
    assert _check(df, max_examples=0, **_col(name="a", allowed=["ok"]))[0]["examples"] == []


def test_integer_column_labels():
    df = pd.DataFrame({0: [1, 2], 1: ["a", "b"]})
    assert not _check(df, columns=[{"name": "0", "dtype": "numeric"}, {"name": "1"}])
    assert _check(df, **_col(name="0", max=1))


def test_fit_state_counts_only():
    t = get_transform("expected_schema")
    p = t.parse({"columns": [{"name": "a", "max": 1}], "extra_columns": "ignore"})
    state = t.fit(pd.DataFrame({"a": [1, 2, 3]}), p)
    assert state == {"n_rows": 3, "violations": {"error": 1, "warning": 0, "info": 0}}


# --- params at the boundary ------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"columns": [{"name": "a", "min": 5, "max": 1}]}, "above max"),
        ({"columns": [{"name": "a", "min": "x"}]}, "not a number or a date"),
        ({"columns": [{"name": "a", "dtype": "string", "min": 1}]}, "numeric or datetime"),
        ({"columns": [{"name": "a", "dtype": "numeric", "min": "2020-01-01"}]}, "must be numbers"),
        ({"columns": [{"name": "a", "allowed": []}]}, "allowed"),
        ({"columns": [{"name": "a", "dtype": "numeric", "allowed": ["x"]}]}, "do not fit"),
        ({"columns": [{"name": "a", "dtype": "datetime", "allowed": ["x"]}]}, "datetime"),
        ({"columns": [{"name": "a", "nullable": False, "max_missing_pct": 5}]}, "contradictory"),
        ({"columns": [{"name": "a", "max_missing_pct": 101}]}, "100"),
        ({"columns": [{"name": "a", "min": True}]}, "min"),
        ({"columns": [{"name": "a", "min": float("inf")}]}, "finite"),
        ({"columns": [{"name": ""}]}, "name"),
        ({"columns": [{"name": "a"}, {"name": "a"}]}, "duplicate column rules"),
        ({"columns": [{"name": "a", "pattern": "x"}]}, "pattern"),
        ({"columns": [{"name": str(i)} for i in range(501)]}, "500"),
        ({"columns": [{"name": "a", "allowed": list(range(1001))}]}, "1000"),
        ({"min_rows": 5, "max_rows": 2}, "above max_rows"),
        ({"min_rows": -1}, "min_rows"),
        ({"schema_version": 2}, "schema_version"),
        ({"on_violation": "explode"}, "on_violation"),
        ({"max_examples": 51}, "max_examples"),
        ({"bogus": 1}, "bogus"),
    ],
)
def test_bad_params_are_rejected_clearly(params, message):
    with pytest.raises(KeyParamsError, match=message) as exc:
        get_transform("expected_schema").parse(params)
    assert "expected_schema" in str(exc.value)


def test_params_json_schema_is_a_nested_array_of_objects():
    schema = get_transform("expected_schema").params_model.model_json_schema()
    cols = schema["properties"]["columns"]
    assert cols["type"] == "array" and cols["items"] == {"$ref": "#/$defs/ColumnRule"}
    assert "name" in schema["$defs"]["ColumnRule"]["required"]


# --- sklearn door ----------------------------------------------------------------


def test_dtk_transformer_error_policy_raises_and_report_warns():
    df = pd.DataFrame({"a": [1, 2, 99]})
    rule = [{"name": "a", "max": 10}]
    with pytest.raises(ValueError, match="expected schema violated"):
        DtkTransformer("expected_schema", columns=rule, on_violation="error").fit(df)
    est = DtkTransformer("expected_schema", columns=rule).fit(df.iloc[:2])
    with pytest.warns(UserWarning, match="expected_schema"):
        out = est.transform(df)
    pd.testing.assert_frame_equal(out, df)
    strict = DtkTransformer("expected_schema", columns=rule, on_violation="error").fit(df.iloc[:2])
    with pytest.raises(ValueError, match="expected schema violated"):
        strict.transform(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        est.transform(df.iloc[:2])
