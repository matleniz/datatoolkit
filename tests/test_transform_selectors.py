"""Column-selector hints on transform params (x-dtk-source "step")."""

from dtk_engine import list_transforms, transform_schema

COLUMNISH = {
    "columns",
    "column",
    "target",
    "group",
    "by",
    "sort_by",
    "subset",
    "a",
    "b",
}


def _props(schema):
    yield from schema["properties"].items()
    for sub in schema.get("$defs", {}).values():
        yield from sub.get("properties", {}).items()


def test_every_column_param_carries_a_step_selector():
    checked = 0
    for t in list_transforms():
        for name, prop in _props(transform_schema(t["op"])):
            if name in COLUMNISH:
                checked += 1
                assert prop.get("x-dtk-widget") in ("columns", "column"), (
                    t["op"],
                    name,
                )
                assert prop["x-dtk-source"] == "step", (t["op"], name)
                assert prop["x-dtk-dtype"] in ("any", "numeric"), (t["op"], name)
    assert checked >= 30


def test_hints_do_not_change_requiredness_or_bounds():
    schema = transform_schema("drop_columns")
    assert "columns" in schema["required"]
    assert schema["properties"]["columns"]["minItems"] == 1
    assert "default" not in schema["properties"]["columns"]
    assert transform_schema("interactions")["properties"]["columns"]["maxItems"] == 10
    assert transform_schema("pca")["properties"]["columns"]["x-dtk-dtype"] == "numeric"
    assert "default" in transform_schema("pca")["properties"]["columns"]


def test_group_agg_value_is_a_numeric_selector():
    prop = transform_schema("group_agg")["properties"]["value"]
    assert (prop["x-dtk-widget"], prop["x-dtk-dtype"]) == ("column", "numeric")
