from dtk_engine import key_schema, run_key
from dtk_streamlit.schema import (
    build_params,
    is_dataset_source,
    resolve,
    workspace_defaults,
    workspace_from_form,
    workspace_summary,
)


class DefaultWidgets:
    """Answers every widget with its default; records what was asked."""

    def __init__(self, choices=None):
        self.fields, self.sections, self.choices = [], [], choices or {}

    def field(self, name, prop, wkey):
        self.fields.append(wkey)
        return prop.get("default")

    def choose(self, label, options, index, wkey):
        return self.choices.get(wkey, options[index])

    def section(self, label):
        self.sections.append(label)


def _union_schema():
    return {
        "$defs": {
            "A": {
                "type": "object",
                "properties": {
                    "kind": {"const": "a", "type": "string"},
                    "path": {"type": "string"},
                },
            },
            "B": {
                "type": "object",
                "properties": {
                    "kind": {"const": "b", "type": "string"},
                    "url": {"type": "string", "default": "http://x"},
                },
            },
        },
        "type": "object",
        "properties": {
            "src": {
                "oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}],
                "discriminator": {"propertyName": "kind"},
                "default": {"kind": "a", "path": "p.csv"},
            },
            "n": {"type": "integer", "default": 3},
        },
    }


def test_dataset_overview_form_defaults_roundtrip():
    schema = key_schema("dataset_overview")
    widgets = DefaultWidgets()
    params = build_params(schema, widgets, prefix="k")
    src = params["source"]
    assert src["kind"] == "csv"  # default member of the csv | dataset union
    assert src["path"].endswith("train.csv")  # parent default reaches the sub-form
    assert "k:source[csv]:kind" not in widgets.fields  # const: fixed, no widget
    assert "k:source[csv]:path" in widgets.fields
    assert run_key("dataset_overview", params)["metrics"]["rows"] == 41


def test_single_member_union_resolves_to_object():
    schema = _union_schema()
    schema["properties"]["src"]["oneOf"] = [{"$ref": "#/$defs/A"}]
    prop = resolve(schema["properties"]["src"], schema)
    assert "oneOf" not in prop
    assert prop["type"] == "object" and "path" in prop["properties"]


def test_multi_member_union_uses_default_kind():
    params = build_params(_union_schema(), DefaultWidgets())
    assert params == {"src": {"kind": "a", "path": "p.csv"}, "n": 3}


def test_multi_member_union_switch_kind_drops_other_defaults():
    widgets = DefaultWidgets(choices={":src:kind": "b"})
    params = build_params(_union_schema(), widgets)
    assert params["src"] == {"kind": "b", "url": "http://x"}
    assert ":src[b]:url" in widgets.fields


def test_nested_object_subform():
    schema = {
        "type": "object",
        "properties": {
            "opts": {
                "type": "object",
                "title": "Options",
                "properties": {"x": {"type": "number", "default": 1.5}},
            }
        },
    }
    widgets = DefaultWidgets()
    assert build_params(schema, widgets) == {"opts": {"x": 1.5}}
    assert widgets.sections == ["Options"]


def test_is_dataset_source():
    schema = key_schema("train_test_check")
    assert is_dataset_source(schema["properties"]["train"], schema)
    assert not is_dataset_source(schema["properties"]["id_columns"], schema)
    assert not is_dataset_source(_union_schema()["properties"]["src"], _union_schema())


def test_workspace_defaults_roles():
    schema = key_schema("train_test_check")
    assert workspace_defaults(schema, None) == {}
    assert workspace_defaults(schema, "p") == {
        "train": {"kind": "dataset", "workspace": "p", "role": "train"},
        "test": {"kind": "dataset", "workspace": "p", "role": "test"},
    }
    overview = key_schema("dataset_overview")
    assert workspace_defaults(overview, "p") == {
        "source": {"kind": "dataset", "workspace": "p", "role": "train"}
    }


def test_workspace_defaults_second_source_keeps_default():
    schema = key_schema("train_test_check")
    props = schema["properties"]
    schema["properties"] = {"a": props["train"], "b": props["train"]}
    assert list(workspace_defaults(schema, "p")) == ["a"]


def test_workspace_prefill_flows_into_form():
    schema = key_schema("train_test_check")
    widgets = DefaultWidgets()
    params = build_params(schema, widgets, defaults=workspace_defaults(schema, "p"))
    assert params["train"] == {
        "kind": "dataset",
        "workspace": "p",
        "role": "train",
        "labeled": True,
    }
    assert params["test"]["role"] == "test"
    assert ":train[dataset]:workspace" in widgets.fields


def test_workspace_from_form_and_summary():
    ws = workspace_from_form("p", "x.csv", "t.csv", y_train="y.csv")
    assert ws["datasets"]["train"] == {
        "x": {"kind": "csv", "path": "x.csv"},
        "y": {"kind": "csv", "path": "y.csv"},
        "target_column": None,
    }
    assert ws["datasets"]["test"]["x"]["path"] == "t.csv"
    assert ws["label"] == {"mode": "order", "key": None} and ws["steps"] == []
    summary = workspace_summary(ws)
    assert summary == {
        "train X": "x.csv",
        "train y": "y.csv (join by order)",
        "test X": "t.csv",
        "steps": "0",
    }


def test_workspace_from_form_keeps_base():
    base = workspace_from_form("p", "x.csv", "", target_column="t")
    base["datasets"]["train"]["x"]["sep"] = ";"
    base["steps"] = [{"op": "o", "target": "both", "params": {}}]
    ws = workspace_from_form(
        "p", "x.csv", "", y_train="y.csv", label_mode="key", label_key="id", base=base
    )
    assert ws["datasets"]["train"]["x"]["sep"] == ";"  # unchanged path keeps options
    assert ws["datasets"]["train"]["y"]["path"] == "y.csv"
    assert ws["datasets"]["test"] is None
    assert ws["steps"] == base["steps"]
    assert workspace_summary(ws)["train y"] == "y.csv (join by key on 'id')"
    assert workspace_summary(base)["train y"] == "column 't' of X"
    assert workspace_from_form("p", "z.csv", "", base=base)["datasets"]["train"][
        "x"
    ] == {"kind": "csv", "path": "z.csv"}
