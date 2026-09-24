from dtk_engine import key_schema, run_key
from dtk_streamlit.schema import build_params, resolve


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
    assert src["kind"] == "csv"  # const: fixed, no widget
    assert src["path"].endswith("train.csv")  # parent default reaches the sub-form
    assert "k:source:kind" not in widgets.fields
    assert "k:source:path" in widgets.fields
    assert run_key("dataset_overview", params)["metrics"]["rows"] == 41


def test_single_member_union_resolves_to_object():
    schema = key_schema("dataset_overview")
    prop = resolve(schema["properties"]["source"], schema)
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
