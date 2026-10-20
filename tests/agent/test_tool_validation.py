"""Tool arguments are checked against each tool's input_schema before the handler runs."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from mcp import Client

from dtk_engine.agent import server as server_mod
from dtk_engine.agent.policy import AuditLog
from dtk_engine.agent.ports import LocalUiPort
from dtk_engine.agent.server import build_server
from dtk_engine.agent.tools import build_tools
from dtk_engine.agent.validate import (
    check_arguments,
    strict_schema,
    unsupported_keywords,
)
from dtk_engine.errors import KeyParamsError
from dtk_engine.ui_bridge import UiBridge

pytestmark = pytest.mark.anyio

SPECS = {s.name: s for s in build_tools(LocalUiPort(UiBridge("t")))}
_OVERRIDES = {"memory_id": "m1", "name": "x_1"}
_WRONG = {
    "string": 123, "integer": "x", "number": "x", "boolean": "x", "array": "x", "object": "x",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def calls(monkeypatch):
    """Names of the tools whose handler ran (handlers are replaced by a spy)."""
    seen: list[str] = []

    def spied(port):
        def wrap(spec):
            async def handler(args):
                seen.append(spec.name)
                # a conforming frame where the tool declares an output schema
                return {"identity": None, "data": {}} if spec.output_schema else {}

            return replace(spec, handler=handler)

        return [wrap(s) for s in build_tools(port)]

    monkeypatch.setattr(server_mod, "build_tools", spied)
    return seen


@pytest.fixture
async def client(calls):
    audit = AuditLog()
    async with Client(build_server(LocalUiPort(UiBridge("t")), audit)) as c:
        c.audit = audit
        yield c


def _first(kind):
    return kind[0] if isinstance(kind, list) else kind


def _array(schema):
    return [_sample(schema.get("items", {"type": "string"}))] * schema.get("minItems", 0)


def _object(schema):
    if not schema.get("minProperties"):
        return {}
    props = schema.get("properties")
    return {k: _sample(v, k) for k, v in list(props.items())[:1]} if props else {"a": 1}


_BY_TYPE = {
    "string": lambda s: "x",
    "integer": lambda s: max(s.get("minimum", 1), 1),
    "number": lambda s: max(s.get("minimum", 1), 1),
    "boolean": lambda s: True,
    "array": _array,
    "object": _object,
}


def _sample(schema: dict, field: str = ""):
    """A value that satisfies ``schema`` (the subset our tools use)."""
    kind = _first(schema.get("type", "string"))
    if field in _OVERRIDES and kind == "string":
        return _OVERRIDES[field]
    if "enum" in schema:
        return schema["enum"][0]
    return _BY_TYPE.get(kind, lambda s: None)(schema)


def _valid(schema: dict) -> dict:
    props = schema["properties"]
    return {k: _sample(props[k], k) for k in schema.get("required", [])}


def _body(result):
    return json.loads(result.content[0].text)


def _wrong_field(schema: dict):
    """(field, value) of a property whose type a wrong value can violate."""
    props = schema["properties"]
    for name in [*schema.get("required", []), *props]:
        kind = _first(props[name].get("type", "")) if "enum" not in props[name] else "string"
        if kind in _WRONG and "null" not in (props[name].get("type") or []):
            return name, _WRONG[kind] if "enum" not in props[name] else 123
    return "session", 123  # accepted by every tool, even without a declared property


@pytest.mark.parametrize("tool", sorted(SPECS))
async def test_wrong_type_is_refused(client, calls, tool):
    schema = SPECS[tool].input_schema
    field, bad = _wrong_field(schema)
    result = await client.call_tool(tool, {**_valid(schema), field: bad})
    assert result.is_error
    body = _body(result)
    assert body["type"] == "KeyParamsError"
    assert body["message"].startswith(f"{field}: must be ")
    assert body["details"][0]["loc"] == [field]
    assert calls == []


@pytest.mark.parametrize("tool", sorted(SPECS))
async def test_unknown_key_is_refused(client, calls, tool):
    result = await client.call_tool(tool, {**_valid(SPECS[tool].input_schema), "foo": 1})
    assert result.is_error
    assert "unknown argument 'foo'" in _body(result)["message"]
    assert calls == []


@pytest.mark.parametrize("tool", sorted(t for t, s in SPECS.items() if s.input_schema.get("required")))
async def test_missing_required_is_refused(client, calls, tool):
    required = SPECS[tool].input_schema["required"]
    result = await client.call_tool(tool, {})
    assert result.is_error
    message = _body(result)["message"]
    assert all(f"'{name}'" in message for name in required)
    assert calls == []


@pytest.mark.parametrize("tool", sorted(SPECS))
async def test_valid_minimal_call_reaches_the_handler(client, calls, tool):
    result = await client.call_tool(tool, _valid(SPECS[tool].input_schema))
    assert not result.is_error, result.content
    assert calls == [tool]


@pytest.mark.parametrize("tool", ["run_key", "source_columns", "set_tool_params"])
async def test_params_not_a_dict(client, calls, tool):
    args = {**_valid(SPECS[tool].input_schema)}
    field = "spec" if tool == "source_columns" else "params"
    args[field] = "notadict"
    result = await client.call_tool(tool, args)
    assert result.is_error
    assert _body(result)["message"] == f"{field}: must be object, got string"
    assert calls == []


async def test_out_of_range_names_the_bound(client, calls):
    result = await client.call_tool("get_rows", {"offset": -1})
    assert _body(result)["message"] == "offset: must be >= 0, got -1"
    result = await client.call_tool("get_rows", {"limit": 10_000})
    assert _body(result)["message"] == "limit: must be <= 500, got 10000"
    assert calls == []


async def test_nested_path_and_enum(client, calls):
    result = await client.call_tool("export_workspace", {"formats": ["csv", "xls"]})
    body = _body(result)
    assert body["details"][0]["loc"] == ["formats", 1]
    assert "must be one of" in body["message"]
    assert calls == []


async def test_refusal_is_audited(client, calls):
    await client.call_tool("get_rows", {"offset": -1})
    entry = client.audit.entries()[-1]
    assert (entry["tool"], entry["status"]) == ("get_rows", "error")
    assert entry["error"] == "offset: must be >= 0, got -1"


async def test_existing_permissive_nested_objects_stay_open(client, calls):
    """run_key params and step objects are free-form: extra keys inside them pass."""
    await client.call_tool("run_key", {"key": "k", "params": {"anything": [1, {"a": 2}]}})
    await client.call_tool("preview_step", {"step": {"op": "x", "weird": 1}})
    assert calls == ["run_key", "preview_step"]


@pytest.mark.parametrize("tool", sorted(SPECS))
async def test_session_is_accepted_by_every_tool(client, calls, tool):
    """``tracked`` reads ``session`` on every tool, whether or not its schema lists it."""
    result = await client.call_tool(tool, {**_valid(SPECS[tool].input_schema), "session": "s1"})
    assert not result.is_error, result.content
    assert calls == [tool]


# -- the validator is ours: it must cover every keyword the tool schemas use ----------


@pytest.mark.parametrize("tool", sorted(SPECS))
def test_every_schema_keyword_is_enforced(tool):
    """A tool using a keyword the validator ignores would silently bypass the check."""
    assert unsupported_keywords(strict_schema(SPECS[tool].input_schema)) == set()


def test_unsupported_keywords_are_reported():
    assert unsupported_keywords({"type": "object", "properties": {"a": {"const": 1}}}) == {"const"}


_SAMPLES = [
    ({"type": "string"}, ["a", ""], [1, None, True]),
    ({"type": "integer"}, [1, 2.0], [1.5, "1", True, None]),
    ({"type": "number"}, [1, 1.5], ["1", True]),
    ({"type": "boolean"}, [True], [0, "x"]),
    ({"type": ["integer", "null"]}, [None, 3], ["3"]),
    ({"enum": ["a", "b"]}, ["a"], ["c", 1]),
    ({"type": "integer", "minimum": 0}, [0, 5], [-1]),
    ({"type": "integer", "maximum": 5}, [5], [6]),
    ({"type": "string", "minLength": 1}, ["a"], [""]),
    ({"type": "string", "maxLength": 2}, ["ab"], ["abc"]),
    ({"type": "string", "pattern": "^m[0-9]+$"}, ["m12"], ["x", "m"]),
    ({"type": "array", "items": {"type": "string"}}, [[], ["a"]], [["a", 1], "a"]),
    ({"type": "array", "minItems": 1}, [[1]], [[]]),
    ({"type": "array", "maxItems": 1}, [[1]], [[1, 2]]),
    ({"type": "object", "minProperties": 1}, [{"a": 1}], [{}]),
    ({"type": "object", "required": ["a"]}, [{"a": 1}], [{}]),
    ({"type": "object", "properties": {"a": {"type": "integer"}}}, [{"a": 1}, {"b": 1}],
     [{"a": "x"}]),
    ({"type": "object", "properties": {"a": {}}, "additionalProperties": False}, [{"a": 1}],
     [{"b": 1}]),
    ({"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
     ["a", ["a"]], [1, [1]]),
]


@pytest.mark.parametrize(("schema", "good", "bad"), _SAMPLES, ids=str)
def test_validator_agrees_with_jsonschema(schema, good, bad):
    jsonschema = pytest.importorskip("jsonschema")
    validator = schema
    reference = jsonschema.Draft202012Validator(schema)
    for value in good:
        assert reference.is_valid(value)
        check_arguments(validator, value)
    for value in bad:
        assert not reference.is_valid(value)
        with pytest.raises(KeyParamsError):
            check_arguments(validator, value)
