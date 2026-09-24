"""Pure JSON Schema -> params logic, UI-agnostic (widgets are injected).

Kept free of streamlit so the recursion is testable without a running app.
"""

from __future__ import annotations

from typing import Any, Protocol


class Widgets(Protocol):
    def field(self, name: str, prop: dict, wkey: str) -> Any:
        """Ask for one scalar value; ``prop["default"]`` is the effective default."""

    def choose(self, label: str, options: list[str], index: int, wkey: str) -> str:
        """Pick one option (used for the `kind` of a multi-member union)."""

    def section(self, label: str) -> None:
        """Announce a nested sub-form."""


def resolve(prop: dict, root: dict) -> dict:
    """Resolve a local $ref and unwrap single-member unions (incl. Optional).

    Keys on ``prop`` itself (title, default, description) win over the target's.
    """
    if "$ref" in prop:
        target = root
        for part in prop["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        prop = {
            **resolve(target, root),
            **{k: v for k, v in prop.items() if k != "$ref"},
        }
    for union in ("anyOf", "oneOf"):
        if union not in prop:
            continue
        options = [o for o in prop[union] if o.get("type") != "null"]
        if len(options) == 1:
            outer = {k: v for k, v in prop.items() if k not in (union, "discriminator")}
            prop = {**resolve(options[0], root), **outer}
    return prop


def fixed_value(prop: dict) -> tuple[bool, Any]:
    """(True, value) when the schema admits a single value (`const`, 1-item enum)."""
    if "const" in prop:
        return True, prop["const"]
    if len(prop.get("enum", ())) == 1:
        return True, prop["enum"][0]
    return False, None


def union_members(prop: dict, root: dict) -> tuple[str, dict[str, dict]] | None:
    """For a multi-member union of objects: (discriminator, {tag: member schema})."""
    raw = prop.get("oneOf") or prop.get("anyOf")
    if not raw:
        return None
    tag = prop.get("discriminator", {}).get("propertyName", "kind")
    members = {}
    for option in raw:
        member = resolve(option, root)
        is_fixed, value = fixed_value(
            resolve(member.get("properties", {}).get(tag, {}), root)
        )
        if not is_fixed:
            return None
        members[str(value)] = member
    return (tag, members) if len(members) > 1 else None


def build_params(
    schema: dict,
    widgets: Widgets,
    *,
    root: dict | None = None,
    prefix: str = "",
    defaults: dict | None = None,
) -> dict:
    """Walk an object schema, asking ``widgets`` for values; return the params dict.

    Object properties recurse into a sub-form returning a dict; fixed values
    (`const`) need no widget; a multi-member union asks for the discriminator
    then recurses into the chosen member. ``defaults`` (e.g. the parent field's
    default object) override the sub-schema's own defaults.
    """
    root = root if root is not None else schema
    defaults = defaults or {}
    params = {}
    for name, raw in schema.get("properties", {}).items():
        prop = resolve(raw, root)
        default = defaults.get(name, prop.get("default"))
        value = _value(name, prop, default, widgets, root, f"{prefix}:{name}")
        if value is not None:
            params[name] = value
    return params


def _value(
    name: str, prop: dict, default: Any, widgets: Widgets, root: dict, wkey: str
):
    is_fixed, value = fixed_value(prop)
    if is_fixed:
        return value
    label = prop.get("title", name)
    sub_defaults = default if isinstance(default, dict) else None
    union = union_members(prop, root)
    if union:
        tag, members = union
        options = list(members)
        current = (sub_defaults or {}).get(tag)
        index = options.index(current) if current in options else 0
        choice = widgets.choose(f"{label} {tag}", options, index, f"{wkey}:{tag}")
        widgets.section(label)
        return build_params(
            members[choice],
            widgets,
            root=root,
            prefix=f"{wkey}[{choice}]",
            defaults=sub_defaults if current == choice else None,
        )
    if prop.get("type") == "object" and "properties" in prop:
        widgets.section(label)
        return build_params(
            prop, widgets, root=root, prefix=wkey, defaults=sub_defaults
        )
    return widgets.field(name, {**prop, "default": default}, wkey)
