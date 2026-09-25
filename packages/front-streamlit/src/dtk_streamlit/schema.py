"""Pure JSON Schema -> params logic, UI-agnostic (widgets are injected).

Kept free of streamlit so the recursion is testable without a running app.
"""

from __future__ import annotations

import json
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


# --- Workspace prefill -------------------------------------------------------

DATASET_KIND = "dataset"


def is_dataset_source(prop: dict, root: dict) -> bool:
    """True when ``prop`` is a source union that accepts ``kind="dataset"``."""
    union = union_members(resolve(prop, root), root)
    return bool(union) and union[0] == "kind" and DATASET_KIND in union[1]


def workspace_defaults(schema: dict, workspace: str | None) -> dict:
    """Top-level defaults pointing the key's sources at the active workspace.

    The first source param reads the train dataset, a source param named
    ``test`` the test dataset; other source params keep their own default.
    Pass the result as ``build_params(..., defaults=...)``.
    """
    if not workspace:
        return {}
    defaults, first = {}, True
    for name, raw in schema.get("properties", {}).items():
        if not is_dataset_source(raw, schema):
            continue
        role = "test" if name == "test" else ("train" if first else None)
        first = False
        if role:
            defaults[name] = {
                "kind": DATASET_KIND,
                "workspace": workspace,
                "role": role,
            }
    return defaults


def _source_label(spec: dict | None) -> str:
    if not spec:
        return "-"
    return spec.get("path") or json.dumps(spec)


def workspace_summary(ws: dict) -> dict[str, str]:
    """Compact, human-readable view of a workspace dict (train / test / steps)."""
    train, test = ws["datasets"]["train"], ws["datasets"].get("test")
    if train.get("target_column"):
        y = f"column {train['target_column']!r} of X"
    elif train.get("y"):
        label = ws.get("label", {})
        how = label.get("mode", "order")
        if how == "key":
            how += f" on {label.get('key')!r}"
        y = f"{_source_label(train['y'])} (join by {how})"
    else:
        y = "none"
    return {
        "train X": _source_label(train["x"]),
        "train y": y,
        "test X": _source_label(test["x"]) if test else "-",
        "steps": str(len(ws.get("steps", []))),
    }


def workspace_from_form(
    name: str,
    x_train: str,
    x_test: str,
    y_train: str = "",
    target_column: str = "",
    label_mode: str = "order",
    label_key: str = "",
    base: dict | None = None,
) -> dict:
    """Workspace dict from the top-bar fields (paths as csv sources).

    ``base`` (the stored workspace) keeps its steps and the non-path options of
    sources whose path is unchanged. Empty strings mean "not set".
    """
    base = base or {}
    old = base.get("datasets", {})

    def src(path: str, previous: dict | None) -> dict | None:
        path = path.strip()
        if not path:
            return None
        if previous and previous.get("path") == path:
            return previous
        return {"kind": "csv", "path": path}

    old_train = old.get("train") or {}
    old_test = old.get("test") or {}
    target = target_column.strip()
    train = {
        "x": src(x_train, old_train.get("x")),
        "y": None if target else src(y_train, old_train.get("y")),
        "target_column": target or None,
    }
    test_x = src(x_test, old_test.get("x"))
    return {
        "name": name.strip(),
        "datasets": {
            "train": train,
            # Test labels are not edited in the top bar: keep the stored ones.
            "test": (
                {"y": None, "target_column": None, **old_test, "x": test_x}
                if test_x
                else None
            ),
        },
        "label": {"mode": label_mode, "key": label_key.strip() or None},
        "steps": base.get("steps", []),
    }
