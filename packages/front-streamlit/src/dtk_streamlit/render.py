"""Generic rendering: JSON Schema -> widgets, Result dict -> output."""

import json

import pandas as pd
import plotly.io
import streamlit as st


def _resolve(prop: dict, schema: dict) -> dict:
    """Resolve a local $ref and unwrap Optional (anyOf with null)."""
    if "$ref" in prop:
        target = schema
        for part in prop["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        prop = {**target, **{k: v for k, v in prop.items() if k != "$ref"}}
    if "anyOf" in prop:
        options = [o for o in prop["anyOf"] if o.get("type") != "null"]
        if len(options) == 1:
            prop = {
                **{k: v for k, v in prop.items() if k != "anyOf"},
                **_resolve(options[0], schema),
            }
    return prop


def _widget(name: str, prop: dict, wkey: str):
    label = prop.get("title", name)
    help_ = prop.get("description")
    default = prop.get("default")
    if "enum" in prop:
        options = prop["enum"]
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, help=help_, key=wkey)
    kind = prop.get("type")
    if kind == "boolean":
        return st.checkbox(label, value=bool(default), help=help_, key=wkey)
    if kind in ("integer", "number"):
        integer = kind == "integer"
        cast = int if integer else float
        lo = prop.get("minimum", prop.get("exclusiveMinimum"))
        hi = prop.get("maximum", prop.get("exclusiveMaximum"))
        return st.number_input(
            label,
            min_value=cast(lo) if lo is not None else None,
            max_value=cast(hi) if hi is not None else None,
            value=cast(default) if default is not None else None,
            step=1 if integer else None,
            help=help_,
            key=wkey,
        )
    if kind == "string":
        return st.text_input(label, value=default or "", help=help_, key=wkey)
    raw = st.text_input(
        f"{label} (JSON)",
        value=json.dumps(default) if default is not None else "",
        help=help_,
        key=wkey,
    )
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.error(f"{label}: invalid JSON")
        return None


def form_from_schema(schema: dict, key_prefix: str = "") -> dict:
    """Render one widget per property; return the params dict."""
    params = {}
    for name, prop in schema.get("properties", {}).items():
        value = _widget(name, _resolve(prop, schema), f"{key_prefix}:{name}")
        if value is not None:
            params[name] = value
    return params


def result(res: dict) -> None:
    """Show a Result dict: metrics, tables, figures, text."""
    metrics = res.get("metrics", {})
    if metrics:
        for col, (name, value) in zip(
            st.columns(len(metrics)), metrics.items(), strict=True
        ):
            col.metric(name, value)
    for table in res.get("tables", []):
        st.subheader(table["title"])
        st.dataframe(pd.DataFrame(table["records"]))
    for figure in res.get("figures", []):
        st.subheader(figure["title"])
        st.plotly_chart(plotly.io.from_json(json.dumps(figure["plotly"])))
    if res.get("text"):
        st.markdown(res["text"])
