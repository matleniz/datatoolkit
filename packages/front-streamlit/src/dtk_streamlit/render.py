"""Generic rendering: JSON Schema -> widgets, Result dict -> output."""

import json

import pandas as pd
import plotly.io
import streamlit as st

from dtk_streamlit.schema import build_params


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


class _StreamlitWidgets:
    def field(self, name: str, prop: dict, wkey: str):
        return _widget(name, prop, wkey)

    def choose(self, label: str, options: list[str], index: int, wkey: str) -> str:
        return st.selectbox(label, options, index=index, key=wkey)

    def section(self, label: str) -> None:
        st.markdown(f"**{label}**")


def form_from_schema(schema: dict, key_prefix: str = "") -> dict:
    """Render widgets for every property (recursively); return the params dict."""
    return build_params(schema, _StreamlitWidgets(), prefix=key_prefix)


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
