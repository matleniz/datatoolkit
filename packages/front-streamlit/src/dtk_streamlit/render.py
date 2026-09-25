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


def form_from_schema(
    schema: dict, key_prefix: str = "", defaults: dict | None = None
) -> dict:
    """Render widgets for every property (recursively); return the params dict.

    ``defaults`` override the schema's top-level defaults (e.g. workspace sources).
    """
    return build_params(
        schema, _StreamlitWidgets(), prefix=key_prefix, defaults=defaults
    )


OVERVIEW_GROUP = "Overview"
FILTER_COLUMN = "column"


def group_items(res: dict) -> dict[str, dict[str, list[dict]]]:
    """Bucket tables / figures by `group`, tabs in first-appearance order.

    Returns `{}` when nothing is grouped (flat layout). Otherwise ungrouped items
    go to a leading "Overview" tab; each value is `{"tables": [...], "figures": [...]}`.
    """
    tables, figures = res.get("tables", []), res.get("figures", [])
    if not any(item.get("group") for item in (*tables, *figures)):
        return {}
    groups: dict[str, dict[str, list[dict]]] = {
        OVERVIEW_GROUP: {"tables": [], "figures": []}
    }
    for kind, items in (("tables", tables), ("figures", figures)):
        for item in items:
            name = item.get("group") or OVERVIEW_GROUP
            groups.setdefault(name, {"tables": [], "figures": []})[kind].append(item)
    # Overview leads; dropped again below if no item landed in it.
    return {name: g for name, g in groups.items() if g["tables"] or g["figures"]}


def filter_options(records: list[dict]) -> list:
    """Distinct values of the `column` field, in first-appearance order ([] if absent)."""
    if not any(FILTER_COLUMN in r for r in records):
        return []
    return list(dict.fromkeys(r[FILTER_COLUMN] for r in records if FILTER_COLUMN in r))


def filter_records(records: list[dict], selected: list) -> list[dict]:
    """Keep the records whose `column` is in `selected` (all of them if empty)."""
    if not selected:
        return records
    return [r for r in records if r.get(FILTER_COLUMN) in selected]


def _show_table(table: dict, scope: str) -> None:
    st.subheader(table["title"])
    records = table["records"]
    options = filter_options(records)
    if len(options) > 1:
        selected = st.multiselect(
            "Filter on column",
            options,
            key=f"filter-{scope}-{table['title']}",
        )
        records = filter_records(records, selected)
    st.dataframe(pd.DataFrame(records))


def _show_figure(figure: dict) -> None:
    st.subheader(figure["title"])
    st.plotly_chart(plotly.io.from_json(json.dumps(figure["plotly"])))


def result(res: dict) -> None:
    """Show a Result dict: metrics, tables, figures (tabs per group), text."""
    metrics = res.get("metrics", {})
    if metrics:
        for col, (name, value) in zip(
            st.columns(len(metrics)), metrics.items(), strict=True
        ):
            col.metric(name, value)
    groups = group_items(res)
    if groups:
        for tab, (name, items) in zip(
            st.tabs(list(groups)), groups.items(), strict=True
        ):
            with tab:
                for table in items["tables"]:
                    _show_table(table, name)
                for figure in items["figures"]:
                    _show_figure(figure)
    else:
        for table in res.get("tables", []):
            _show_table(table, "")
        for figure in res.get("figures", []):
            _show_figure(figure)
    if res.get("text"):
        st.markdown(res["text"])
