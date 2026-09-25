"""Streamlit entry point. Generic: no key-specific code."""

import streamlit as st

from dtk_streamlit import render
from dtk_streamlit.client import EngineClient, LocalClient
from dtk_streamlit.schema import (
    workspace_defaults,
    workspace_from_form,
    workspace_summary,
)

ACTIVE = "active_workspace"
NONE, NEW = "(none)", "+ new workspace"


def _workspace_form(client: EngineClient, ws: dict | None) -> None:
    """Create (``ws`` None) or edit a workspace; save through the contract."""
    ws = ws or {}
    train = (ws.get("datasets") or {}).get("train") or {}
    test = (ws.get("datasets") or {}).get("test") or {}
    label = ws.get("label") or {}
    with st.form(f"workspace-form:{ws.get('name', NEW)}"):
        name = ws.get("name") or st.text_input("Name", placeholder="parkinson")
        x_train = st.text_input("X_train path", (train.get("x") or {}).get("path", ""))
        y_in_x = st.checkbox(
            "y already in X_train", value=bool(train.get("target_column"))
        )
        c1, c2 = st.columns(2)
        y_train = c1.text_input(
            "y_train path",
            (train.get("y") or {}).get("path", ""),
            help="Ignored when y is already in X_train",
        )
        target = c2.text_input(
            "Target column (in X_train)",
            train.get("target_column") or "",
            help="Used when y is already in X_train",
        )
        x_test = st.text_input("X_test path", (test.get("x") or {}).get("path", ""))
        c3, c4 = st.columns(2)
        modes = ["order", "key"]
        mode = c3.radio(
            "Label join",
            modes,
            index=modes.index(label.get("mode", "order")),
            horizontal=True,
        )
        join_key = c4.text_input(
            "Key column",
            label.get("key") or "",
            help='Required for "key"; for "order", an optional id column checked',
        )
        if not st.form_submit_button("Save workspace"):
            return
    new = workspace_from_form(
        name,
        x_train,
        x_test,
        y_train="" if y_in_x else y_train,
        target_column=target if y_in_x else "",
        label_mode=mode,
        label_key=join_key,
        base=ws,
    )
    try:
        saved = client.save_workspace(new)
    except Exception as exc:  # noqa: BLE001 - surface any engine error in the UI
        st.error(str(exc))
        return
    st.session_state[ACTIVE] = saved["name"]
    st.rerun()


def workspace_bar(client: EngineClient) -> str | None:
    """Top area: pick / create / edit the active workspace; return its name."""
    try:
        workspaces = {w["name"]: w for w in client.list_workspaces()}
    except Exception as exc:  # noqa: BLE001
        st.error(f"cannot list workspaces: {exc}")
        return None
    options = [NONE, *workspaces, NEW]
    active = st.session_state.get(ACTIVE)
    index = options.index(active) if active in workspaces else 0
    col_pick, col_summary = st.columns([1, 3])
    choice = col_pick.selectbox("Workspace", options, index=index)
    if choice == NEW:
        st.session_state[ACTIVE] = None
        with st.expander("New workspace", expanded=True):
            _workspace_form(client, None)
        return None
    if choice == NONE:
        st.session_state[ACTIVE] = None
        col_summary.caption("No workspace: key forms use their demo defaults.")
        return None

    st.session_state[ACTIVE] = choice
    ws = workspaces[choice]
    col_summary.markdown(
        " · ".join(f"**{k}** `{v}`" for k, v in workspace_summary(ws).items())
    )
    with st.expander("Edit workspace"):
        _workspace_form(client, ws)
        if st.button("Delete workspace", key=f"delete:{choice}"):
            client.delete_workspace(choice)
            st.session_state[ACTIVE] = None
            st.rerun()
    return choice


def main(client: EngineClient | None = None) -> None:
    client = client or LocalClient()
    st.set_page_config(page_title="datatoolkit", layout="wide")
    keys = client.list_keys()
    if not keys:
        st.info("No keys registered.")
        return

    keys = sorted(keys, key=lambda k: (k["category"], k["title"]))
    labels = {k["id"]: f"{k['category']} / {k['title']}" for k in keys}
    with st.sidebar:
        st.title("datatoolkit")
        choice = st.radio("Keys", [k["id"] for k in keys], format_func=labels.get)

    active = workspace_bar(client)
    st.divider()
    chosen = next(k for k in keys if k["id"] == choice)
    st.header(chosen["title"])
    st.write(chosen["description"])
    schema = client.key_schema(choice)
    params = render.form_from_schema(
        schema,
        # The workspace in the prefix resets the widgets when it changes.
        key_prefix=f"{choice}@{active or ''}",
        defaults=workspace_defaults(schema, active),
    )
    if st.button("Run"):
        try:
            render.result(client.run_key(choice, params))
        except Exception as exc:  # noqa: BLE001 - surface any engine error in the UI
            st.error(str(exc))


main()
