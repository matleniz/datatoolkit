"""Streamlit entry point. Generic: no key-specific code."""

import streamlit as st

from dtk_streamlit import render
from dtk_streamlit.client import EngineClient, LocalClient


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

    chosen = next(k for k in keys if k["id"] == choice)
    st.header(chosen["title"])
    st.write(chosen["description"])
    params = render.form_from_schema(client.key_schema(choice), key_prefix=choice)
    if st.button("Run"):
        try:
            render.result(client.run_key(choice, params))
        except Exception as exc:  # noqa: BLE001 - surface any engine error in the UI
            st.error(str(exc))


main()
