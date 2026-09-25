"""The engine <-> front boundary. Everything in and out is plain JSON."""

from __future__ import annotations

from pydantic import ValidationError

from . import keys  # noqa: F401  (registers every key)
from .errors import KeyParamsError
from .registry import all_keys, get_key
from .workspace import JsonWorkspaceStore, Workspace


def list_keys() -> list[dict]:
    return [
        {
            "id": k.id,
            "title": k.title,
            "category": k.category,
            "description": k.description,
        }
        for k in all_keys()
    ]


def key_schema(key_id: str) -> dict:
    return get_key(key_id).params_model.model_json_schema()


def run_key(key_id: str, params: dict) -> dict:
    k = get_key(key_id)
    try:
        parsed = k.params_model.model_validate(params)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    return k.run(parsed).model_dump(mode="json")


# Workspaces: stored under $DTK_HOME/workspaces (default ~/.datatoolkit).


def list_workspaces() -> list[dict]:
    store = JsonWorkspaceStore()
    return [store.get(name).model_dump(mode="json") for name in store.list()]


def get_workspace(name: str) -> dict:
    """The workspace dict; unknown name -> raises WorkspaceNotFoundError."""
    return JsonWorkspaceStore().get(name).model_dump(mode="json")


def save_workspace(workspace: dict) -> dict:
    """Validate and store (create or overwrite); returns the normalized dict."""
    try:
        parsed = Workspace.model_validate(workspace)
    except ValidationError as exc:
        raise KeyParamsError(str(exc)) from exc
    JsonWorkspaceStore().save(parsed)
    return parsed.model_dump(mode="json")


def delete_workspace(name: str) -> None:
    JsonWorkspaceStore().delete(name)
