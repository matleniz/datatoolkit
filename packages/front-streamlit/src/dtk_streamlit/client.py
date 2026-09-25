"""EngineClient: the only place the front touches the engine."""

from typing import Protocol

from dtk_engine import contract


class EngineClient(Protocol):
    def list_keys(self) -> list[dict]: ...

    def key_schema(self, key_id: str) -> dict: ...

    def run_key(self, key_id: str, params: dict) -> dict: ...

    def list_workspaces(self) -> list[dict]: ...

    def get_workspace(self, name: str) -> dict: ...

    def save_workspace(self, workspace: dict) -> dict: ...

    def delete_workspace(self, name: str) -> None: ...


class LocalClient:
    """In-process client calling `dtk_engine.contract` directly."""

    def list_keys(self) -> list[dict]:
        return contract.list_keys()

    def key_schema(self, key_id: str) -> dict:
        return contract.key_schema(key_id)

    def run_key(self, key_id: str, params: dict) -> dict:
        return contract.run_key(key_id, params)

    def list_workspaces(self) -> list[dict]:
        return contract.list_workspaces()

    def get_workspace(self, name: str) -> dict:
        return contract.get_workspace(name)

    def save_workspace(self, workspace: dict) -> dict:
        return contract.save_workspace(workspace)

    def delete_workspace(self, name: str) -> None:
        contract.delete_workspace(name)
