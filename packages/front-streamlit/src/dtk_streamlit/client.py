"""EngineClient: the only place the front touches the engine."""

from typing import Protocol

from dtk_engine import contract


class EngineClient(Protocol):
    def list_keys(self) -> list[dict]: ...

    def key_schema(self, key_id: str) -> dict: ...

    def run_key(self, key_id: str, params: dict) -> dict: ...


class LocalClient:
    """In-process client calling `dtk_engine.contract` directly."""

    def list_keys(self) -> list[dict]:
        return contract.list_keys()

    def key_schema(self, key_id: str) -> dict:
        return contract.key_schema(key_id)

    def run_key(self, key_id: str, params: dict) -> dict:
        return contract.run_key(key_id, params)
