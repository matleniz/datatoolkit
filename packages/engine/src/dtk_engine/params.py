"""KeyParams: strict base for every key's Params."""

from pydantic import BaseModel, ConfigDict


class KeyParams(BaseModel):
    """Unknown or misspelled params raise instead of being silently ignored."""

    model_config = ConfigDict(extra="forbid")
