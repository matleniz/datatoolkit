"""Unit tests for concise pydantic -> KeyParamsError helpers."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from dtk_engine.errors import (
    KeyParamsError,
    concise_validation_message,
    key_params_from_validation,
    message_from_validation_details,
    validation_error_details,
)


class _Sample(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    n: int = 1

    @model_validator(mode="after")
    def _check(self) -> _Sample:
        if self.name == "bad":
            raise ValueError("sample: name is bad")
        return self


def test_validation_error_details_and_concise_message():
    with pytest.raises(ValidationError) as caught:
        _Sample.model_validate({"name": "bad"})
    details = validation_error_details(caught.value)
    assert details == [
        {"loc": [], "msg": "sample: name is bad", "type": "value_error"}
    ]
    assert concise_validation_message(caught.value) == "sample: name is bad"
    assert message_from_validation_details(details) == "sample: name is bad"


def test_validation_error_with_loc_joined():
    with pytest.raises(ValidationError) as caught:
        _Sample.model_validate({"__nope__": True})
    msg = concise_validation_message(caught.value)
    assert "name: Field required" in msg
    assert "__nope__: Extra inputs are not permitted" in msg
    assert "; " in msg
    assert "For further information visit" not in msg


def test_key_params_from_validation_prefix_and_details():
    with pytest.raises(ValidationError) as caught:
        _Sample.model_validate({"name": "bad"})
    err = key_params_from_validation(caught.value, prefix="wrap: ")
    assert isinstance(err, KeyParamsError)
    assert str(err) == "wrap: sample: name is bad"
    assert err.details == [
        {"loc": [], "msg": "sample: name is bad", "type": "value_error"}
    ]
