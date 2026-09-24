import json

import pytest
from dtk_engine import key_schema, list_keys, run_key
from dtk_engine.errors import KeyParamsError, UnknownKeyError

KEY_IDS = [k["id"] for k in list_keys()]


def test_keys_registered():
    assert KEY_IDS


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_schema_is_dict(key_id):
    assert isinstance(key_schema(key_id), dict)


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_run_defaults_is_json(key_id):
    assert json.dumps(run_key(key_id, {}))


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_bad_params_raise(key_id):
    with pytest.raises(KeyParamsError):
        run_key(key_id, {"__not_a_param__": object()})


def test_unknown_key_raises():
    with pytest.raises(UnknownKeyError):
        run_key("does-not-exist", {})
    with pytest.raises(UnknownKeyError):
        key_schema("does-not-exist")


@pytest.mark.parametrize("key_id", KEY_IDS)
def test_unknown_param_rejected(key_id):
    with pytest.raises(KeyParamsError):
        run_key(key_id, {"not_a_param": 1})
