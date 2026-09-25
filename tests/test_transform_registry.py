import pytest

from dtk_engine import transform_registry as registry
from dtk_engine.errors import UnknownTransformError
from dtk_engine.transform_registry import TransformParams, get_transform, transform


@pytest.fixture(autouse=True)
def empty_registry(monkeypatch):
    monkeypatch.setattr(registry, "_TRANSFORMS", {})


def test_defaults_from_op_and_docstring():
    @transform("fill_it", params_model=TransformParams)
    def fill_it(df, params, state):
        """Fill things.

        More detail.
        """
        return df

    t = get_transform("fill_it")
    assert (t.title, t.description) == ("Fill it", "Fill things.")
    assert t.fit(None, t.parse({})) == {}


def test_duplicate_and_bad_model():
    transform("x", params_model=TransformParams)(lambda df, p, s: df)
    with pytest.raises(ValueError, match="duplicate"):
        transform("x", params_model=TransformParams)(lambda df, p, s: df)
    with pytest.raises(TypeError, match="pydantic model"):
        transform("y", params_model=dict)


def test_state_must_be_json_safe():
    transform("bad", params_model=TransformParams, fit=lambda df, p: {"o": object()})(
        lambda df, p, s: df
    )
    t = get_transform("bad")
    with pytest.raises(TypeError, match="JSON-safe"):
        t.fit(None, t.parse({}))


def test_unknown():
    with pytest.raises(UnknownTransformError, match="unknown transform op 'nope'"):
        get_transform("nope")
