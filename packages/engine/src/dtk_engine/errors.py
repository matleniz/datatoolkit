"""Engine errors raised across the contract."""


class UnknownKeyError(KeyError):
    """The requested key id is not registered."""


class KeyParamsError(ValueError):
    """The params passed to a key failed validation."""
