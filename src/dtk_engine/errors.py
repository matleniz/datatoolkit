"""Engine errors raised across the contract."""


class UnknownKeyError(KeyError):
    """The requested key id is not registered."""


class KeyParamsError(ValueError):
    """The params passed to a key failed validation."""


class SourceError(ValueError):
    """A source could not be loaded (missing file, unreadable content, unknown kind)."""


class UnknownTransformError(KeyError):
    """The requested transform op is not registered."""
