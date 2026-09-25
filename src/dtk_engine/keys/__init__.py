"""Import every key module so registration happens on `import dtk_engine`."""

from . import (  # noqa: F401
    dataset_overview,
    duplicates,
    file_inspect,
    inconsistencies,
    train_test_check,
)
