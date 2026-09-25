"""Import every key module so registration happens on `import dtk_engine`."""

from . import (  # noqa: F401
    dataset_overview,
    duplicates,
    feature_selection,
    file_inspect,
    inconsistencies,
    missing_values,
    outliers,
    preprocessing_advisor,
    train_test_check,
)
