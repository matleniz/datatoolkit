"""Transform ops, one module per family. Importing this registers them all.

Each module owns its ops (``@transform`` from ``dtk_engine.transform_registry``);
add ops to the right module, never here.
"""

from . import align, cleaning, encode, features, impute, scale, selection  # noqa: F401
