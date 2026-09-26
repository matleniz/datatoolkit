"""Ops: the DataFrame logic under the keys, the api and the workspaces.

- Analysis ops (``profile``, ``missing``, ``outliers``, ``compare/``,
  ``correlation``, ``selection``, ...): DataFrame in, DataFrames / plain values
  out, no ``Result``; the keys (``dtk_engine/keys``) wrap them into a Result.
- Transform ops (``transforms/``): registered with ``@transform`` (pydantic
  params, fit / apply, see ``transform_registry.py``); replayed as workspace
  steps and wrapped by ``DtkTransformer`` for scikit-learn.
- ``advisor/`` sits on both: it emits transform steps and applies the cleaning
  ones (with the real transform ops) to copies of the frames before its checks.
"""
