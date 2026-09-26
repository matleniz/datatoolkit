# datatoolkit

Python package `dtk-engine`, usable standalone (notebook, script). Surface at
a glance (`src/dtk_engine`): `contract.py` (the JSON contract fronts call:
keys, transforms, workspaces, export), `api.py` (notebook door, DataFrame in),
`keys/` (analyses -> `Result`), `ops/` (DataFrame logic; `ops/transforms/` =
fit/apply transform ops), `workspace/` (steps, replay, parquet export),
`pipeline.py` (scikit-learn wrappers), `sources/`. Keys default to the shipped
`demo_data` CSVs when no `source` is given.

No front here: the Streamlit front lives in `matleniz/datatoolkit-streamlit`
(`~/datatoolkit-streamlit`) and depends on this repo via git. Never add streamlit (or any UI dependency) to this repo.

- Spec = hub `~/datatoolkit-hub` (read-only from here): `ARCHITECTURE.md` is the
  contract, `STACK.md` the dependency allow-list (add nothing else),
  `HOWTO/add-a-key.md` the recipe. Contradiction → `propose-doc-change`.
- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Gate: `fleet gate` must be clean before any PR.
