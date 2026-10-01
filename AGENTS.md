# datatoolkit

Python package `dtk-engine`, usable standalone (notebook, script). Surface at
a glance (`src/dtk_engine`): `contract.py` (the JSON contract fronts call:
keys, transforms, workspaces, export), `http.py` (`dtk-api`: FastAPI routes
under `/api` over the contract, optional extra `api`), `api.py` (notebook door,
DataFrame in), `keys/` (analyses -> `Result`), `ops/` (DataFrame logic;
`ops/transforms/` = fit/apply transform ops), `workspace/` (steps, replay,
parquet export), `pipeline.py` (scikit-learn wrappers), `sources/`. Keys
default to the shipped `demo_data` CSVs when no `source` is given.

No front here: the web front (Studio) lives in `matleniz/datatoolkit-web`
(`~/datatoolkit-web`) and talks to this repo over the HTTP API only (the
`ghcr.io/matleniz/datatoolkit-engine` image in its `compose.yml`). Never add
streamlit (or any UI dependency) to this repo.

- Spec = hub `~/datatoolkit-hub` (read-only from here): `ARCHITECTURE.md` is the
  contract, `STACK.md` the dependency allow-list (add nothing else),
  `HOWTO/add-a-key.md` the recipe. Contradiction → `propose-doc-change`.
- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .` (incl. complexity ≤ 10); layer contract: `tests/test_layers.py`
- Gate: `fleet gate` must be clean before any PR.
