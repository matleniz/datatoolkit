# datatoolkit

Python monorepo (uv workspace): `packages/engine` (`dtk_engine`: keys, registry,
JSON contract) and `packages/front-streamlit` (`dtk_streamlit`: generic front).

- Spec = hub `~/datatoolkit-hub` (read-only from here): `ARCHITECTURE.md` is the
  contract, `STACK.md` the dependency allow-list (add nothing else),
  `HOWTO/add-a-key.md` the recipe. Contradiction → `propose-doc-change`.
- The front never imports `dtk_engine` except `client.py` (ruff TID251 + test).
- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- UI: `uv run streamlit run packages/front-streamlit/src/dtk_streamlit/app.py`
- Gate: `fleet gate` must be clean before any PR.
