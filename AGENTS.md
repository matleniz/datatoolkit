# datatoolkit

Python package `dtk-engine` (`src/dtk_engine`: keys, registry, JSON contract),
usable standalone (notebook, script). No front here: the Streamlit front lives
in `matleniz/datatoolkit-streamlit` (`~/datatoolkit-streamlit`) and depends on
this repo via git. Never add streamlit (or any UI dependency) to this repo.

- Spec = hub `~/datatoolkit-hub` (read-only from here): `ARCHITECTURE.md` is the
  contract, `STACK.md` the dependency allow-list (add nothing else),
  `HOWTO/add-a-key.md` the recipe. Contradiction → `propose-doc-change`.
- Setup: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Gate: `fleet gate` must be clean before any PR.
