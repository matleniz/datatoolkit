# datatoolkit

Engine of "keys" (typed analyses returning JSON), usable on its own from a
notebook or a script. Spec lives in `~/datatoolkit-hub`. The Streamlit front
lives in its own repo, [datatoolkit-streamlit](https://github.com/matleniz/datatoolkit-streamlit).

Install (no front, no streamlit):

```bash
uv pip install git+https://github.com/matleniz/datatoolkit
```

Notebook:

```python
from dtk_engine import run_key, Result

Result(**run_key("train_test_check", {})).show()
```

Develop:

```bash
uv sync
uv run pytest
uv run ruff check .
```
