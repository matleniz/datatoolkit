# datatoolkit

Engine of "keys" (typed analyses returning JSON) + a generic Streamlit front.
Spec lives in `~/datatoolkit-hub`.

```bash
uv sync
uv run pytest                                                            # tests
uv run streamlit run packages/front-streamlit/src/dtk_streamlit/app.py   # UI
```

Notebook:

```python
from dtk_engine import run_key, Result

Result(**run_key("hello", {"n": 10})).show()
```
