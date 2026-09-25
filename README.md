# datatoolkit

Engine of "keys" (typed analyses returning JSON), usable on its own from a
notebook or a script. Spec lives in `~/datatoolkit-hub`. The Streamlit front
lives in its own repo, [datatoolkit-streamlit](https://github.com/matleniz/datatoolkit-streamlit).

Install (no front, no streamlit):

```bash
uv pip install git+https://github.com/matleniz/datatoolkit
```

Notebook (`dtk_engine.api`: DataFrame in, `Result` or DataFrame out; a
`Result` renders itself in Jupyter):

```python
from dtk_engine import api

train, test = api.load("train.csv"), api.load("test.csv")  # or a source spec dict
api.overview(train)                  # dataset_overview on a DataFrame
api.check(train, test)               # train_test_check
api.list_transforms()                # registered transform ops
api.transform(train, "drop_columns", columns=["Name"])  # DataFrame out
```

scikit-learn: every transform op is a fit / transform estimator (fitted on the
training fold, pandas in / out), and a workspace's `both` steps form a pipeline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from dtk_engine import DtkTransformer, workspace_pipeline

pipe = make_pipeline(DtkTransformer("drop_columns", columns=["Name"]), LogisticRegression())
cross_val_score(pipe, X, y)
workspace_pipeline("my_workspace")   # unfitted Pipeline from the workspace steps
```

JSON contract (what fronts call): `run_key`, `list_keys`, `key_schema`,
workspaces, `list_transforms`, `transform_schema`:

```python
from dtk_engine import run_key, Result

Result(**run_key("train_test_check", {})).show()
```

Add a transform op: pick its family module in `src/dtk_engine/ops/transforms/`
(`cleaning`, `impute`, `encode`, `scale`, `features`) and register it with
`@transform(op, params_model=..., fit=...)` (see `drop_columns` in
`cleaning.py` and the protocol in `transform_registry.py`).

Develop:

```bash
uv sync
uv run pytest
uv run ruff check .
```
