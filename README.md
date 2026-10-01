# datatoolkit

Engine of "keys" (typed analyses returning JSON), usable on its own from a
notebook or a script. Spec lives in `~/datatoolkit-hub`. The web front (Studio)
lives in its own repo, [datatoolkit-web](https://github.com/matleniz/datatoolkit-web),
and talks to the engine over the HTTP API (`dtk-api`) only.

Install (engine only, no front):

```bash
uv pip install git+https://github.com/matleniz/datatoolkit
```

Docker (HTTP API, image `ghcr.io/matleniz/datatoolkit-engine`): workspaces and
uploads are stored under `DTK_HOME=/data`, so mount a volume there. The
entrypoint fixes ownership of `/data` (e.g. a bind mount auto-created by the
daemon as root) and runs the engine as uid 1000, so no pre-creation is needed:

```bash
docker build -t dtk-engine .
docker run -d -p 127.0.0.1:8765:8765 -v dtk-data:/data dtk-engine
curl localhost:8765/api/keys
```

Notebook (`dtk_engine.api`: DataFrame in, `Result` or DataFrame out; a
`Result` renders itself in Jupyter):

```python
from dtk_engine import api

train, test = api.load("train.csv"), api.load("test.csv")  # or a source spec dict
api.overview(train)  # dataset_overview on a DataFrame
api.check(train, test)  # train_test_check
api.list_transforms()  # registered transform ops
api.transform(train, "drop_columns", columns=["Name"])  # DataFrame out
```

scikit-learn: every transform op is a fit / transform estimator (fitted on the
training fold, pandas in / out), and a workspace's `both` steps form a pipeline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from dtk_engine import DtkTransformer, workspace_pipeline

pipe = make_pipeline(
    DtkTransformer("drop_columns", columns=["Name"]), LogisticRegression()
)
cross_val_score(pipe, X, y)
workspace_pipeline("my_workspace")  # unfitted Pipeline from the workspace steps
```

JSON contract (what fronts call, `src/dtk_engine/contract.py`): `run_key`,
`list_keys`, `key_schema`, `list_transforms`, `transform_schema`, workspaces
(`list_workspaces`, `get_workspace`, `save_workspace`, rename / duplicate /
delete, `export_workspace`) and the studio reads (`source_columns`,
`preview_workspace`, `preview_step`, `workspace_rows`, `column_profiles`,
`align_report`):

```python
from dtk_engine import run_key, Result

Result(**run_key("train_test_check", {})).show()
```

Omitted params take each key's defaults, and the default `source` / `test` are
the demo CSVs shipped in `dtk_engine/demo_data` (synthetic, Titanic-like): the
call above analyses those files, not your data. Always pass a `source` spec.

`list_keys()` / `list_transforms()` flag `needs_target` (the key or op requires
a target column). A `Result` table with `kind: "steps"` holds workspace steps
(`op`, `target`, `params` per row) that a front can apply as is.

Add a transform op: pick its family module in `src/dtk_engine/ops/transforms/`
(`cleaning`, `impute`, `encode`, `scale`, `features`, `selection`, `formula`,
`align`) and register it with
`@transform(op, params_model=..., fit=...)` (see `drop_columns` in
`cleaning.py` and the protocol in `transform_registry.py`).

HTTP API (optional extra `api`, serves the JSON contract for the web front):

```bash
uv sync --extra api
uv run dtk-api                 # http://127.0.0.1:8765/api/...
uv run dtk-api --host 0.0.0.0 --port 8765
```

CORS allows the Vite dev origins (`http://localhost:5173`,
`http://127.0.0.1:5173`); add more via `DTK_CORS_ORIGINS` (comma-separated).
Uploads land under `$DTK_UPLOAD_DIR` (default `$DTK_HOME/uploads`), workspaces
under `$DTK_HOME/workspaces`; `DTK_HOME` defaults to `~/.datatoolkit`.

Develop:

```bash
uv sync --extra api
uv run pytest
uv run ruff check .
```
