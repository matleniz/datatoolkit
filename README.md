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

Sources: csv/tsv, parquet, `.xlsx`, json/jsonl, sql. Legacy `.xls` is not
supported (no extra dependency): loading one raises a `SourceError` asking for
`.xlsx`; re-save the file as `.xlsx`.

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
`transform_schema(op)` params may carry `x-dtk-when` (`{"strategy": "formula"}`:
the param only applies for that sibling value, or one of a list, see
`params.py`) and `x-dtk-semantic` (`"group_id"`: prefill with the column of that
semantic type; `workspace_rows` column meta carries each column's `semantic`).

`impute` with `strategy="formula"` fills one numeric column from an expression
of other columns (the `formula` op's language, no `@variables`), only on its
missing rows; nothing is learned, so train and test are filled the same way:

```python
from dtk_engine import api
api.transform(df, "impute", columns=["age_at_diagnosis"], strategy="formula",
              expr="age - years_since_diagnosis")
```

Entity-aware fills (rows of one entity, e.g. a patient's visits): the formula
language has `group_mean(x, by=patient_id)`, `group_prev(x, by=..., order=age)`
(last earlier value) and `group_interp(x, by=..., order=age)` (linear between
the entity's neighbours), usable in `formula` and `impute(strategy="formula")`.
`impute` also offers them as strategies (`group_mean` / `group_prev` /
`group_interp` with `by`, `order`, several columns, and an optional train-fitted
`fallback`), and `ffill` takes `by`. Each frame (train, test) is filled from the
entity's own rows in that frame:

```python
from dtk_engine import DtkTransformer
imp = DtkTransformer("impute", columns=["ledd", "on", "off"],
                     strategy="group_interp", by="patient_id", order="age",
                     fallback="median").fit(X_train)
X_test_filled = imp.transform(X_test)
```

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

## Agent (MCP)

Optional extra `agent` (the official `mcp` SDK): the contract as MCP tools
(keys, transforms, workspaces, rows / profiles) plus tools that drive the open
Studio (`propose_steps`, `open_window`, `select_columns`, `set_view`). Reads are
capped and path-scoped (`dtk_engine.agent.policy`).

```bash
uv sync --extra agent
uv run dtk-api        # serves /mcp too; token + url in $DTK_HOME/agent/runtime.json
uv run dtk-mcp        # or: stdio server (UI tools reach the running dtk-api)
```

`/mcp` (and `/mcp/`) takes `Authorization: Bearer <token>` with the same
Host / Origin checks as `/api/ui`. Without the extra, `dtk-api` has no `/mcp`.

### Plug an agent

Run your own agent CLI next to Studio, wired to the `dtk` server only:

```bash
uv sync --extra agent
uv run dtk-api                                          # keep it running, open Studio
uv run dtk-mcp config claude-code --write ~/dtk-agent   # then run the printed command
uv run dtk-mcp doctor                                   # engine up? token valid? Studio? CLIs?
```

Packs: `claude-code`, `gemini`, `opencode` (and `stub`, scripted, for tests).
Without `--write` the files and the command are printed; `--write` never
overwrites an existing file unless `--force`. The config launches the stdio
server with this env's interpreter: no token in any file, it survives
`dtk-api` restarts. `--http` points at the running `/mcp` with this run's
token instead (needs a running engine; valid until it stops).

| pack | file | launch | turned off |
| --- | --- | --- | --- |
| `claude-code` | `dtk.mcp.json` | `claude --strict-mcp-config --mcp-config …/dtk.mcp.json --tools '' --allowedTools 'mcp__dtk__*'` | every built-in tool, every other MCP server |
| `gemini` | `.gemini/settings.json` | `cd <dir> && gemini --skip-trust --allowed-mcp-server-names dtk` | shell, file read / write / edit, web, subagents, skills (`tools.exclude`); other MCP servers |
| `opencode` | `opencode.json` | `cd <dir> && opencode` | every built-in tool and other MCP servers' tools (`permission: {"*": "deny", "dtk_*": "allow"}`) |

What a CLI cannot turn off (its own settings, memory files, hooks, model
choice) is your own session. Data egress: rows and profiles the agent reads
go to that CLI's model provider, capped at 50 rows per call by default (500
max). Destructive edits to a workspace still wait for your review in Studio.

### Agent chat in Studio (`agent-sdk` pack)

Extra `agent-sdk` adds a chat with Claude to Studio's agent panel. The engine
runs the agent loop through `claude-agent-sdk`, which drives the Claude Code
CLI as a subprocess. The agent gets **only** the dtk MCP tools above: built-in
tools are off and settings, CLAUDE.md and the user's other MCP servers are not
loaded. Destructive steps still wait for your review in Studio. Protocol:
[`docs/agent-chat-protocol.md`](docs/agent-chat-protocol.md).

```bash
uv sync --extra agent-sdk
uv run dtk-api --agent          # = DTK_AGENT_PACK=agent-sdk; then Studio: npm run dev
# or Studio + engine in one process (Studio build with the agent panel):
DTK_AGENT_PACK=agent-sdk uvx --from "git+https://github.com/matleniz/datatoolkit-web#subdirectory=launcher" \
  --with "dtk-engine[agent-sdk] @ git+https://github.com/matleniz/datatoolkit" dtk-studio
```

Auth: the CLI's own. DTK never reads or stores a credential. The CLI is
`DTK_AGENT_CLI`, else `claude` on PATH, else the one bundled in the SDK wheel;
all of them use the same `~/.claude` login. With `ANTHROPIC_API_KEY` (or
Bedrock / Vertex / Foundry env) set, that is used. Otherwise it is your own
`claude` login (`claude` then `/login`). `GET /api/ui/agent` says which one
(`claude auth status`, cached). Anthropic's terms
(<https://code.claude.com/docs/en/legal-and-compliance>, checked 2026-10-03)
allow a user to sign in to the unmodified Claude Code binary with **their
own** subscription. Agent SDK use on a Pro / Max plan counts against that
plan's usage limits
(<https://support.claude.com/en/articles/15036540>). Developers must not offer
claude.ai login to *other* users of their product or route their requests
through it. So: your own machine and your own login are fine. Anyone you ship
this to brings their own login or `ANTHROPIC_API_KEY`.

Env: `DTK_AGENT_PACK` (`agent-sdk`, `stub`; unset = off),
`DTK_AGENT_MAX_TOKENS` (per Studio session, input + output, cache reads
included; stops the turn), `DTK_AGENT_MODEL` (default: the CLI's),
`DTK_AGENT_MAX_TURNS` (default 25 tool round trips per message),
`DTK_AGENT_CLI`. The `stub` pack (`dtk-api --agent stub`) is scripted and
makes no network calls; Studio's e2e tests use it.

Develop:

```bash
uv sync --extra api --extra agent
uv run pytest
uv run ruff check .
```

`ruff` also enforces complexity (`C90` max 10, `PLR09xx`), `SIM`, `PERF` and
`B`. `tests/test_layers.py` enforces the layer contract: http → contract →
api | pipeline → workspace → keys → ops → sources (a module may import its own
layer or a lower one, never a higher one).
