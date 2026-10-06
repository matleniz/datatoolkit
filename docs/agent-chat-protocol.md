# Agent chat protocol (Studio panel ↔ engine)

The in-Studio agent panel (datatoolkit-issues#67) talks to the engine over the
UI bridge. Every route below sits under the `/api/ui` guard: `Authorization:
Bearer <token>` (or `?token=` for `EventSource`), plus the Origin and Host
checks. Code: `src/dtk_engine/agent/chat.py` (hub, adapter interface),
`src/dtk_engine/agent/packs/` (`agent-sdk`, `stub`), routes in `http.py`.
The v2 additions (pack / model choice, API packs, terminal, attachments)
are at the end.

## Transport

| Route | Body | Answer |
|---|---|---|
| `GET /api/ui/agent?session=<sid>` | none | status (below) |
| `POST /api/ui/agent/send` | `{session, text}` | `202 {turn}`; `409 AgentBusy` while a turn runs; `503 NoAgent` (no pack, or it is unusable: `message` says why) |
| `POST /api/ui/agent/cancel` | `{session}` | `200 {cancelled: bool}` (false when idle) |
| `POST /api/ui/agent/permission` | `{session, id, allow}` | `200 {ok: true}`; `404 UnknownPermission` (unknown, answered or expired id) |
| `GET /api/ui/events?session=<sid>` | none | the existing SSE stream; chat events arrive as `event: agent`, `data:` = one JSON event |

Errors use the engine's body `{type, message, details}`. `session` is the
Studio tab id, the same one used for `/api/ui/context` and `/api/ui/events`.
Each Studio session has its own conversation, usage totals and turn counter.
A session's events reach only that session's SSE listeners. An event sent
while nobody listens is dropped (not replayed).

Idle sessions are reaped: 5 minutes (`IDLE_GRACE`) after a Studio session
loses its last `/api/ui/events` listener (tab closed), the engine cancels its
running turn, closes its adapter (for `agent-sdk`, the `claude` CLI process)
and drops its conversation, usage totals and attachments. A reconnect within
the grace (a reload keeps the session id) keeps everything. A reaped session
that comes back starts from zero: new conversation, totals at 0 (so the
`DTK_AGENT_MAX_TOKENS` cap counts again from 0).

Status:

```json
{
  "available": true,
  "pack": "agent-sdk",
  "provider": "Claude (claude.ai login, pro, via the claude CLI)",
  "model": null,
  "running": false,
  "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
  "max_tokens": null,
  "reason": "only when available is false: why (extra missing, CLI not logged in, pack off)"
}
```

`usage` holds the session's cumulative totals (`input_tokens` includes the cache
writes and reads, which are also given apart). `max_tokens` is
`DTK_AGENT_MAX_TOKENS`, or null. `model` null = the CLI's default model.

## Events (`event: agent`)

Every event has `type` and `turn` (`"t1"`, `"t2"`, … per Studio session).

| `type` | Fields | Meaning |
|---|---|---|
| `user_message` | `text` | echo of the sent text, first event of a turn |
| `assistant_delta` | `text` | append to the current assistant message |
| `tool_call` | `id, name, input` | `name` = bare dtk tool name (`propose_steps`, `run_key`, …; no `mcp__dtk__` prefix) |
| `tool_result` | `id, ok, summary?, error?, identity?, pending?, command?` | `id` = its `tool_call`. UI commands carry `command` (bridge command id). A destructive `propose_steps` answers `ok: true, pending: "review"`: Studio's own review banner is the permission ("waiting for your review in Studio"). `identity` = the data frame the tool answered for |
| `permission_request` | `id, tool, input, summary, lines?` | ask the user; answer with `POST /api/ui/agent/permission`. No answer within `DTK_UI_REVIEW_TIMEOUT` (default 900 s) or a cancel = denied. Always comes after the `tool_call` it gates; a deny then gives `tool_result {ok: false, error: "denied"}` |
| `usage` | `input_tokens, output_tokens, uncached_input_tokens, cache_creation_input_tokens, cache_read_input_tokens, context_tokens`, and `total_` + each of the first five | per turn and cumulative per session; sent once, just before `done`. `input_tokens` = uncached + cache writes + cache reads (cache reads bill ~0.1x); `context_tokens` = input of the turn's last API call, i.e. the context size (0 when the pack does not report it) |
| `compacted` | `trigger?, pre_tokens?` | `agent-sdk`: the CLI summarised the older turns (`DTK_AGENT_COMPACT_AT`); `pre_tokens` = context size before. The panel shows a "context compacted" line |
| `error` | `message, code?` | `code`: `max_tokens` (cap reached), `pack_error` (agent / CLI failure), or the provider's error kind |
| `done` | `stop_reason` | last event of every turn: `end_turn`, `cancelled`, `max_tokens`, `max_turns`, `error` |

Turn order: `user_message`, then any mix of `assistant_delta`, `tool_call`,
`tool_result`, `permission_request` and `error`, then `usage`, then `done`.
`usage` and `done` always close a turn, whether it ended normally, was
cancelled, failed or hit the cap.

Input tokens count cache reads and writes as well. They are what the
`DTK_AGENT_MAX_TOKENS` cap is checked against (input + output, cumulative).
The cap is checked during a turn: the agent is interrupted and the turn ends
with `error {code: "max_tokens"}` and then `done {stop_reason: "max_tokens"}`.
Once over the cap, every later send in that session ends the same way at
once.

## Packs

`DTK_AGENT_PACK` picks the pack. `dtk-api --agent [PACK]` sets it (default
`agent-sdk`); unset means no agent.

- `agent-sdk`: Claude through `claude-agent-sdk` and the Claude Code CLI
  (auth = the CLI's own login or `ANTHROPIC_API_KEY`; see the README). Tools =
  the dtk MCP server in-process only. `can_use_tool` denies anything else and
  pins `session` to the sending tab. This pack sends no `permission_request`:
  destructive steps go through Studio's review.
- `stub`: scripted, no network, for tests. Text containing `add a step` →
  `tool_call propose_steps` (`scale` on `age`, `target: both`, workspace and
  `base_identity` taken from the Studio context) through the real bridge, then
  `tool_result`, `assistant_delta`, `usage` (10 / 5 tokens), `done`. Text
  containing `permission` → the same, with a `permission_request` after the
  `tool_call` (deny → `tool_result {ok: false, error: "denied"}`). Any other
  text → `assistant_delta "stub: <text>"`, `usage`, `done`. Text containing
  `attachments` → `tool_call list_attachments` through the real server
  (`session` pinned), `tool_result`, then `assistant_delta "stub: attachments:
  <names>"` (or `none`). Text containing `read the workspace` → `tool_call
  get_workspace` of the open workspace, then `assistant_delta "stub: steps <id>
  <op>, ..."`. Text `remove step <id>` → `tool_call propose_steps` with
  `{remove: {id}}` (`base_steps` filled from the last read), then `stub: removed
  <id>`, `stub: <ack error>` (e.g. `stub: stale: step s1 (scale) removed`) or the
  review wait.

Each turn's text reaches the adapter prefixed by a short Studio note (the
`user_message` event keeps the user's text): workspace, role, viewed version,
identity, then the whole step list (`s3 impute columns=age strategy=median`)
on the first turn and afterwards only the steps added / changed / removed since
the agent last looked, so it need not re-read `get_workspace` each turn.

## Adapter interface (engine side)

```python
class Adapter(Protocol):
    async def start(self, chat: ChatSession) -> None: ...
    async def send(self, text: str) -> str | None: ...   # one turn, returns stop reason
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
```

`ChatSession` gives the adapter `emit(type, **fields)`, `ask(tool, input,
summary, lines)` (returns allow / deny), `set_turn_usage(input, output, *,
cache_write, cache_read, context)` (returns True once the cap is reached; the
cap counts `input + output`), `mcp_server()` and `session_tools`. A
new pack is a `Pack(id, provider, model, detect, create)` registered in
`dtk_engine/agent/packs/chat_packs.py`.

## v2: options, API packs, terminal, attachments

Contract fixed 2026-10-03 (datatoolkit-issues#117), built by #118 (options and
selection), #119 (API packs), #120 (terminal), #121 (attachments). Until a
sub-issue lands, its routes answer `404` (options and selection, #118, have landed). Everything stays under the `/api/ui`
guard. Nothing below changes the v1 routes and events above; v2 only adds
fields and routes.

### Options and per-session selection (#118)

Studio picks the pack (the agent) and its model per session. The env values
(`DTK_AGENT_PACK`, `DTK_AGENT_MODEL`, …) are the defaults a new session
starts with. With the agent off (`DTK_AGENT_PACK` unset), every route below
answers like today (`available: false`, `reason`).

| Route | Body | Answer |
|---|---|---|
| `GET /api/ui/agent/options[?refresh=1]` | none | options (below); `refresh=1` re-runs detection and model discovery |
| `POST /api/ui/agent/config` | `{session, pack, model?}` | `200` = the session's status (as `GET /api/ui/agent`); `409 AgentBusy` while a turn runs; `422 UnknownPack` / `UnknownModel` / `PackUnavailable` (`message` = why) / `WrongPanel` (a terminal pack: open `/api/ui/terminal` instead) |

Options:

```json
{
  "default": {"pack": "agent-sdk", "model": null},
  "packs": [
    {
      "id": "agent-sdk",
      "title": "Claude (Agent SDK, claude CLI)",
      "mode": "cli",
      "panel": "chat",
      "available": true,
      "reason": null,
      "provider": "Claude (claude.ai login, pro, via the claude CLI)",
      "default_model": null,
      "models": [
        {"id": "sonnet", "label": "Sonnet 5.5", "description": "…", "resolved": "claude-sonnet-5-5"}
      ],
      "model_free_text": false
    }
  ]
}
```

- `mode`: `cli` (an agent CLI on this machine does the auth: `agent-sdk`
  and the terminal packs), `api` (direct HTTP with a key / URL from the env:
  `api-anthropic`, `api-openai`), `test` (`stub`).
- `panel`: `chat` (this protocol) or `terminal` (`/api/ui/terminal`, below).
- `models`: what the provider says this user may use, never a hard-coded
  list. `agent-sdk` and `claude-code`: the Claude Code CLI's `initialize`
  answer (`value` → `id`, `displayName` → `label`, `resolvedModel` →
  `resolved`; aliases such as `default`, `sonnet`, `opus` included).
  `api-anthropic`: `GET /v1/models`. `api-openai`: `GET {base}/models`.
  `opencode`: `opencode models`. `gemini`: no list command, so `models: []`
  and `model_free_text: true`. Discovery is cached per run; a failed
  discovery gives `models: []`, `model_free_text: true` and a `models_error`
  string, it never makes the pack unavailable.
- `default_model`: what `model: null` means for this pack (`null` = the
  CLI's or provider's own default).
- `available: false` packs are listed with their `reason` (CLI missing, key
  env var unset, terminal off) so Studio can say what to do.
- Offered packs: every chat pack whose code is installed except `stub`
  (listed only when it is the default pack), plus the terminal packs
  (`claude-code`, `gemini`, `opencode`) whose CLI is on PATH or not, with
  `available: false` and `reason: "terminal off: set DTK_AGENT_TERMINAL=1"`
  unless the terminal is on.

Selection:

- Per Studio session, held in memory like the conversation. A session that
  never posted `config` uses `default`.
- Same pack, other model: the conversation goes on with the new model from
  the next turn (`agent-sdk`: `ClaudeSDKClient.set_model`).
- Other pack: the old adapter is closed and the next turn starts a new
  conversation. The session's usage totals are kept (the
  `DTK_AGENT_MAX_TOKENS` cap is per Studio session, whatever the pack).
- `model` must be one of `models[].id` unless `model_free_text`; `null` or
  omitted = `default_model`.
- After a change the engine sends `event: agent` `{type: "config", turn:
  null, pack, model, reset}` (`reset: true` when the conversation was reset)
  on that session's stream.

Status (`GET /api/ui/agent?session=`) gains `mode`, `panel` and `title`, and
`pack` / `model` / `provider` are the session's choice (or the default).

### API packs (#119)

Mode `api`, panel `chat`: the engine calls the provider over HTTP (`httpx`),
no CLI, no vendor SDK. Same tools, policy and audit as `agent-sdk`: the
in-process dtk MCP server (`ChatSession.mcp_server()`), `session` pinned to
the sending tab, `DTK_AGENT_MAX_TURNS` tool round trips per message,
`DTK_AGENT_MAX_TOKENS` cap, cancel. Credentials come from the environment
only; they never appear in an event, a status, an option or a log.

| Pack | Endpoint | Env |
|---|---|---|
| `api-anthropic` | Anthropic Messages (`POST {base}/v1/messages`, streaming) | `ANTHROPIC_API_KEY` (required), `DTK_ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`), `DTK_ANTHROPIC_MODEL` (default model) |
| `api-openai` | OpenAI-compatible Chat Completions (`POST {base}/chat/completions`, streaming): OpenAI, Ollama, vLLM, LM Studio, … | `DTK_OPENAI_BASE_URL` (required, e.g. `http://127.0.0.1:11434/v1`), `DTK_OPENAI_API_KEY` (optional for a local server), `DTK_OPENAI_MODEL` (default model) |

- `available: false` with a reason when the required env var is unset.
- `provider` names the endpoint host (e.g. `"Anthropic API (api.anthropic.com)"`,
  `"OpenAI-compatible (127.0.0.1:11434)"`) so the panel can state where the
  data goes; a loopback host means the data stays on the machine.
- Without a default model env var: `api-anthropic` uses the first listed
  model whose id contains `sonnet`, else the first listed;
  `api-openai` the first listed.
- Events are the v1 ones: text deltas → `assistant_delta`, tool calls →
  `tool_call` / `tool_result` (bare dtk names), `usage` from the provider's
  usage fields (Anthropic input counts cache reads and writes, as for
  `agent-sdk`; an endpoint that reports no usage gives zeros), provider HTTP
  errors → `error {message, code}` with `code` = the provider's error type,
  or `http_<status>`, or `network`.

### Terminal (#120)

The CLI itself in Studio: an xterm over a WebSocket to a PTY running the
pack's CLI with its dtk-only config (`dtk-mcp config <pack>`: stdio dtk
server, built-in tools off where the CLI allows; what it cannot turn off is
listed by the pack's `tool_policy`). Opt-in: `DTK_AGENT_TERMINAL=1` or
`dtk-api --terminal`; Studio labels it as the weaker guarantee.

`WS /api/ui/terminal?session=<sid>&pack=<id>&model=<m>&cols=<n>&rows=<n>&token=<token>`

- Guard: the `/api/ui` checks (token as `?token=`, a browser cannot set
  headers on a WebSocket; Host); `Origin` is **required** and must be the
  same origin or one of `DTK_CORS_ORIGINS`. A refused connection is accepted
  then closed at once (so the browser sees the code) with `4401` (token), `4403` (Host / Origin / terminal off), `4404` (unknown or
  non-terminal pack), `4409` (this session already has a terminal for that
  pack), `4422` (bad model / size).
- `pack`: `claude-code`, `gemini`, `opencode`. `model` optional (passed as
  the CLI's flag: `claude --model`, `gemini -m`, `opencode -m
  <provider/model>`). `cols` / `rows`: initial size (default 120 x 32).
- Frames, client → engine: **binary** = keystrokes, written to the PTY as
  is; **text** = JSON control `{"type": "resize", "cols": n, "rows": n}`.
- Frames, engine → client: **binary** = PTY output bytes, as is (feed them
  to xterm, no decoding); **text** = JSON control: `{"type": "started",
  "pack", "model", "command": [...]}` once, `{"type": "exit", "code": n}`
  when the CLI exits (then the socket closes with `1000`), `{"type":
  "error", "message"}` (spawn failure, then close `1011`).
- The CLI runs without a shell (argv exec), cwd
  `$DTK_HOME/agent/terminal/<pack>` (config files written there),
  `TERM=xterm-256color`, the user's env otherwise (its own login / keys).
- Stop: closing the socket, or the engine shutting down, sends `SIGHUP` to
  the CLI's process group, then `SIGTERM`, then `SIGKILL` after a 3 s
  grace. One terminal per (session, pack); reopening after a close starts a
  fresh CLI.
- Usage and the token cap do not apply: the CLI bills its own plan.
- First launch: Claude Code asks whether to trust the terminal folder (it
  only holds the dtk config); the user answers in the terminal, the CLI
  remembers it.

### Attachments (#121)

Working files and extra sources the user attaches to the chat, **read-only**
for the agent (decision 2026-10-03: the agent never changes the workspace's
sources, label join or merges, as #86). Upload with the existing
`PUT /api/uploads/{filename}` (raw body → `{path}`), then attach the path:

| Route | Body | Answer |
|---|---|---|
| `POST /api/ui/agent/attachments` | `{session, path}` | `200` attachment (below); `422 NotAnUpload` when `path` is not under the upload dir (`$DTK_UPLOAD_DIR`, else `$DTK_HOME/uploads`) or not a file |
| `GET /api/ui/agent/attachments?session=<sid>` | none | `[attachment, …]` in attach order |
| `DELETE /api/ui/agent/attachments/<id>?session=<sid>` | none | `200 {removed: true}`; `404 UnknownAttachment`. The file stays on disk |

```json
{"id": "a1", "name": "visits.csv", "path": "/…/uploads/1a2b3c4d5e6f/visits.csv",
 "size": 20480, "kind": "table", "columns": ["patient_id", "age", "…"]}
```

- `kind`: `table` (csv / tsv / parquet / xlsx / json / jsonl: read with the
  usual tools and a `source` spec, e.g. `{"kind": "csv", "path": …}`;
  `columns` from `source_columns`, omitted if that fails), `text` (txt, md,
  and other UTF-8 text up to 1 MB), `other` (listed, not readable).
- Attachments are per Studio session, in memory, kept across turns and
  across a pack change; detach removes one Like the conversation, they are dropped
  when the session is reaped (below).
- `POST /api/ui/agent/send` takes optional `attachments: [id, …]` (default:
  none): the turn's `user_message` echoes them (`attachments: [{id, name,
  kind}]`) and the text the adapter gets starts with a short note naming
  them (name, kind, path, columns), marked as data. Unknown ids → `422
  UnknownAttachment`.
- Events on the session's stream: `{type: "attachment_added", turn: null,
  attachment}` and `{type: "attachment_removed", turn: null, id}`.
- MCP tools (chat packs and external / terminal CLIs alike):
  `list_attachments(session?)` and `read_attachment(id, offset?, max_chars?)`
  (`text` kind only, at most `DTK_AGENT_MAX_CHARS` per call, wrapped in the
  policy's data framing). No tool attaches, writes, or turns an attachment
  into a workspace source; tabular attachments are read through the existing
  tools, whose path guard already allows the upload dir.
- `keep_attachment {attachment_id, note?, workspace?}` (a Studio UI command, not
  an attachment tool) keeps an attachment as a **workspace document** (below):
  Studio adds it as one undoable change and acks `{id, ok, document_id}`.

### Workspace documents (datatoolkit-issues#178)

Reference files (data dictionary, protocol, paper) kept **with a workspace**,
across chat sessions: `Workspace.documents: [{id, name, path, mime, size, kind,
added_at, note}]` (`id` `d<n>` filled by the engine; at most 50). `path` is an
upload ref (`PUT /api/uploads/{filename}`), never copied: duplicate / rename
keep the list, deleting a workspace leaves the uploads. `kind`: `text` (UTF-8
up to 1 MB), `pdf` (text extracted with the optional `pdf` extra, pypdf),
`table` (read with the usual tools on its source spec), `other` (listed only).
Every read re-resolves `path` under the upload dir.

| Route | Body / query | Answer |
|---|---|---|
| `POST /api/workspaces/{name}/documents` | `{path, name?, note?}` | `{document, workspace}` (saved) |
| `DELETE /api/workspaces/{name}/documents/{id}` | none | `{workspace}`; the upload stays |
| `GET /api/workspaces/{name}/documents` | none | `[document, …]` |
| `GET /api/workspaces/{name}/documents/{id}/text` | `offset?`, `max_chars?` | `{id, name, kind, offset, text, total_chars, next_offset}` (PDF pages marked `--- page N ---`); `422 DocumentNotReadable` for table / other, a PDF without the extra or an unreadable PDF |
| `GET /api/workspaces/{name}/documents/{id}/file` | none | the raw file (`Content-Type` = `mime`, inline) |
| `POST /api/documents/describe` | `{path, name?}` | an entry without `id`, for a front that edits `documents` itself and `PUT`s the workspace |

A `PUT` of the workspace may also add / edit / remove entries: a path not
already stored must be under the upload dir (`422 NotAnUploadError`).

- MCP tools: `list_documents(workspace?, session?)` (`{id, name, kind, mime,
  size, note}`, plus `source` for a table) and `read_document(id, offset?,
  max_chars?)` (same framing and `DTK_AGENT_MAX_CHARS` cap as
  `read_attachment`; a table answers with its `source` spec).
- A conversation's first turn on a workspace (a new adapter = a new
  conversation) names them after the `[Studio: …]` note: `[Workspace
  documents (read_document): d1 dictionary.md, d2 protocol.pdf]`.
- Export: the manifest lists them (`documents: [{id, name, path, mime, size,
  note, sha256 | missing}]`), files not copied.
