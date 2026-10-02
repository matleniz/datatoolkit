# Agent chat protocol (Studio panel ↔ engine)

The in-Studio agent panel (datatoolkit-issues#67) talks to the engine over the
UI bridge. Every route below sits under the `/api/ui` guard: `Authorization:
Bearer <token>` (or `?token=` for `EventSource`), plus the Origin and Host
checks. Code: `src/dtk_engine/agent/chat.py` (hub, adapter interface),
`src/dtk_engine/agent/packs/` (`agent-sdk`, `stub`), routes in `http.py`.

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

Status:

```json
{
  "available": true,
  "pack": "agent-sdk",
  "provider": "Claude (claude.ai login, pro, via the claude CLI)",
  "model": null,
  "running": false,
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "max_tokens": null,
  "reason": "only when available is false: why (extra missing, CLI not logged in, pack off)"
}
```

`usage` holds the session's cumulative totals. `max_tokens` is
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
| `usage` | `input_tokens, output_tokens, total_input_tokens, total_output_tokens` | per turn and cumulative per session; sent once, just before `done` |
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
  text → `assistant_delta "stub: <text>"`, `usage`, `done`.

## Adapter interface (engine side)

```python
class Adapter(Protocol):
    async def start(self, chat: ChatSession) -> None: ...
    async def send(self, text: str) -> str | None: ...   # one turn, returns stop reason
    async def cancel(self) -> None: ...
    async def close(self) -> None: ...
```

`ChatSession` gives the adapter `emit(type, **fields)`, `ask(tool, input,
summary, lines)` (returns allow / deny), `set_turn_usage(input, output)`
(returns True once the cap is reached), `mcp_server()` and `session_tools`. A
new pack is a `Pack(id, provider, model, detect, create)` registered in
`dtk_engine/agent/packs/chat_packs.py`.
