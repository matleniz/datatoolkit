#!/usr/bin/env bash
# Bare-install smoke test (datatoolkit-issues#242): for the bare package and for
# each extra declared in pyproject.toml, create a fresh venv, install the way the
# README tells a new user and run the commands that install enables.
# Fails on any "Traceback" and on an unexpected exit code.
# Usage: scripts/smoke_install.sh [extra ...]   (default: bare + every extra)
# Needs uv and curl; never touches ~/.datatoolkit (temp DTK_HOME).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
export DTK_HOME="$WORK/home"
SERVER_PID=""
cleanup() {
  [ -z "$SERVER_PID" ] || kill "$SERVER_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

# run <expected exit code> <cmd...>: output in $OUT; fails on a traceback or a wrong exit.
run() {
  local want=$1 got=0; shift
  echo "+ $*"
  OUT=$("$@" 2>&1) || got=$?
  head -n 6 <<<"$OUT"
  if grep -q Traceback <<<"$OUT"; then fail "traceback from: $*"; fi
  [ "$got" = "$want" ] || fail "exit $got (wanted $want) from: $*"
}

# engine="duckdb": reads the csv with the extra `large`, else a SourceError naming it.
DUCKDB_CSV='
import sys
from dtk_engine.errors import SourceError
from dtk_engine.sources import CsvSource, load
try:
    print(len(load(CsvSource(path="data.csv", engine="duckdb"))), "rows")
except SourceError as exc:
    print(exc)
    sys.exit(1)
'

# hint <script> <extra>: the script prints the one-line install hint and exits 1.
hint() {
  run 1 "$BIN/$1" --help
  [ "$(wc -l <<<"$OUT")" = 1 ] || fail "$1: hint is not one line"
  grep -q "needs the \"$2\" extra" <<<"$OUT" || fail "$1: no install hint for $2"
}

serve_and_run_key() {
  local port
  port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # Started in $WORK: the relative path of the README curl resolves server-side.
  (cd "$WORK" && exec "$BIN/dtk-api" --port "$port") > "$WORK/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    curl -fsS "localhost:$port/api/keys" > /dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { cat "$WORK/server.log"; fail "dtk-api died"; }
    sleep 0.5
  done
  curl -fsS "localhost:$port/api/keys" | grep -q dataset_overview || fail "dtk-api: no keys"
  # README quickstart: the same key through curl, on a relative CSV path.
  curl -fsS -X POST "localhost:$port/api/keys/dataset_overview/run" \
    -H 'content-type: application/json' \
    -d '{"params": {"source": {"kind": "csv", "path": "data.csv"}}}' > "$WORK/run.json" \
    || { cat "$WORK/server.log"; fail "dtk-api: run dataset_overview"; }
  head -c 200 "$WORK/run.json"; echo
  kill "$SERVER_PID"; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
  if grep -q Traceback "$WORK/server.log"; then cat "$WORK/server.log"; fail "traceback in dtk-api log"; fi
}

check_extra() {
  local extra=$1 spec venv
  venv="$WORK/venv-$extra"
  echo "=== $extra ==="
  uv venv -q "$venv"
  BIN="$venv/bin"
  # The README commands, with the checkout instead of the git URL.
  if [ "$extra" = bare ]; then spec="$ROOT"; else spec="dtk-engine[$extra] @ file://$ROOT"; fi
  uv pip install -q --python "$BIN/python" "$spec"

  # The engine has no `dtk` CLI (README): the script must not exist.
  [ ! -e "$BIN/dtk" ] || fail "unexpected dtk script"

  # README "Run a key on your own CSV" (Python door), on a CSV in the temp dir.
  (cd "$WORK" && run 0 "$BIN/python" -c '
from dtk_engine import Result, key_schema, list_keys, run_key
assert "dataset_overview" in [k["id"] for k in list_keys()]
key_schema("dataset_overview")
Result(**run_key("dataset_overview", {"source": {"kind": "csv", "path": "data.csv"}})).show()
')
  run 0 "$BIN/python" -c "from dtk_engine import api; print(api.list_transforms()[:1])"

  case "$extra" in
    bare | pdf | large) hint dtk-api api; hint dtk-mcp agent ;;
    api) run 0 "$BIN/dtk-api" --help; hint dtk-mcp agent ;;
    agent | agent-sdk) run 0 "$BIN/dtk-api" --help; run 0 "$BIN/dtk-mcp" --help ;;
    *) fail "no smoke checks for extra $extra: add them" ;;
  esac
  case "$extra" in
    api | agent | agent-sdk) serve_and_run_key ;;
  esac
  case "$extra" in
    bare) (cd "$WORK" && run 1 "$BIN/python" -c "$DUCKDB_CSV") ;;
    large) (cd "$WORK" && run 0 "$BIN/python" -c "$DUCKDB_CSV") ;;
    pdf) run 0 "$BIN/python" -c "import pypdf" ;;
    agent) run 0 "$BIN/python" -c "import mcp, httpx, websockets" ;;
    agent-sdk) run 0 "$BIN/python" -c "import claude_agent_sdk, mcp" ;;
  esac
  rm -rf "$venv"
}

printf 'id,age,fare\n1,22,7.25\n2,38,71.3\n3,26,7.9\n4,35,53.1\n' > "$WORK/data.csv"

if [ "$#" -gt 0 ]; then
  extras=("$@")
else
  mapfile -t extras < <(python3 -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print("bare", *tomllib.load(f)["project"]["optional-dependencies"], sep="\n")
' "$ROOT/pyproject.toml")
fi
for e in "${extras[@]}"; do check_extra "$e"; done
echo "SMOKE OK: ${extras[*]}"
