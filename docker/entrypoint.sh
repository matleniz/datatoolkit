#!/bin/sh
# Opt-in agent chat (DTK_AGENT=1), then fix /data ownership when a bind mount
# was created by the daemon as root and drop privileges to the dtk user (uid 1000).
set -e

# The image has no agent CLI: the terminal packs are never enabled here.
unset DTK_AGENT_TERMINAL

case "$(printf %s "${DTK_AGENT:-}" | tr '[:upper:]' '[:lower:]')" in
    1 | true | yes | on) agent=1 ;;
    *) agent= ;;
esac

if [ -z "$agent" ]; then
    # DTK_AGENT is the only switch: a stray DTK_AGENT_PACK must not turn the chat on.
    unset DTK_AGENT_PACK
elif [ "${1:-}" = dtk-api ]; then
    if [ -z "${DTK_UI_TOKEN:-}" ]; then
        echo "dtk-entrypoint: DTK_AGENT=1 needs DTK_UI_TOKEN (the bearer token Studio sends);" \
            "e.g. -e DTK_UI_TOKEN=\$(openssl rand -hex 24)" >&2
        exit 64
    fi
    # Chat packs that need no CLI; agent-sdk runs the claude CLI, absent from the image.
    pack=${DTK_AGENT_PACK:-api-anthropic}
    case "$pack" in
        api-anthropic | api-openai | stub) ;;
        *)
            echo "dtk-entrypoint: DTK_AGENT_PACK=$pack is not available in the image" \
                "(api-anthropic, api-openai or stub)" >&2
            exit 64
            ;;
    esac
    shift
    set -- dtk-api "$@" --agent "$pack"
fi

if [ "$(id -u)" = 0 ]; then
    mkdir -p /data
    if [ "$(stat -c %u /data)" != 1000 ]; then
        chown 1000:1000 /data
    fi
    # Only touch sub-entries still owned by root; leave anything else alone.
    find /data -mindepth 1 -user 0 -exec chown 1000:1000 {} +
    exec setpriv --reuid=1000 --regid=1000 --init-groups "$@"
fi

exec "$@"
