#!/bin/sh
# Fix /data ownership when a bind mount was created by the daemon as root,
# then drop privileges to the dtk user (uid 1000).
set -e

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
