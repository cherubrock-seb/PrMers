#!/usr/bin/env bash
# -tune (legacy PRP iterforce tuning) must stop on Ctrl-C, SIGTERM and SIGHUP like every other run.  It never looked
# at the stop request: the search went on through its one-minute measurements, for half an hour or more.
# Each signal, sent during the search, must end the run within 30 s with the stopped exit code (1) and
# "Tuning stopped".
# usage: run_tune_stop.sh <device> [p]
set -euo pipefail
# Job control: a background job of a non-interactive shell would start with SIGINT ignored.
set -m

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${PRMERS_BIN:-$ROOT/prmers}"
DEVICE="${1:-0}"
P="${2:-9941}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0
# check <name> <signal> <marker>: start -tune, send the signal once the log shows the marker, expect a quick stop
check() {
    local name="$1" sig="$2" marker="$3" dir="$WORK/$1" pid rc=0 waited=0
    mkdir -p "$dir"; ln -s "$ROOT/kernels" "$dir/kernels"
    (cd "$dir" && exec "$BIN" "$P" -tune -marin -proof 0 -d "$DEVICE" -noask -f "$dir" > "$dir/run.log" 2>&1) &
    pid=$!
    while ! grep -aq -- "$marker" "$dir/run.log" 2>/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null || [ "$waited" -ge 600 ]; then
            echo "FAIL($name): '$marker' never appeared"; tail -n 5 "$dir/run.log"; kill -9 "$pid" 2>/dev/null || true
            fail=1; return
        fi
        sleep 0.5; waited=$((waited + 1))
    done
    sleep 1
    kill -"$sig" "$pid"
    waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 0.5; waited=$((waited + 1)); done
    if kill -0 "$pid" 2>/dev/null; then
        echo "FAIL($name): still running 30 s after SIG$sig"; kill -9 "$pid"; wait "$pid" 2>/dev/null || true
        fail=1; return
    fi
    wait "$pid" || rc=$?
    if [ "$rc" != 1 ]; then echo "FAIL($name): exit $rc, expected 1 (stopped)"; tail -n 5 "$dir/run.log"; fail=1; fi
    if ! grep -aq "Tuning stopped" "$dir/run.log"; then echo "FAIL($name): no 'Tuning stopped'"; fail=1; fi
    if grep -aq "Optimal iterforce" "$dir/run.log"; then echo "FAIL($name): a tuning result was reported"; fail=1; fi
    echo "case $name: exit $rc"
}

# The search: its first measurement.  The sample before it is 64 iterations, too short to aim a signal at.
check search-int INT "Range: "
check search-term TERM "Range: "
check search-hup HUP "Range: "

if [ "$fail" -ne 0 ]; then echo "Tune stop regression: FAIL"; exit 1; fi
echo "Tune stop regression: PASS"
