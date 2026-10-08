#!/usr/bin/env bash
# When the Prime95 stage-2 handoff is interrupted, Prime95 keeps its own stage-2
# progress in its state file (m<p>).  The rerun must hand that file back to
# Prime95, not rewrite it from the stage-1 residue (which restarts stage 2 at
# B1).  A stub "mprime" stands in for Prime95: the first time it records
# progress in the state file and waits for SIGTERM; when it finds that progress
# it finishes by copying a canned result.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

mkdir -p "$WORK/p95"
printf '%s\n' '{"status":"NF", "exponent":113, "worktype":"P-1", "b1":4, "b2":2141, "d":30}' >"$WORK/p95/canned.json"
cat >"$WORK/p95/mprime" <<'EOF'
#!/bin/sh
if grep -aq PRIME95_PROGRESS m0000113 2>/dev/null; then
  cp canned.json results.json.txt
  exit 0
fi
echo PRIME95_PROGRESS >> m0000113
trap 'exit 0' TERM INT
while :; do sleep 0.1; done
EOF
chmod +x "$WORK/p95/mprime"
ln -s "$ROOT/kernels" "$WORK/kernels"
cd "$WORK"

# 1. The handoff starts; interrupt it once the stub has recorded its progress.
"$ROOT/prmers" 113 -pm1 -b1 4 -b2 2141 -p95path "$WORK/p95" -d "$DEVICE" --noask >first.log 2>&1 &
pid=$!
for _ in $(seq 1 400); do
  grep -aq PRIME95_PROGRESS p95/m0000113 2>/dev/null && break
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.25
done
grep -aq PRIME95_PROGRESS p95/m0000113 2>/dev/null || { kill -KILL "$pid" 2>/dev/null || true; echo "first run: the stub never ran" >&2; exit 1; }
sleep 1
kill -INT "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || break
  sleep 0.1
done
if kill -0 "$pid" 2>/dev/null; then
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo "first run: prmers did not stop within 30 s of SIGINT" >&2; exit 1
fi
wait "$pid" 2>/dev/null || true
ls pm1_m_113.ckpt >/dev/null 2>&1 || { echo "first run: the stage-1 checkpoint was deleted" >&2; exit 1; }

# 2. Rerun: Prime95 must find its progress and finish.
timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" 113 -pm1 -b1 4 -b2 2141 -p95path "$WORK/p95" -d "$DEVICE" --noask >second.log 2>&1 || true
grep -aq 'keeping the interrupted state file' second.log || { echo "rerun rewrote Prime95's state file" >&2; exit 1; }
grep -aq 'No factor P-1 (stage 2) until B2 = 2141' second.log || { echo "rerun: Prime95 did not get to finish stage 2" >&2; exit 1; }
[ ! -e p95/m0000113.prmers ] || { echo "the handoff marker was left behind" >&2; exit 1; }
echo "pm1 Prime95 interrupted-state test passed"
