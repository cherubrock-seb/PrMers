#!/usr/bin/env bash
# Deterministic production-path test for the opt-in V-trace no-progress
# watchdog. First create a valid non-pair95 Stage-2 checkpoint by interrupting
# M521, then resume it with a test-only stall injected inside the monitored
# linear-prime loop. The watchdog must exit with code 124 and leave every
# checkpoint/backend file byte-for-byte unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export AEVUM_CARRY_WMUL=1
export AEVUM_AUTOTUNE=off

ARGS=(
  521 -pm1 -b1 10 -b2 3000000
  -pm1-vtrace-pair95-off
  -pm1-vtrace-d 6
  -pm1-progress-seconds 1
  -d "$DEVICE" --noask
)

(
  cd "$WORK"
  exec "$ROOT/prmers" "${ARGS[@]}" >first.log 2>&1
) &
pid=$!

seen=0
for _ in $(seq 1 12000); do
  if grep -aq '\[PM1-VTRACE\] V1 = H + H\^-1 built' "$WORK/first.log" 2>/dev/null; then
    seen=1
    break
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    break
  fi
  sleep 0.01
done

if [ "$seen" -ne 1 ]; then
  echo "FAIL: initial run never reached V-trace Stage 2 setup" >&2
  tail -40 "$WORK/first.log" >&2 || true
  kill -INT "$pid" 2>/dev/null || true
  wait "$pid" || true
  exit 1
fi

sleep 0.20
kill -INT "$pid" 2>/dev/null || true
wait "$pid" || true

grep -aq 'Interrupted by user, V-trace Stage 2 state saved at prime' "$WORK/first.log" || {
  echo "FAIL: initial run did not commit a direct V-trace Stage-2 checkpoint" >&2
  tail -60 "$WORK/first.log" >&2
  exit 1
}

CKPT="$(find "$WORK" -maxdepth 1 -type f -name 'pm1_s2_vtrace_*.ckpt' | head -1)"
[ -n "$CKPT" ] && [ -s "$CKPT" ] || {
  echo "FAIL: no durable Stage-2 checkpoint after interrupt" >&2
  exit 1
}

snapshot_hashes() {
  find "$WORK" -maxdepth 1 -type f \
    \( -name 'pm1_s2_vtrace_*.ckpt' -o \
       -name 'pm1_s2_vtrace_*.ckpt.old' -o \
       -name 'pm1_s2_vtrace_*.ckpt.backend' \) \
    -print0 |
    sort -z |
    xargs -0 sha256sum
}

snapshot_hashes >"$WORK/before.sha256"
cp "$WORK/before.sha256" "$WORK/before.saved"

set +e
(
  cd "$WORK"
  PRMERS_TEST_PM1_WATCHDOG_STALL_MS=1500 \
    timeout --kill-after=5s 30 \
    "$ROOT/prmers" "${ARGS[@]}" -pm1-no-progress-seconds 1 \
    >watchdog.log 2>&1
)
rc=$?
set -e

if [ "$rc" -ne 124 ]; then
  echo "FAIL: watchdog run exit code=$rc, expected 124" >&2
  tail -80 "$WORK/watchdog.log" >&2
  exit 1
fi

grep -aq 'Resuming Stage 2 from checkpoint' "$WORK/watchdog.log" || {
  echo "FAIL: watchdog run did not resume the durable checkpoint" >&2
  tail -80 "$WORK/watchdog.log" >&2
  exit 1
}

grep -aq '\[PM1-WATCHDOG\] no V-trace Stage 2 advancement for 1 s; phase=test-stall-linear-prime' "$WORK/watchdog.log" || {
  echo "FAIL: expected watchdog timeout diagnostic is missing" >&2
  tail -80 "$WORK/watchdog.log" >&2
  exit 1
}

snapshot_hashes >"$WORK/after.sha256"
if ! cmp -s "$WORK/before.saved" "$WORK/after.sha256"; then
  echo "FAIL: watchdog timeout modified checkpoint/backend state" >&2
  echo "--- before ---" >&2
  cat "$WORK/before.saved" >&2
  echo "--- after ---" >&2
  cat "$WORK/after.sha256" >&2
  exit 1
fi

if find "$WORK" -maxdepth 1 -type f -name '*stage2*result*.json' | grep -q .; then
  echo "FAIL: watchdog timeout incorrectly published a Stage-2 result" >&2
  find "$WORK" -maxdepth 1 -type f -name '*stage2*result*.json' -print >&2
  exit 1
fi

echo "WATCHDOG_TIMEOUT_EXIT=124"
echo "WATCHDOG_CHECKPOINT_PRESERVED=PASS"
echo "WATCHDOG_RESULT_NOT_PUBLISHED=PASS"
echo "pm1 V-trace no-progress watchdog timeout test passed"
