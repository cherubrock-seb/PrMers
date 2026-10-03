#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-44497}"
B=10000
WORK="$(mktemp -d)"

cleanup() {
chmod 755 "$WORK" 2>/dev/null || true
rm -rf "$WORK"
}
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels

saved=""

for delay in 2.5 3.0 3.5 4.0 4.5 5.0 6.0; do
rm -f llsafe2_m_*.ckpt*
rm -f interrupt.log

set +e
env AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off timeout -s INT "$delay" "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$WORK" > interrupt.log 2>&1
set -e

saved="$(sed -n 's/.*state saved at iteration \([0-9][0-9]*\).*/\1/p' interrupt.log | tail -n 1)"

if [ -z "$saved" ]; then continue; fi
if [ "$saved" = "0" ]; then saved=""; continue; fi
if [ $((saved % B)) -eq 0 ]; then saved=""; continue; fi
break

done

if [ -z "$saved" ]; then
echo "FAIL: no intra-block checkpoint"
cat interrupt.log
exit 1
fi

[ -f "llsafe2_m_$P.ckpt" ] || {
echo "FAIL: checkpoint missing after interrupt"
exit 1
}

echo "CHECKPOINT | saved=$saved remainder=$((saved % B))"

mkdir results.txt

set +e
env AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off timeout 180 "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$WORK" > failure.log 2>&1
failure_rc=$?
set -e

echo "FAILURE_RUN_RC=$failure_rc"

grep -q 'Resuming from a checkpoint' failure.log
grep -q 'Cannot open .*results.txt for appending' failure.log
grep -q 'Result persistence failed; keeping checkpoint for recovery' failure.log

[ -f "llsafe2_m_$P.ckpt" ] || {
echo "FAIL: checkpoint destroyed after persistence failure"
cat failure.log
exit 1
}

[ -s "${P}_llsafe2_result.json" ] || {
echo "FAIL: individual JSON missing after partial persistence"
exit 1
}

echo "SAVE_FAILURE_RETENTION=PASS"

rmdir results.txt
rm -f "${P}_llsafe2_result.json"

set +e
env AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off timeout 180 "$ROOT/prmers" "$P" -llsafe2 -llsafeb "$B" -engine-marin -d "$DEVICE" -noask -f "$WORK" > success.log 2>&1
success_rc=$?
set -e

echo "SUCCESS_RUN_RC=$success_rc"

if [ "$success_rc" -ne 0 ]; then
cat success.log
exit 1
fi

grep -q 'Resuming from a checkpoint' success.log
grep -q "M$P is prime" success.log

[ -s results.txt ]
[ -s "${P}_llsafe2_result.json" ]

grep -q '"status":"P"' results.txt
grep -q '"worktype":"LL"' results.txt
grep -q '"res64":"0000000000000000"' results.txt

if find . -maxdepth 1 -type f -name 'llsafe2_m_*.ckpt*' | grep -q .; then
echo "FAIL: stale checkpoint after durable save"
exit 1
fi

echo "DURABLE_SAVE_CLEANUP=PASS"
