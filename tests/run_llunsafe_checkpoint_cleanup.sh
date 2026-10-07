#!/usr/bin/env bash
# An LL-UNSAFE run keeps its state in llunsafe_m_<p>.ckpt. When it completes it must remove
# that checkpoint and leave m_<p>.ckpt, the PRP checkpoint of the same exponent, alone.
# A Wagstaff run must record its verdict and remove its wagstaff_ checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P="${2:-9689}"
Q="${3:-42737}"
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# Interrupt early to leave an LL-UNSAFE checkpoint on disk.
saved=""
for delay in 2 3 4 5 6; do
    rm -f llunsafe_m_*.ckpt* interrupt.log
    set +e
    $RUN timeout -s INT "$delay" "$ROOT/prmers" "$P" -llunsafe -engine-marin -d "$DEVICE" -noask -f "$WORK" > interrupt.log 2>&1
    set -e
    saved="$(sed -n 's/.*state saved at iteration \([0-9][0-9]*\).*/\1/p' interrupt.log | tail -n 1)"
    if [ -n "$saved" ] && [ "$saved" != "0" ] && [ -f "llunsafe_m_$P.ckpt" ]; then break; fi
    saved=""
done
if [ -z "$saved" ]; then
    echo "FAIL: no LL-UNSAFE checkpoint after interrupt"
    cat interrupt.log
    exit 1
fi
echo "LL_CHECKPOINT | saved=$saved"

# Stand-in for the PRP checkpoint of the same exponent.
echo "unrelated prp checkpoint" > "m_$P.ckpt"

set +e
$RUN timeout 120 "$ROOT/prmers" "$P" -llunsafe -engine-marin -d "$DEVICE" -noask -f "$WORK" > resume.log 2>&1
rc=$?
set -e
echo "RESUME_RC=$rc"
grep -q 'Resuming from a checkpoint' resume.log
grep -q "2^$P - 1 is prime" resume.log

if ls llunsafe_m_"$P".ckpt* >/dev/null 2>&1; then
    echo "FAIL: LL-UNSAFE checkpoint left behind"
    exit 1
fi
if [ ! -f "m_$P.ckpt" ]; then
    echo "FAIL: LL-UNSAFE completion deleted the PRP checkpoint"
    exit 1
fi
echo "LLUNSAFE_CHECKPOINT_CLEANUP=PASS"

# Wagstaff: verdict recorded, checkpoint removed.
saved=""
for delay in 4 5 6 7; do
    rm -f wagstaff_m_*.ckpt* interrupt.log
    set +e
    $RUN timeout -s INT "$delay" "$ROOT/prmers" "$Q" -wagstaff -engine-marin -d "$DEVICE" -noask -f "$WORK" > interrupt.log 2>&1
    set -e
    saved="$(sed -n 's/.*state saved at iteration \([0-9][0-9]*\).*/\1/p' interrupt.log | tail -n 1)"
    if [ -n "$saved" ] && [ "$saved" != "0" ] && [ -f "wagstaff_m_$((2 * Q)).ckpt" ]; then break; fi
    saved=""
done
if [ -z "$saved" ]; then
    echo "FAIL: no Wagstaff checkpoint after interrupt"
    cat interrupt.log
    exit 1
fi
set +e
$RUN timeout 120 "$ROOT/prmers" "$Q" -wagstaff -engine-marin -d "$DEVICE" -noask -f "$WORK" > wagstaff.log 2>&1
rc=$?
set -e
echo "WAGSTAFF_RC=$rc"
grep -q 'Resuming from a checkpoint' wagstaff.log
grep -q 'Wagstaff PRP confirmed' wagstaff.log
[ -s "${Q}_wagstaff_result.json" ]
grep -q '"worktype":"Wagstaff-PRP"' results.txt
grep -q '"status":"P"' results.txt
if ls wagstaff_m_*.ckpt* >/dev/null 2>&1; then
    echo "FAIL: Wagstaff checkpoint left behind"
    exit 1
fi
echo "WAGSTAFF_RESULT=PASS"
