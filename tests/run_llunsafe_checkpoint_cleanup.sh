#!/usr/bin/env bash
# An LL-UNSAFE run keeps its state in llunsafe_m_<p>.ckpt. When it completes it must remove
# that checkpoint and leave m_<p>.ckpt, the PRP checkpoint of the same exponent, alone.
# A Wagstaff run must record its verdict and remove its wagstaff_ checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
# With no exponent given, work up through a ladder of exponents (LL: Mersenne
# primes; Wagstaff: Wagstaff PRP exponents) until one runs long enough, on this
# device, to be interrupted part way.
if [ -n "${2:-}" ]; then LL_LADDER=("$2"); else LL_LADDER=(9689 19937 44497 110503 216091 756839); fi
if [ -n "${3:-}" ]; then WAG_LADDER=("$3"); else WAG_LADDER=(10501 14479 42737 83339 138937 267017 374321); fi
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

# interrupt_run <dir> <ckpt> <prmers args...>
# Start prmers in <dir> with a 1 s backup interval and send it SIGINT as soon as
# its first periodic backup is on disk, so the interrupt always lands part way
# through the run, whatever the device speed.  Returns 0 only if the run reports
# the interrupt with a non-zero iteration and <ckpt> exists; returns 1 if the run
# finished first (the exponent is too small for this device) or the interrupt
# did not leave a checkpoint.
interrupt_run() {
    local dir="$1" ckpt="$2" pid saved; shift 2
    rm -rf "$dir"; mkdir -p "$dir"; ln -s "$ROOT/kernels" "$dir/kernels"
    ( cd "$dir" && exec $RUN timeout --signal=INT --kill-after=10s 600 "$ROOT/prmers" "$@" \
        -t 1 -engine-marin -d "$DEVICE" -noask -f "$dir" > interrupt.log 2>&1 ) &
    pid=$!
    until grep -Eq 'Backup point done at iter \+ 1=[0-9]+ done' "$dir/interrupt.log" 2>/dev/null; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.01
    done
    kill -INT "$pid" 2>/dev/null || true
    wait "$pid" || true
    saved="$(sed -n 's/.*state saved at iteration \([0-9][0-9]*\).*/\1/p' "$dir/interrupt.log" | tail -n 1)"
    [ -n "$saved" ] && [ "$saved" != "0" ] && [ -f "$dir/$ckpt" ] || return 1
    echo "$saved" > "$dir/saved"
}

# Interrupt a run to leave an LL-UNSAFE checkpoint on disk.
n=0
for P in "${LL_LADDER[@]}"; do
    n=$((n + 1))
    if interrupt_run "$WORK/ll$n" "llunsafe_m_$P.ckpt" "$P" -llunsafe; then break; fi
    echo "LL-UNSAFE M$P was not interrupted part way (the run finished first); trying a larger exponent" >&2
    P=""
done
if [ -z "$P" ]; then
    echo "FAIL: no LL-UNSAFE checkpoint after interrupt"
    cat "$WORK/ll$n/interrupt.log"
    exit 1
fi
cd "$WORK/ll$n"
saved="$(cat saved)"
echo "LL_CHECKPOINT | exponent=$P saved=$saved"

# Stand-in for the PRP checkpoint of the same exponent.
echo "unrelated prp checkpoint" > "m_$P.ckpt"

set +e
$RUN timeout 900 "$ROOT/prmers" "$P" -llunsafe -engine-marin -d "$DEVICE" -noask -f "$PWD" > resume.log 2>&1
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
n=0
for Q in "${WAG_LADDER[@]}"; do
    n=$((n + 1))
    if interrupt_run "$WORK/wag$n" "wagstaff_m_$((2 * Q)).ckpt" "$Q" -wagstaff; then break; fi
    echo "Wagstaff $Q was not interrupted part way (the run finished first); trying a larger exponent" >&2
    Q=""
done
if [ -z "$Q" ]; then
    echo "FAIL: no Wagstaff checkpoint after interrupt"
    cat "$WORK/wag$n/interrupt.log"
    exit 1
fi
cd "$WORK/wag$n"
set +e
$RUN timeout 900 "$ROOT/prmers" "$Q" -wagstaff -engine-marin -d "$DEVICE" -noask -f "$PWD" > wagstaff.log 2>&1
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
