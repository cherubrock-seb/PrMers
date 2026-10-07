#!/usr/bin/env bash
# Legacy (-marin) PRP resume checks.
# 1. An interrupt that arrives before the first iteration must not record that iteration as done. Interrupt M9689 (prime) at many early delays, resume each run and
# check that the verdict is still "probably prime". Error checking is off (-gerbiczli) so a
# skipped iteration is not repaired by a rollback.
# 2. A loop file whose state file is missing or truncated must not resume from a zero state:
# the run starts over and still gives the right verdict.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
P=9689
WORK="$(mktemp -d)"
RUN="${PRMERS_TEST_RUN_PREFIX:-}"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

cd "$WORK"
ln -s "$ROOT/kernels" kernels
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

early=0
for delay in 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.12; do
    rm -f "${P}prp".* results.txt "${P}_prp_result.json"
    set +e
    $RUN timeout -s INT "$delay" "$ROOT/prmers" "$P" -prp -marin -proof 0 -gerbiczli -d "$DEVICE" -noask -f "$WORK" > interrupt.log 2>&1
    set -e
    [ -f "${P}prp.loop" ] || continue
    loop="$(cat "${P}prp.loop")"
    [ "$loop" -le 1 ] || continue
    early=$((early + 1))
    echo "EARLY_INTERRUPT | delay=$delay loop=$loop"

    set +e
    $RUN timeout 120 "$ROOT/prmers" "$P" -prp -marin -proof 0 -gerbiczli -d "$DEVICE" -noask -f "$WORK" > resume.log 2>&1
    set -e
    if ! grep -q 'probably prime' resume.log; then
        echo "FAIL: resume after an interrupt before the first iteration gave the wrong verdict (loop=$loop)"
        grep -E 'PRP test|composite' resume.log || true
        exit 1
    fi
done

echo "EARLY_INTERRUPTS_CHECKED=$early"
echo "LEGACY_PRP_EARLY_INTERRUPT=PASS"

saved=""
for delay in 1.0 1.5 2.0 3.0; do
    rm -f "${P}prp".* results.txt "${P}_prp_result.json"
    set +e
    $RUN timeout -s INT "$delay" "$ROOT/prmers" "$P" -prp -marin -proof 0 -gerbiczli -d "$DEVICE" -noask -f "$WORK" > interrupt.log 2>&1
    set -e
    if [ -f "${P}prp.loop" ] && [ "$(cat "${P}prp.loop")" -gt 1 ] && [ -f "${P}prp.mers" ]; then saved=1; break; fi
done
[ -n "$saved" ] || { echo "FAIL: no mid-run state after interrupt"; cat interrupt.log; exit 1; }

for variant in missing truncated; do
    cp "${P}prp.mers" mers.keep
    cp "${P}prp.loop" loop.keep
    if [ "$variant" = missing ]; then rm -f "${P}prp.mers"; else head -c 100 mers.keep > "${P}prp.mers"; fi
    set +e
    $RUN timeout 120 "$ROOT/prmers" "$P" -prp -marin -proof 0 -gerbiczli -d "$DEVICE" -noask -f "$WORK" > "resume-$variant.log" 2>&1
    set -e
    if ! grep -q 'probably prime' "resume-$variant.log"; then
        echo "FAIL: resume with a $variant state file gave the wrong verdict"
        grep -E 'Warning|PRP test|composite' "resume-$variant.log" || true
        exit 1
    fi
    echo "STATE_FILE_${variant}=PASS"
    rm -f "${P}prp".* results.txt "${P}_prp_result.json"
    cp mers.keep "${P}prp.mers"
    cp loop.keep "${P}prp.loop"
done
echo "LEGACY_PRP_STATE_FILE=PASS"
