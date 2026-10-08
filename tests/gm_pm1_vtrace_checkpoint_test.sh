#!/usr/bin/env bash
# A default fresh -gm-pm1 run with B2 > B1 uses the V-trace driver. Its Stage 1
# must write the legacy `_stage1.ckpt` checkpoint on a SIGINT, and a restart
# must resume from it through the legacy path instead of starting Stage 1 over.
# GM_20011 has the factor 3922157 = 4*20011*7^2 + 1, found by Stage 1; a resumed
# residue that was not restored exactly cannot produce it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="${1:-0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export AEVUM_CARRY_WMUL=1 AEVUM_AUTOTUNE=off

cd "$WORK"
ln -s "$ROOT/kernels" kernels
ARGS=(20011 -gm-pm1 -gm-family GM -b1 30000 -b2 30500 -gm-sieve 0 -d "$DEVICE" --noask)
CKPT=gm_pm1_p20011_stage1.ckpt

# Run 1: interrupt in the V-trace Stage 1.
timeout --signal=INT --kill-after=10s 6 "$ROOT/prmers" "${ARGS[@]}" >run1.log 2>&1 || true
grep -q "Gaussian pair P-1 factoring v100.13 V-trace" run1.log || { echo "run 1 did not use the V-trace driver" >&2; exit 1; }
saved=$(sed -n 's/.*checkpoint saved with \([0-9]*\) bits remaining.*/\1/p' run1.log | head -1)
[[ -n "$saved" ]] || { echo "run 1: no Stage 1 checkpoint was reported" >&2; tail -5 run1.log >&2; exit 1; }
[[ -s "$CKPT" ]] || { echo "run 1: $CKPT missing" >&2; exit 1; }

# Run 2: the legacy path picks the checkpoint up and continues from it; interrupt again.
timeout --signal=INT --kill-after=10s 5 "$ROOT/prmers" "${ARGS[@]}" >run2.log 2>&1 || true
grep -q "legacy checkpoint detected" run2.log || { echo "run 2: legacy checkpoint not detected" >&2; exit 1; }
if grep -q "Gaussian pair P-1 factoring v100.13 V-trace" run2.log; then echo "run 2 restarted in the V-trace driver" >&2; exit 1; fi
resumed=$(sed -n 's/.*checkpoint saved with \([0-9]*\) bits remaining.*/\1/p' run2.log | head -1)
[[ -n "$resumed" ]] || { echo "run 2: no checkpoint was reported" >&2; tail -5 run2.log >&2; exit 1; }
(( resumed < saved )) || { echo "run 2 did not resume: $resumed bits remaining after run 1 left $saved" >&2; exit 1; }

# Run 3: finish; the resumed residue must still give the Stage 1 factor.
timeout --signal=INT --kill-after=10s 50 "$ROOT/prmers" "${ARGS[@]}" >run3.log 2>&1 || true
grep -q "Gaussian pair P-1 Stage 1 factor: 3922157" run3.log || { echo "run 3: Stage 1 factor 3922157 not found" >&2; tail -5 run3.log >&2; exit 1; }
[[ ! -e "$CKPT" ]] || { echo "run 3: Stage 1 checkpoint not removed" >&2; exit 1; }
echo "gm pm1 v-trace checkpoint test passed ($saved -> $resumed bits remaining)"
